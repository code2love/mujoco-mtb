import numpy as np
import mujoco


class PedalSlipDetector:
	"""Detect foot lift-off from the normal force at each pedal connection.

	Positive normal force supports the foot. A force below the negative
	``normal_force_threshold`` means the foot is pulling on the pedal and is
	reported as pedal slip.
	"""

	def __init__(
		self,
		model,
		normal_force_threshold=0.0,
		left_connection_name="rider_left_foot_connection",
		right_connection_name="rider_right_foot_connection",
	):
		if normal_force_threshold < 0.0:
			raise ValueError("normal_force_threshold must be non-negative")

		self.normal_force_threshold = float(normal_force_threshold)
		self._connections = {}
		for side, name in (
			("left", left_connection_name),
			("right", right_connection_name),
		):
			equality_id = mujoco.mj_name2id(
				model,
				mujoco.mjtObj.mjOBJ_EQUALITY,
				name,
			)
			if equality_id < 0:
				raise ValueError(f'Connect equality "{name}" was not found in the model')
			if model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_CONNECT:
				raise ValueError(f'Equality "{name}" must be a connect constraint')

			pedal_site_id = int(model.eq_obj2id[equality_id])
			self._connections[side] = (int(equality_id), pedal_site_id)

		self.normal_forces = {"left": np.nan, "right": np.nan}
		self.slipping = {"left": False, "right": False}

	def update(self, data):
		"""Update force measurements and return slip flags by pedal side."""
		active_rows = slice(0, data.nefc)
		row_types = data.efc_type[active_rows]
		row_ids = data.efc_id[active_rows]
		row_forces = data.efc_force[active_rows]

		for side, (equality_id, pedal_site_id) in self._connections.items():
			rows = (row_types == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (
				row_ids == equality_id
			)
			world_force = row_forces[rows]
			if world_force.size == 0:
				self.normal_forces[side] = np.nan
				self.slipping[side] = False
				continue
			if world_force.size != 3:
				raise RuntimeError(
					f"{side.capitalize()} pedal connect produced "
					f"{world_force.size} force components; expected 3"
				)

			pedal_rotation = data.site_xmat[pedal_site_id].reshape(3, 3)
			pedal_local_force = pedal_rotation.T @ world_force
			normal_force = float(pedal_local_force[2])
			self.normal_forces[side] = normal_force
			self.slipping[side] = normal_force < -self.normal_force_threshold

		return self.slipping.copy()
