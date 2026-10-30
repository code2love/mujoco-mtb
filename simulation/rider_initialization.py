import numpy as np
import mujoco
from scipy.optimize import least_squares


def solve_rider_initial_pose(model, data):

	# ------------------------------------------------------------
	# Rider coordinates which are allowed to change.
	# ------------------------------------------------------------

	joint_names = [
		"rider_x",
		"rider_z",
		"rider_pitch",

		"rider_right_shoulder_pitch",
		"rider_right_shoulder_yaw",
		"rider_right_elbow",

		"rider_left_shoulder_pitch",
		"rider_left_shoulder_yaw",
		"rider_left_elbow",

		"rider_right_hip_joint",
		"rider_right_knee",

		"rider_left_hip_joint",
		"rider_left_knee",
	]

	def qadr(model, name):
		jid = mujoco.mj_name2id(
			model,
			mujoco.mjtObj.mjOBJ_JOINT,
			name,
		)

		return model.jnt_qposadr[jid]

	def site_id(model, name):
		return mujoco.mj_name2id(
			model,
			mujoco.mjtObj.mjOBJ_SITE,
			name,
		)

	qadrs = np.array([
		qadr(model, name)
		for name in joint_names
	])

	# ------------------------------------------------------------
	# Site pairs which must coincide.
	# ------------------------------------------------------------

	site_pairs = [
		(
			site_id(model, "rider_right_wrist"),
			site_id(model, "right_hand_grip"),
		),
		(
			site_id(model, "rider_left_wrist"),
			site_id(model, "left_hand_grip"),
		),
		(
			site_id(model, "rider_right_foot_contact"),
			site_id(model, "right_pedal_center"),
		),
		(
			site_id(model, "rider_left_foot_contact"),
			site_id(model, "left_pedal_center"),
		),
	]

	# Initial guess.
	x0 = data.qpos[qadrs].copy()

	# Small regularization keeps the solution close to initial posture.
	regularization = 1e-6

	def residual(x):

		data.qpos[qadrs] = x

		mujoco.mj_forward(
			model,
			data,
		)

		errors = []

		for rider_site, bike_site in site_pairs:

			error = (
				data.site_xpos[rider_site]
				- data.site_xpos[bike_site]
			)

			errors.extend(error)

		# Prefer solutions close to the original pose.
		errors.extend(
			np.sqrt(regularization)
			* (x - x0)
		)

		return np.asarray(errors)

	result = least_squares(
		residual,
		x0,
		xtol=1e-12,
		ftol=1e-12,
		gtol=1e-12,
		max_nfev=1000,
	)

	data.qpos[qadrs] = result.x

	# Start stationary.
	data.qvel[:] = 0.0

	mujoco.mj_forward(
		model,
		data,
	)

	print(
		"Maximum attachment error:",
		np.max(np.abs(residual(result.x)[:12])),
		"m",
	)

	return result