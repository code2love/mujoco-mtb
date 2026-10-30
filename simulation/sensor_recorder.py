import numpy as np
import mujoco


class SensorRecorder:
	def __init__(
		self,
		model,
		initial_duration=10.0,
		record_controls=True,
		record_connect_forces=True,
	):
		self.model = model
		self.record_controls = record_controls
		self.record_connect_forces = record_connect_forces

		# ------------------------------------------------------------
		# Cache sensor names and their slices in data.sensordata.
		# This lookup is done only once.
		# ------------------------------------------------------------

		self._sensor_slices = {}

		for sensor_id in range(model.nsensor):
			name = mujoco.mj_id2name(
				model,
				mujoco.mjtObj.mjOBJ_SENSOR,
				sensor_id,
			)

			adr = int(model.sensor_adr[sensor_id])
			dim = int(model.sensor_dim[sensor_id])

			self._sensor_slices[name] = slice(
				adr,
				adr + dim,
			)

		# ------------------------------------------------------------
		# Cache actuator names as well.
		# ------------------------------------------------------------

		self._actuator_ids = {}

		for actuator_id in range(model.nu):
			name = mujoco.mj_id2name(
				model,
				mujoco.mjtObj.mjOBJ_ACTUATOR,
				actuator_id,
			)

			self._actuator_ids[name] = actuator_id

		self._connect_ids = {}
		self._connect_columns = {}
		for equality_id in range(model.neq):
			if model.eq_type[equality_id] != mujoco.mjtEq.mjEQ_CONNECT:
				continue
			name = mujoco.mj_id2name(
				model,
				mujoco.mjtObj.mjOBJ_EQUALITY,
				equality_id,
			)
			if name is not None:
				self._connect_columns[equality_id] = len(self._connect_ids)
				self._connect_ids[name] = equality_id

		# ------------------------------------------------------------
		# Preallocate storage.
		# ------------------------------------------------------------

		self._capacity = max(
			1,
			int(
				np.ceil(
					initial_duration
					/ model.opt.timestep
				)
			),
		)

		self._size = 0

		self._time = np.empty(
			self._capacity,
			dtype=np.float64,
		)

		self._sensor_data = np.empty(
			(
				self._capacity,
				model.nsensordata,
			),
			dtype=np.float64,
		)

		if record_controls:
			self._ctrl = np.empty(
				(
					self._capacity,
					model.nu,
				),
				dtype=np.float64,
			)
		else:
			self._ctrl = None

		if record_connect_forces:
			self._connect_force_data = np.zeros(
				(self._capacity, len(self._connect_ids), 3),
				dtype=np.float64,
			)
		else:
			self._connect_force_data = None

	def _grow(self):
		"""
		Double the allocated storage.
		"""

		new_capacity = 2 * self._capacity

		new_time = np.empty(
			new_capacity,
			dtype=np.float64,
		)

		new_sensor_data = np.empty(
			(
				new_capacity,
				self.model.nsensordata,
			),
			dtype=np.float64,
		)

		new_time[:self._size] = self._time[:self._size]

		new_sensor_data[:self._size] = (
			self._sensor_data[:self._size]
		)

		self._time = new_time
		self._sensor_data = new_sensor_data

		if self.record_controls:
			new_ctrl = np.empty(
				(
					new_capacity,
					self.model.nu,
				),
				dtype=np.float64,
			)

			new_ctrl[:self._size] = (
				self._ctrl[:self._size]
			)

			self._ctrl = new_ctrl

		if self.record_connect_forces:
			new_connect_force_data = np.zeros(
				(new_capacity, len(self._connect_ids), 3),
				dtype=np.float64,
			)
			new_connect_force_data[:self._size] = (
				self._connect_force_data[:self._size]
			)
			self._connect_force_data = new_connect_force_data

		self._capacity = new_capacity

	def record(self, data):
		"""
		Record one simulation sample.
		"""

		if self._size == self._capacity:
			self._grow()

		i = self._size

		self._time[i] = data.time

		self._sensor_data[i, :] = (
			data.sensordata
		)

		if self.record_controls:
			self._ctrl[i, :] = data.ctrl

		if self.record_connect_forces:
			self._connect_force_data[i, :, :] = 0.0
			active_connect_rows = (
				data.efc_type[:data.nefc]
				== mujoco.mjtConstraint.mjCNSTR_EQUALITY
			)
			active_connect_ids = data.efc_id[:data.nefc][active_connect_rows]
			active_connect_forces = data.efc_force[:data.nefc][active_connect_rows]

			for equality_id, column in self._connect_columns.items():
				forces = active_connect_forces[active_connect_ids == equality_id]
				if forces.size not in (0, 3):
					raise RuntimeError(
						f"Connect equality {equality_id} produced {forces.size} force rows; expected 3"
					)
				if forces.size:
					self._connect_force_data[i, column, :] = forces

		self._size += 1

	def clear(self):
		"""
		Discard all recorded samples without reallocating memory.
		"""

		self._size = 0

	def sensor(self, name):
		"""
		Return the complete time series of a sensor.

		Scalar sensors:
		    shape = (N,)

		Vector sensors:
		    shape = (N, dim)
		"""

		if name not in self._sensor_slices:
			raise KeyError(
				f'Unknown sensor "{name}". '
				f"Available sensors: "
				f"{list(self._sensor_slices)}"
			)

		sensor_slice = self._sensor_slices[name]

		values = self._sensor_data[
			:self._size,
			sensor_slice,
		]

		if values.shape[1] == 1:
			return values[:, 0]

		return values

	def control(self, actuator_name):
		"""
		Return the recorded control signal for an actuator.
		"""

		if not self.record_controls:
			raise RuntimeError(
				"Control recording is disabled."
			)

		if actuator_name not in self._actuator_ids:
			raise KeyError(
				f'Unknown actuator "{actuator_name}".'
			)

		actuator_id = self._actuator_ids[
			actuator_name
		]

		return self._ctrl[
			:self._size,
			actuator_id,
		]

	def connect_force(self, connect_name):
		"""Return recorded 3D force vectors for a named connect equality."""
		if not self.record_connect_forces:
			raise RuntimeError("Connect-force recording is disabled.")
		if connect_name not in self._connect_ids:
			raise KeyError(
				f'Unknown connect equality "{connect_name}". '
				f"Available connect equalities: {list(self._connect_ids)}"
			)
		column = self._connect_columns[self._connect_ids[connect_name]]
		return self._connect_force_data[:self._size, column, :]

	def timeseries(self, sensor_name):
		"""
		Return (time, sensor_values).
		"""

		return (
			self.time,
			self.sensor(sensor_name),
		)

	def __getitem__(self, sensor_name):
		return self.sensor(sensor_name)

	def __len__(self):
		return self._size

	@property
	def time(self):
		return self._time[:self._size]

	@property
	def sensor_names(self):
		return tuple(self._sensor_slices.keys())

	@property
	def actuator_names(self):
		return tuple(self._actuator_ids.keys())

	@property
	def connect_names(self):
		return tuple(self._connect_ids.keys())

	@property
	def sample_period(self):
		if self._size < 2:
			return np.nan

		return np.mean(
			np.diff(self.time)
		)

	@property
	def sample_rate(self):
		dt = self.sample_period

		if not np.isfinite(dt):
			return np.nan

		return 1.0 / dt