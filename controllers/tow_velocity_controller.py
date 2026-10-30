import numpy as np
import mujoco


class PIDTowController:
	def __init__(
		self,
		model,
		target_velocity=5.0,
		kp=200.0,
		ki=0.0,
		kd=0.0,
		freq_cutoff=10.0,
		force_limit=None,
		integral_limit=None,
	):
		self.force_limit = force_limit
		self.dt = model.opt.timestep
		self.actuator_id = mujoco.mj_name2id(
			model,
			mujoco.mjtObj.mjOBJ_ACTUATOR,
			"tow_force",
		)
		if self.actuator_id < 0:
			raise ValueError("Tow actuator 'tow_force' was not found in the model")

		self.target_velocity = target_velocity
		self.kp = kp
		self.ki = ki
		self.kd = kd
		if freq_cutoff <= 0.0:
			raise ValueError("freq_cutoff must be greater than zero")
		self.freq_cutoff = freq_cutoff
		self.integral_limit = integral_limit

		self.integral = 0.0
		self.derivative = 0.0
		self.last_velocity = None
		self._enabled = True

		# Translational bike DOF.
		frame_x_joint_id = mujoco.mj_name2id(
			model,
			mujoco.mjtObj.mjOBJ_JOINT,
			"frame_x",
		)

		if frame_x_joint_id < 0:
			raise ValueError("Translational joint 'frame_x' was not found in the model")

		self.frame_x_dof = model.jnt_dofadr[
			frame_x_joint_id
		]

	@property
	def enabled(self):
		return self._enabled

	@enabled.setter
	def enabled(self, value):
		value = bool(value)
		if value and not self._enabled:
			self.last_velocity = None
			self.derivative = 0.0
		self._enabled = value

	def reset(self, data):
		self.integral = 0.0
		self.derivative = 0.0
		self.last_velocity = None

	def __call__(self, model, data):
		if not self.enabled:
			data.ctrl[self.actuator_id] = 0.0
			return

		velocity = data.qvel[self.frame_x_dof]

		error = self.target_velocity - velocity

		self.integral += error * self.dt
		if self.integral_limit is not None:
			self.integral = np.clip(
				self.integral,
				-self.integral_limit,
				self.integral_limit,
			)

		raw_derivative = 0.0
		if self.last_velocity is not None:
			raw_derivative = -(velocity - self.last_velocity) / self.dt

		filter_alpha = 1.0 - np.exp(
			-2.0 * np.pi * self.freq_cutoff * self.dt
		)
		self.derivative += filter_alpha * (
			raw_derivative - self.derivative
		)
		self.last_velocity = velocity

		force = (
			self.kp * error
			+ self.ki * self.integral
			+ self.kd * self.derivative
		)

		if self.force_limit is not None:
			force = np.clip(
				force,
				-self.force_limit,
				self.force_limit,
			)

		data.ctrl[self.actuator_id] = force

TowVelocityController = PIDTowController