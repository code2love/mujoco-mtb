import time
import numpy as np
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer

shock_targets = np.linspace(0.0, 0.050, 51)
settle_time = 1.0
fast_simulation = True


def analyze_model(model_name, xml_path):
	model = mujoco.MjModel.from_xml_path(xml_path)
	data = mujoco.MjData(model)
	dt = model.opt.timestep
	settle_steps = int(settle_time / dt)

	def eq(name):
		return mujoco.mj_name2id(
			model,
			mujoco.mjtObj.mjOBJ_EQUALITY,
			name
		)

	def qadr(name):
		jid = mujoco.mj_name2id(
			model,
			mujoco.mjtObj.mjOBJ_JOINT,
			name
		)
		return model.jnt_qposadr[jid]

	# Analysis constraints defined in the XML.
	eq_frame = eq("analysis_fix_main_frame")
	eq_shock = eq("analysis_rear_shock_position")
	eq_wheel = eq("analysis_fix_rear_wheel")
	eq_freehub = eq("analysis_lock_freehub")

	# Disconnect rider from the bike for the purely kinematic analysis.
	for name in (
		"rider_right_hand_connection",
		"rider_left_hand_connection",
		"rider_right_foot_connection",
		"rider_left_foot_connection",
	):
		data.eq_active[eq(name)] = 0

	data.eq_active[eq_frame] = 1
	data.eq_active[eq_shock] = 1
	data.eq_active[eq_wheel] = 1
	data.eq_active[eq_freehub] = 1

	# Remove ground and gravity effects.
	model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
	model.opt.gravity[:] = 0.0

	shock_qadr = qadr("rear_shock_compression")
	crank_qadr = qadr("crank_spin")
	rear_site = mujoco.mj_name2id(
		model,
		mujoco.mjtObj.mjOBJ_SITE,
		"rear_wheel_center"
	)

	with mujoco.viewer.launch_passive(model, data) as viewer:
		wheel_x = []
		wheel_z = []
		crank_angle = []
		shock_pos = []

		cam_id = mujoco.mj_name2id(
			model,
			mujoco.mjtObj.mjOBJ_CAMERA,
			"bike_side"
		)

		with viewer.lock():
			viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
			viewer.cam.fixedcamid = cam_id

		mujoco.mj_forward(model, data)
		viewer.sync()

		for target in shock_targets:
			model.eq_data[eq_shock, 0] = target

			for _ in range(settle_steps):
				start = time.time()

				if model.nu:
					data.ctrl[:] = 0.0

				mujoco.mj_step(model, data)
				viewer.sync()

				elapsed = time.time() - start
				if not fast_simulation and elapsed < dt:
					time.sleep(dt - elapsed)

			mujoco.mj_forward(model, data)

			shock_pos.append(data.qpos[shock_qadr])
			wheel_x.append(data.site_xpos[rear_site, 0])
			wheel_z.append(data.site_xpos[rear_site, 2])
			crank_angle.append(data.qpos[crank_qadr])

	shock_pos = np.asarray(shock_pos)
	wheel_x = np.asarray(wheel_x)
	wheel_z = np.asarray(wheel_z)
	wheel_x_travel = wheel_x - wheel_x[0]
	wheel_z_travel = wheel_z - wheel_z[0]
	crank_angle = np.unwrap(np.asarray(crank_angle))
	kickback = -np.rad2deg(crank_angle - crank_angle[0])

	print(f"{model_name}:")
	print(f"  Rear-wheel vertical travel: {1000 * wheel_z_travel[-1]:.1f} mm")
	print(f"  Rear-wheel horizontal shift: {1000 * wheel_x_travel[-1]:.1f} mm")
	print(f"  Pedal kickback:              {kickback[-1]:.2f} deg")

	return {
		"shock_pos": shock_pos,
		"wheel_x_travel": wheel_x_travel,
		"wheel_z_travel": wheel_z_travel,
		"kickback": kickback,
	}


models = (
	("model_four_link", "mtb_model/model_four_link.xml", "--"),
	("model_single_pivot", "mtb_model/model_single_pivot.xml", "-."),
)
results = {
	name: analyze_model(name, xml_path)
	for name, xml_path, _ in models
}

plot_specs = (
	("wheel_z_travel", "kickback", 1000, 1, "Rear-wheel vertical travel [mm]", "Pedal kickback [deg]", None),
	("shock_pos", "wheel_z_travel", 1000, 1000, "Shock compression [mm]", "Rear-wheel vertical travel [mm]", None),
	("wheel_x_travel", "wheel_z_travel", 1000, 1000, "Rear-axle horizontal movement [mm]", "Rear-axle vertical movement [mm]", "Rear axle path"),
)

for x_key, y_key, x_scale, y_scale, x_label, y_label, title in plot_specs:
	plt.figure()
	for name, _, line_style in models:
		result = results[name]
		plt.plot(
			x_scale * result[x_key],
			y_scale * result[y_key],
			linestyle=line_style,
			label=name,
		)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	if title is not None:
		plt.title(title)
		plt.axis("equal")
	plt.legend()
	plt.grid(True)
	plt.tight_layout()

plt.show(block=True)