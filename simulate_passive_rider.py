import time
import numpy as np
import mujoco
import mujoco.viewer
from controllers import FreehubController, TowVelocityController
from simulation import (
    PedalSlipDetector,
    SensorRecorder,
    populate_rectangular_bump,
    populate_sinusoidal_terrain,
    solve_rider_initial_pose,
)


model = mujoco.MjModel.from_xml_path("mtb_model/model_four_link.xml")

data = mujoco.MjData(model) 

# populate_sinusoidal_terrain(
# 	model,
# 	hfield_name="terrain",
# 	amplitude=0.1,
# 	wavelength=2.5,
#     flat_length=25.0,
# 	smooth_transition=True,
# 	transition_length=1.5,
# )

populate_rectangular_bump(
    model,
    height=0.25,
    length=0.25,
    distance_from_left=30.0,
    hfield_name="terrain",
)

solve_rider_initial_pose(
    model,
    data
)

recorder = SensorRecorder(
    model,
    initial_duration=20.0,
    record_controls=True,
)

pedal_slip_detector = PedalSlipDetector(
    model,
    normal_force_threshold=0.0,
)

tow_controller = TowVelocityController(
    model,
    target_velocity=5.0,
    kp=200.0,
    ki=0.0,
    kd=0.0,
)

freehub_controller = FreehubController(
    model
)

mujoco.set_mjcb_control(freehub_controller) 

total_mass = np.sum(model.body_mass)

rider_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "rider_hip"
)

rider_mass = model.body_subtreemass[rider_id]

print(f"Total model mass: {total_mass:.3f} kg")

print(f"Rider mass: {rider_mass:.3f} kg")

print(f"Bike mass: {total_mass - rider_mass:.3f} kg")

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Get ID of the XML camera
    cam_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_CAMERA,
        "bike_side"
    )

    # Select the model-defined camera
    with viewer.lock():
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id

    viewer.sync()

    snapshot = None

    t = [data.time]

    pedal_slip_detected = [pedal_slip_detector.slipping]
    
    start = time.time()

    while viewer.is_running():

        freehub_controller.update_state(data)

        tow_controller(model, data)

        mujoco.mj_step(model, data)

        recorder.record(data)

        pedal_slip_detected.append(pedal_slip_detector.update(data))

        t.append(data.time)
    
        viewer.sync()

        elapsed = time.time() - start

        if elapsed < data.time:
            time.sleep(data.time - elapsed)

        if recorder["frame_position"][-1,0] >= 10.0 and snapshot is None:
            tow_controller.enabled = False

            snapshot = mujoco.MjData(model)

            mujoco.mj_copyData(
                snapshot,
                model,
                data,
            )

        if recorder["frame_position"][-1,0] >= 50.0 or data.time >= 20.0:
            break

    print("Simulation finished. Press ESC or close the window to exit.")