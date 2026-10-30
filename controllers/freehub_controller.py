import numpy as np
import mujoco

class FreehubController:
    def __init__(
        self,
        model,
        freehub_joint_name="freehub_rel",
        clutch_actuator_name="freehub_clutch",
        rear_wheel_joint_name="rear_wheel_spin",
        clutch_stiffness=5000.0,
        freewheel_drag=0.1,
        max_clutch_torque=10000.0,
        engagement_speed_threshold=0.0,
        release_torque_threshold=0.5,
    ):
        # ============================================================
        # MuJoCo joint / actuator addresses
        # ============================================================

        # freehub_rel describes the rotation of the cassette relative
        # to the rear wheel:
        #
        #   theta_cassette
        #       = theta_wheel + theta_freehub_rel
        #
        # Therefore:
        #
        #   omega_freehub_rel
        #       = omega_cassette - omega_wheel
        #
        freehub_joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            freehub_joint_name,
        )

        self.freehub_qpos_adr = model.jnt_qposadr[freehub_joint_id]
        self.freehub_dof_adr = model.jnt_dofadr[freehub_joint_id]

        # Rear-wheel angular velocity is required to reconstruct the
        # absolute cassette angular velocity.
        rear_wheel_joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            rear_wheel_joint_name,
        )

        self.rear_wheel_dof_adr = model.jnt_dofadr[rear_wheel_joint_id]

        # Actuator applies an internal torque between cassette and hub
        # through the relative freehub joint.
        self.clutch_actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            clutch_actuator_name,
        )

        # ============================================================
        # Freehub parameters
        # ============================================================

        # Torsional stiffness of the engaged freehub.
        self.clutch_stiffness = clutch_stiffness

        # Small viscous drag acting on the cassette while the freehub
        # is disengaged. In this implementation the drag depends on
        # the absolute cassette angular velocity rather than on the
        # cassette-wheel relative velocity.
        self.freewheel_drag = freewheel_drag

        # Maximum torque that can be transmitted through the freehub.
        self.max_clutch_torque = max_clutch_torque

        # The freehub engages when the cassette starts rotating faster
        # than the rear wheel by more than this threshold.
        self.engagement_speed_threshold = engagement_speed_threshold

        # The freehub disengages once maintaining engagement would
        # require a sufficiently large torque in the reverse direction.
        self.release_torque_threshold = release_torque_threshold

        # ============================================================
        # Discrete freehub state
        # ============================================================

        self.engaged = False

        # Relative cassette-wheel angle at the instant of engagement.
        #
        # While engaged, the freehub behaves as a torsional spring
        # around this position.
        self.engagement_angle = 0.0


    def reset(self, data):
        """
        Reset the discrete freehub state.

        Call this after mj_resetData() or whenever the simulation is
        restarted.
        """
        self.engaged = False
        self.engagement_angle = data.qpos[self.freehub_qpos_adr]


    # ================================================================
    # Discrete engagement-state update
    #
    # Call exactly ONCE per complete simulation timestep, outside
    # mjcb_control. This keeps the discrete state independent of the
    # selected MuJoCo integrator (Euler, implicit, RK4, ...).
    # ================================================================

    def update_state(self, data):

        relative_angle = data.qpos[self.freehub_qpos_adr]

        # Relative cassette-wheel angular velocity:
        #
        #   omega_relative = omega_cassette - omega_wheel
        #
        relative_angular_velocity = data.qvel[self.freehub_dof_adr]

        if not self.engaged:

            # The freehub engages when the cassette attempts to outrun
            # the rear wheel.
            if relative_angular_velocity > self.engagement_speed_threshold:

                self.engaged = True

                # Store the current relative angle as the zero-deflection
                # position of the engaged freehub.
                #
                # This allows engagement at any cassette-wheel phase and
                # avoids forcing the relative coordinate back to zero.
                self.engagement_angle = relative_angle

        else:

            # Relative angular displacement accumulated since engagement.
            clutch_deflection = (
                relative_angle
                - self.engagement_angle
            )

            # Torsional spring torque generated by the engaged freehub.
            spring_torque = (
                self.clutch_stiffness
                * clutch_deflection
            )

            # The actuator must oppose positive cassette motion relative
            # to the wheel. Therefore the actual clutch torque has the
            # opposite sign to the spring torque.
            required_clutch_torque = -spring_torque

            # A real freehub cannot transfer torque from the rear wheel
            # back into the cassette.
            #
            # If keeping the clutch engaged would require a positive
            # actuator torque, the wheel is attempting to overrun the
            # cassette and the freehub is released.
            if required_clutch_torque > self.release_torque_threshold:
                self.engaged = False


    # ================================================================
    # Continuous clutch torque law
    #
    # This function is intended for mjcb_control and may therefore be
    # evaluated multiple times during one simulation timestep, e.g. at
    # the intermediate stages of RK4.
    #
    # It does NOT modify any discrete state. `engaged` and
    # `engagement_angle` remain constant throughout a complete MuJoCo
    # integration step.
    # ================================================================

    def __call__(self, model, data):

        relative_angle = data.qpos[self.freehub_qpos_adr]

        # Relative cassette-wheel speed:
        #
        #   omega_relative = omega_cassette - omega_wheel
        #
        relative_angular_velocity = data.qvel[self.freehub_dof_adr]

        rear_wheel_angular_velocity = data.qvel[self.rear_wheel_dof_adr]

        # Since the cassette body is a child of the rear-wheel body:
        #
        #   omega_cassette
        #       = omega_wheel + omega_relative
        #
        cassette_angular_velocity = (
            rear_wheel_angular_velocity
            + relative_angular_velocity
        )

        # ------------------------------------------------------------
        # Freewheeling
        # ------------------------------------------------------------

        if not self.engaged:

            # When disengaged, no driving torque is transmitted through
            # the freehub. Only a small viscous drag remains.
            #
            # In this implementation the drag is proportional to the
            # absolute cassette angular velocity.
            freewheel_drag_torque = (
                -self.freewheel_drag
                * cassette_angular_velocity
            )

            data.ctrl[self.clutch_actuator_id] = freewheel_drag_torque
            return

        # ------------------------------------------------------------
        # Engaged freehub
        # ------------------------------------------------------------

        # Relative displacement from the angle at which the freehub
        # engaged.
        clutch_deflection = (
            relative_angle
            - self.engagement_angle
        )

        # The engaged freehub is represented as a torsional spring.
        spring_torque = (
            self.clutch_stiffness
            * clutch_deflection
        )

        # Reaction torque acting on the relative cassette-wheel joint.
        required_clutch_torque = -spring_torque

        # The clutch may only transmit torque in the driving direction.
        #
        # The clipping has two purposes:
        #   1. prevent reverse torque transmission,
        #   2. limit the maximum freehub torque.
        #
        # This restriction is applied continuously, including at
        # intermediate RK4 integration stages.
        clutch_torque = np.clip(
            required_clutch_torque,
            -self.max_clutch_torque,
            0.0,
        )

        data.ctrl[self.clutch_actuator_id] = clutch_torque

