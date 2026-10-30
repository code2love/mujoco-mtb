MTB modular MuJoCo model
========================

Model variants
--------------
Load either XML as the model entry point:

- model_single_pivot.xml: direct single-pivot rear suspension.
- model_four_link.xml: Horst-link rear suspension with a rocker-driven shock.

Both variants define the bike-specific frame and mounting geometry, then
include reusable components and shared systems. Include paths are relative to
the model XML files.

Module layout
-------------
common/defaults.xml
	Shared MuJoCo defaults.

components/
	crankset.xml
	front_fork_160.xml
	front_wheel_29.xml
	rear_wheel_29_freehub.xml
	rear_shock_210x50.xml
	rider_passive.xml
	Reusable bike and passive-rider assemblies.

suspension/
	single_pivot.xml
	four_link.xml
	Alternative rear-suspension assemblies.

systems/
	actuators.xml
	drivetrain.xml
	rider_actuation.xml
	rider_connections.xml
	sensors.xml
	suspension_constraints.xml
	Shared actuators, drivetrain, rider, sensor, and shock-connection logic.

analysis/
	kickback_constraints.xml
	Optional pedal-kickback analysis constraints; included by both variants,
	but its equality constraints are inactive by default.

Rider controls and sensors
-------------------------
- Rider hand and foot sites are connected to the grips and pedals in
	systems/rider_connections.xml. The leg spring-dampers are defined there too.
- systems/rider_actuation.xml provides an elbow-pair motor, right/left leg
	preload motors, and independent right/left knee torque motors. Each knee
	motor has unit gear and a control range of -250 to 250 N m.
- systems/sensors.xml reports frame motion, suspension travel and velocity,
	wheel motion and axle positions, drivetrain state, and rider elbow, shoulder
	pitch/yaw, hip, and knee joint positions. It also reports tow force.
- Hinge joint position sensor values are in radians; XML joint ranges use
	degrees because each entry-point compiler sets angle="degree".

Suspension interface
--------------------
Both suspension modules preserve these names for shared drivetrain, shock,
sensor, and analysis systems:

	rear_shock_swingarm_eye
	chain_rear_center
	rear_wheel_center
	rear_wheel_spin
	freehub_rel

Preserve these names when adding or replacing a suspension layout.
