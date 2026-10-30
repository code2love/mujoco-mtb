import numpy as np
import mujoco

def populate_sinusoidal_terrain(
	model,
	hfield_name="terrain",
	amplitude=0.05,
	wavelength=1.0,
	flat_length=2.0,
	smooth_transition=True,
	transition_length=1.0,
):
	"""
	Populate a MuJoCo height field with sinusoidal bumpy terrain.

	The terrain is:
	  - flat at z = 0 from the height field's left edge for ``flat_length``
	  - sinusoidal after that flat region, progressing toward positive x
	  - optionally smoothly blended into the sinusoidal region

	Parameters
	----------
	model : mujoco.MjModel
		MuJoCo model.

	hfield_name : str
		Name of the hfield asset.

	amplitude : float
		Maximum sinusoidal terrain height above z = 0 [m].

		The generated sinusoid ranges from:
		    0 ... amplitude

	wavelength : float
		Wavelength of the terrain [m].

	flat_length : float
		Length of the initial flat region, measured from the height field's
		left edge toward positive x [m].

		For example, ``flat_length = 2.0`` keeps the first 2 m of the
		height field flat before transitioning to the sinusoidal terrain.

	smooth_transition : bool
		If True, gradually increases the bump amplitude after
		the flat region.

	transition_length : float
		Length over which the sinusoidal amplitude ramps from
		zero to its full value [m].

	"""

	hfield_id = mujoco.mj_name2id(
		model,
		mujoco.mjtObj.mjOBJ_HFIELD,
		hfield_name,
	)

	if hfield_id < 0:
		raise ValueError(
			f'Hfield "{hfield_name}" not found.'
		)

	nrow = model.hfield_nrow[hfield_id]
	ncol = model.hfield_ncol[hfield_id]
	adr = model.hfield_adr[hfield_id]

	size_x = model.hfield_size[hfield_id, 0]
	size_y = model.hfield_size[hfield_id, 1]
	max_elevation = model.hfield_size[hfield_id, 2]

	if amplitude < 0:
		raise ValueError("amplitude must be >= 0.")

	if amplitude > max_elevation:
		raise ValueError(
			f"amplitude ({amplitude} m) exceeds the "
			f"hfield maximum elevation ({max_elevation} m)."
		)

	if wavelength <= 0:
		raise ValueError("wavelength must be > 0.")

	if flat_length < 0:
		raise ValueError("flat_length must be >= 0.")

	if transition_length < 0:
		raise ValueError("transition_length must be >= 0.")

	# ------------------------------------------------------------
	# Height-field coordinates
	# ------------------------------------------------------------

	x = np.linspace(-size_x, size_x, ncol)
	y = np.linspace(-size_y, size_y, nrow)

	X, Y = np.meshgrid(x, y)

	# ------------------------------------------------------------
	# Distance from the edge of the central flat region
	# ------------------------------------------------------------

	flat_end_x = x[0] + flat_length
	d = np.maximum(X - flat_end_x, 0.0)

	# ------------------------------------------------------------
	# Non-negative sinusoidal terrain
	#
	# z = 0       at the trough
	# z = A       at the crest
	#
	# Starting with 1 - cos(...) also means the terrain begins
	# with zero height and zero slope at d = 0.
	# ------------------------------------------------------------

	phase = 2.0 * np.pi * d / wavelength

	z_sine = (
		0.5
		* amplitude
		* (1.0 - np.cos(phase))
	)

	# Keep the central region exactly flat.
	z_sine[d == 0.0] = 0.0

	# ------------------------------------------------------------
	# Optional smooth amplitude transition
	# ------------------------------------------------------------

	if smooth_transition and transition_length > 0:

		t = np.clip(
			d / transition_length,
			0.0,
			1.0,
		)

		# Smoothstep:
		# 0 -> 1 with zero derivative at both ends.
		envelope = t**2 * (3.0 - 2.0 * t)

		z = envelope * z_sine

	else:
		z = z_sine

	# ------------------------------------------------------------
	# Convert physical heights to MuJoCo's normalized [0, 1]
	# representation.
	# ------------------------------------------------------------

	height_data = z / max_elevation

	height_data = np.clip(
		height_data,
		0.0,
		1.0,
	)

	# Copy into MuJoCo model.
	model.hfield_data[
		adr:adr + nrow * ncol
	] = height_data.ravel()

	return x, y, z


def populate_rectangular_bump(
	model,
	height,
	length,
	distance_from_left,
	hfield_name="terrain",
):
	"""Raise a full-width rectangular section of the existing height field.

	The bump starts ``distance_from_left`` meters from the height field's left
	edge, extends ``length`` meters along positive x, and adds ``height`` meters
	to the existing terrain across the full y axis.
	"""
	hfield_id = mujoco.mj_name2id(
		model,
		mujoco.mjtObj.mjOBJ_HFIELD,
		hfield_name,
	)
	if hfield_id < 0:
		raise ValueError(f'Hfield "{hfield_name}" not found.')
	if height < 0.0:
		raise ValueError("height must be >= 0")
	if length <= 0.0:
		raise ValueError("length must be > 0")
	if distance_from_left < 0.0:
		raise ValueError("distance_from_left must be >= 0")

	nrow = model.hfield_nrow[hfield_id]
	ncol = model.hfield_ncol[hfield_id]
	data_start = model.hfield_adr[hfield_id]
	size_x = float(model.hfield_size[hfield_id, 0])
	max_elevation = float(model.hfield_size[hfield_id, 2])
	field_length = 2.0 * size_x
	if distance_from_left + length > field_length:
		raise ValueError("bump extends beyond the right edge of the height field")
	if max_elevation <= 0.0:
		raise ValueError("height field maximum elevation must be greater than zero")

	x = np.linspace(-size_x, size_x, ncol)
	y = np.linspace(
		-model.hfield_size[hfield_id, 1],
		model.hfield_size[hfield_id, 1],
		nrow,
	)
	z = (
		model.hfield_data[data_start:data_start + nrow * ncol]
		.reshape(nrow, ncol)
		.copy()
		* max_elevation
	)

	bump_start = -size_x + distance_from_left
	bump_end = bump_start + length
	bump_columns = (x >= bump_start) & (x <= bump_end)
	if not np.any(bump_columns):
		bump_columns[np.argmin(np.abs(x - (bump_start + bump_end) / 2.0))] = True

	if np.any(z[:, bump_columns] + height > max_elevation):
		raise ValueError("bump would exceed the height field maximum elevation")
	z[:, bump_columns] += height

	model.hfield_data[data_start:data_start + nrow * ncol] = (
		z / max_elevation
	).ravel()
	return x, y, z
