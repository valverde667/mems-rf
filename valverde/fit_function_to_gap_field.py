# The fields are simulated in a different script "gap_field_function.py" and then
# loaded here. A function is found to fit a function to the field.

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as SC


# Useful constants
kV = 1000.0
keV = 1000.0
mm = 1.0e-3

# Set geomtry values
gap_width = 2 * mm

# ------------------------------------------------------------------------------
#    Load Field and mesh arrays
# The fields are loaded from .npy fields that were created in a different script
# that models the geometries and generates the various fields. Here we load in
# the electric field in z Ez(r=0, z) and the corresponding zmesh that created
# the field. The current geoemetry of the conductors limits the field to inside
# the gap with just a slight leakage of the gap. This field is isolated.
# ------------------------------------------------------------------------------
Ez0 = np.load("field_arrays.npy")
zmesh = np.load("zmesh.npy")
gap_centers = np.load("gap_centers.npy")

# Visualize field
fig, ax = plt.subplots()
for cent in gap_centers:
    right = cent + gap_width / 2
    left = cent - gap_width / 2
    ax.axvline(left / mm, c="k", ls="--")
    ax.axvline(right / mm, c="k", ls="--")
ax.plot(zmesh / mm, Ez0)
ax.set_xlabel("z [mm]")
ax.set_ylabel(r"$E_z(r=0, z)$ [V/m]")
plt.show()

# ------------------------------------------------------------------------------
#     Isolate gap field
# Here the field for a single gap is isolated. The arrays up to the midpoint
# between to two gaps are first extracted since at this point the field is sure
# to be zero. Then, the field locations for where the field is greater than .001%
# of Emax is selected. These indices then give the location for where the field
# exists.
# ------------------------------------------------------------------------------
# Isolate field of first gap
midpoint = np.where(zmesh <= np.sum(gap_centers) / 2)[0][-1]
Ez_reduced = Ez0[:midpoint]
z_reduced = zmesh[:midpoint]

# Find where field strength is greater than 1% of Emax
exists = np.where(abs(Ez_reduced) >= 1.0e-3 * (abs(Ez_reduced)).max())[0]


# Isolate fields and shift z so that it is centered at z=0
Ez_start = exists[0]
Ez_end = exists[-1]
Ez_iso = abs(Ez0[Ez_start : Ez_end + 1])
z_iso = zmesh[Ez_start : Ez_end + 1] - gap_centers[0]


# ------------------------------------------------------------------------------
#     Fit polynomial function to field
# The field is isolated and centered about zero. A function will now be fit to
# the field that will represent every gap field with a 2*mm gap_width and the
# geometry used in the script that generated the fields.
# ------------------------------------------------------------------------------
def poly4d(x, coeffs):
    a, b, c, d, bias = coeffs
    func = a * pow(x, 4) + b * pow(x, 3) + c * pow(x, 2) + d * x + bias
    return func


def poly2d(x, coeffs):
    a, b, bias = coeffs
    func = a * pow(x, 2) + b * x + bias
    return func


fit4d = np.polyfit(z_iso, Ez_iso, 4)
fit2d = np.polyfit(z_iso, Ez_iso, 2)
print("Paramters for Quartic:", fit4d)
print("Paramters for Quadratic:", fit2d)
fig, ax = plt.subplots()
ax.plot(z_iso / mm, Ez_iso)
ax.plot(z_iso / mm, poly4d(z_iso, fit4d), c="g", lw="2", label="fit4d")
ax.plot(z_iso / mm, poly2d(z_iso, fit2d), c="m", lw="2", label="fit2d")
ax.axvline(-gap_width / 2 / mm, c="k", lw=1, ls="--")
ax.axvline(gap_width / 2 / mm, c="k", lw=1, ls="--")
ax.legend()
ax.set_xlabel("z [mm]")
ax.set_ylabel("Electric Field")
plt.show()

# Save isolated field and mesh
np.save("Ez_isolated_7kV_2mm_20um", Ez_iso)
np.save("z_isolated_7kV_2mm_20um", z_iso)
