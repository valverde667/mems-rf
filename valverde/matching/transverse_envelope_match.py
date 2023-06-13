# Solve KV-envelope equations with given initial conditions and different
# lattice creations.

import numpy as np
import scipy.optimize as sciopt
import itertools
import scipy.constants as SC
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb

import matching_utility as util

import warp as wp

mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.minor.top"] = True
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["ytick.right"] = True
mpl.rcParams["ytick.major.right"] = True
mpl.rcParams["ytick.minor.right"] = True

# ------------------------------------------------------------------------------
#    Useful constants and Parameter initialization
# Define useful constants such as units. Initialize various paramters for the
# system/simulation.
# ------------------------------------------------------------------------------
# Define useful constants
mm = 1e-3
mrad = 1e-3
um = 1e-6
kV = 1e3
mrad = 1e-3
keV = 1e3
uA = 1e-6
MHz = 1e6
twopi = np.pi * 2

# System and Geometry settings
Vg = 7 * kV
Ng = 4
phi_s = np.ones(Ng) * 0
f = 13.6 * MHz
lq = 0.695 * mm
lq_eff = 1.306 * mm
d = 3.0 * mm
g = 2 * mm
Vq = 0.6 * kV
Nq = 4
rp = 0.55 * mm
rsource = 0.25 * mm
G_hardedge = 2 * Vq / pow(rp, 2)
res = 10 * um

# Beam Settings
Q = 6.986e-5
emit = 1.336 * mm * mrad
E_s = 7 * keV
init_I = 10 * uA
div_angle = 3.78 * mrad
Tb = 0.1  # eV

# Beam specifications
beam = wp.Species(type=wp.Argon, charge_state=1)
mass_eV = beam.mass * pow(SC.c, 2) / wp.jperev
beam.ekin = E_s
beam.ibeam = init_I
beam.a0 = rsource
beam.b0 = rsource
beam.ap0 = 0.0
beam.bp0 = 0.0
beam.ibeam = init_I
beam.vbeam = 0.0
beam.ekin = E_s
vth = np.sqrt(Tb * wp.jperev / beam.mass)
wp.derivqty()

# Calculate Parameters
gap_centers = util.calc_gap_centers(E_s, mass_eV, phi_s, f, Vg)
Lp = gap_centers[2] - g / 2

do_matching_section = True
do_accel_section = True

# ------------------------------------------------------------------------------
#    Lattice Setup
# The lattice is generated that will be used to solve the KV-envelope equations.
# -- Matching Section
# The two options are available for setting up the lattice:
#   1) The user-input options will use the extracted gradient for the desired
#   matching section simulated in a separate script. Both the gradient and the
#   simulated mesh is extracted.
#   2) If the user-input option is set to False then the hard edge model is used.
#   The hard edge model will place the ESQs using the physical length of the ESQ
#   but the effective length will be used to place the gradient on the mesh.
# -- Acceleration section
# This only takes a user input at the moment. I do not think it is worthwhile
# adding a hard-edge in.

# Once the lattice is created all the data is stored in the Lattice class and then
# used for the solver. Lastly, the gradients are scaled by to simulate different
# voltage settings.
# ------------------------------------------------------------------------------
if do_matching_section:
    user_input = True
    match_scales = np.array([0.15, 0.2, 0.10, 0.20])

    if user_input:
        # Instantiate the class and use the extracted fields to create the mesh.
        match_lattice = util.Lattice()
        match_file_names = ("iso_zgrad.npy", "iso_esq_grad.npy")
        match_lattice.user_input_match(match_file_names, Nq, scales=match_scales)
        match_z, match_grad = match_lattice.z, match_lattice.grad

    else:
        match_lattice = util.Lattice()
        # Create the hard edge equivalent. The max gradient is hardcoded and
        # corresponds to the gradient in a quadrupole with Vq=400 V.
        match_lattice.hard_edge_match(
            lq, lq_eff, d, Vq, Nq, rp, match_scales, max_grad=2566538624.836261
        )
        match_z, match_grad = match_lattice.z, match_lattice.grad

if do_accel_section:
    accel_scales = np.array([-0.1, -0.1])
    accel_fnames = ("accel_zmesh.npy", "accel_esq_grad.npy")
    accel_lattice = util.Lattice()
    accel_lattice.acceleration_lattice(gap_centers, accel_fnames, accel_scales, Lp)
    accel_z, accel_grad = accel_lattice.z, accel_lattice.grad


# Solve KV equations
dz = z[1] - z[0]
kappa = wp.echarge * gradz / 2.0 / E_s / wp.jperev
ux_initial, uy_initial = rsource, rsource
vx_initial, vy_initial = div_angle, div_angle

soln_matrix = np.zeros(shape=(len(z), 4))
soln_matrix[0, :] = ux_initial, uy_initial, vx_initial, vy_initial

util.solver(soln_matrix, dz, kappa, emit, Q)
# solver_with_accel(soln_matrix, dz, kappa, emit, Q, z, gap_centers, Vg=1*kV)
solutions = soln_matrix[-1, :]

ux, uy = soln_matrix[:, 0], soln_matrix[:, 1]
vx, vy = soln_matrix[:, 2], soln_matrix[:, 3]

uxf, uyf = ux[-1], uy[-1]
vxf, vyf = vx[-1], vy[-1]

data = np.vstack((ux, uy, vx, vy, z))
if user_input:
    np.save("matching_solver_data", data)
else:
    np.save("matching_solver_data_hardedge", data)
    np.save("kappa_he", kappa)


find_voltages = True
if find_voltages:
    x0 = np.array([rsource, rsource, div_angle, div_angle])
    guess = scales
    guess_accel = [0.2, 0.2]
    target = np.array([0.15 * mm, 0.28 * mm, 0.847 * mrad, -11.146 * mrad])
    rp_norm = 27 * mrad
    norms = np.array([1 / rp, 1 / rp, 1 / rp_norm, 1 / rp_norm])
    parameters = {
        "emit": emit,
        "Q": Q,
        "gap centers": gap_centers,
        "Vg": Vg,
        "Lp": Lp,
        "phi_s": phi_s,
        "E": E_s,
        "Nq": Nq,
    }
    opt = util.Optimizer(x0, guess, target, norms, file_names, parameters)
    opt.minimize_cost(opt.func_to_optimize, max_iter=300)
    errors = abs((target - solutions) / target)
    print(f"Fraction Errors: {errors}")

# ------------------------------------------------------------------------------
#    Plot and Save
# Plot various quanities and save the data.
# ------------------------------------------------------------------------------
k0 = 2 * 0.5 * kV
k0 = k0 / 7 / kV / pow(rp, 2)
Fsc = 2 * Q / (ux + uy)
Femitx = pow(emit, 2) / pow(ux, 3)
Femity = pow(emit, 2) / pow(uy, 3)
kappa_he = np.load("kappa_he.npy")
data_he = np.load("matching_solver_data_hardedge.npy")

fig, ax = plt.subplots()
ax.set_xlabel("z (mm)")
ax.set_ylabel(r"$\kappa (z)/\hat{\kappa}$")
plt.plot(z / mm, kappa / k0)
ax.axhline(y=0, c="k", lw=0.5)
plt.show()

fig, ax = plt.subplots()
ax.set_title(r"Envelope Solutions for $r_x$ and $r_y$")
ax.set_xlabel("z (mm)")
ax.set_ylabel("Transverse Position (mm)")
ax.plot(z / mm, ux / mm, label=r"$r_x(s)$", c="k")
ax.plot(z / mm, uy / mm, label=r"$r_y(s)$", c="b")
ax.scatter(z[-1] / mm, target[0] / mm, marker="*", c="k", s=90, alpha=0.6)
ax.scatter(z[-1] / mm, target[1] / mm, marker="*", c="b", s=90, alpha=0.6)
ax.legend()

fig, ax = plt.subplots()
ax.set_title(r"Envelope Solutions for $rp_x$ and $rp_y$")
ax.set_xlabel("z (mm)")
ax.set_ylabel("Transverse Angle (mrad)")
ax.plot(z / mm, vx / mm, label=r"$rp_x(s)$", c="k")
ax.plot(z / mm, vy / mm, label=r"$rp_y(s)$", c="b")
ax.scatter(z[-1] / mm, target[2] / mrad, marker="*", c="k", s=90, alpha=0.6)
ax.scatter(z[-1] / mm, target[3] / mrad, marker="*", c="b", s=90, alpha=0.6)
ax.legend()

fig, ax = plt.subplots()
plt.plot(data[-1, :] / mm, kappa / k0, c="b", label="Extracted")
plt.plot(data_he[-1, :] / mm, kappa_he / k0, c="g", alpha=0.5, label="Hard-Edge")
ax.axhline(y=0, c="k", lw=0.5)
ax.set_xlabel("z (mm)")
ax.set_ylabel(r"$\kappa(z)/\hat{\kappa}$")
plt.savefig("/Users/nickvalverde/Desktop/kappa_lattice.svg")
ax.legend()

fig, ax = plt.subplots()
ax.plot(z / mm, 2 * Q / (ux + uy), c="r", label=r"$F_\mathrm{SC}$")
ax.plot(z / mm, pow(emit, 2) / pow(ux, 3), c="k", label=r"$F_\mathrm{emit-x}$")
ax.plot(z / mm, pow(emit, 2) / pow(uy, 3), c="b", label=r"$F_\mathrm{emit-y}$")
ax.set_ylabel("Defocusing Term Strength (1/m)")
ax.legend()

Fig, ax = plt.subplots()
ax.plot(z / mm, Fsc / Femitx, c="k", label=r"$F_\mathrm{sc} / F_\mathrm{x-emit}$")
ax.plot(z / mm, Fsc / Femity, c="b", label=r"$F_\mathrm{sc} / F_\mathrm{y-emit}$")
ax.set_xlabel("z (mm)")
ax.set_ylabel("Ratio of Defocusing Terms")
ax.legend()


plt.show()


print(f"Final (rx, ry) mm: {uxf/mm:.4f}, {uyf/mm:.4f}")
print(f"Final (rpx, rpy) mrad: {vxf/mrad:.4f}, {vyf/mrad:.4f}")

# ------------------------------------------------------------------------------
#    Acceleration Treatment
# Additional section to do acceleration matching
# ------------------------------------------------------------------------------
accel_fnames = ("accel_zmesh.npy", "accel_esq_grad.npy")
accel_scales = np.array([0.25, 0.25])
accel_scales = np.array([-0.08961465, -0.12368815])
lattice_with_accel = util.Lattice()
lattice_with_accel.accel_lattice(gap_centers, accel_fnames, accel_scales, Lp)
accel_z, accel_grad = lattice_with_accel.z, lattice_with_accel.grad
accel_dz = accel_z[1] - accel_z[0]
accel_kappa = wp.echarge * accel_grad / 2.0 / E_s / wp.jperev

fig, ax = plt.subplots()
ax.set_title(r"Initial (Non-optimized) $\kappa(z)$ ")
ax.set_xlabel("z (mm)")
ax.set_ylabel(r"$\kappa (z)/\hat{\kappa}$")
plt.plot(accel_z / mm, accel_kappa / k0)
ax.axhline(y=0, c="k", lw=0.5)
plt.show()

# Plot the kv-solver
accel_x0 = np.array([0.27954 * mm, 0.34075 * mm, 1.8497 * mrad, -6.8141 * mrad])
accel_target = accel_x0.copy()

ux_initial, uy_initial = accel_x0[:2]
vx_initial, vy_initial = accel_x0[2:]

accel_soln_matrix = np.zeros(shape=(len(accel_z), 4))
accel_soln_matrix[0, :] = ux_initial, uy_initial, vx_initial, vy_initial

util.solver_with_accel(
    accel_soln_matrix,
    accel_dz,
    accel_kappa,
    emit,
    Q,
    accel_z,
    gap_centers,
    Vg=Vg,
    phi_s=phi_s,
    E=E_s,
)
# solver_with_accel(soln_matrix, dz, kappa, emit, Q, z, gap_centers, Vg=1*kV)
accel_solutions = accel_soln_matrix[-1, :]

ux, uy = accel_soln_matrix[:, 0], accel_soln_matrix[:, 1]
vx, vy = accel_soln_matrix[:, 2], accel_soln_matrix[:, 3]

uxf, uyf = ux[-1], uy[-1]
vxf, vyf = vx[-1], vy[-1]


fig, ax = plt.subplots()
ax.set_title(r"Envelope Solutions for $r_x$ and $r_y$ with accel.")
ax.set_xlabel("z (mm)")
ax.set_ylabel("Transverse Position (mm)")
ax.plot(accel_z / mm, ux / mm, label=r"$r_x(s)$", c="k")
ax.plot(accel_z / mm, uy / mm, label=r"$r_y(s)$", c="b")
ax.scatter(accel_z[-1] / mm, accel_target[0] / mm, marker="*", c="k", s=90, alpha=0.6)
ax.scatter(accel_z[-1] / mm, accel_target[1] / mm, marker="*", c="b", s=90, alpha=0.6)
ax.legend()

fig, ax = plt.subplots()
ax.set_title(r"Envelope Solutions for $rp_x$ and $rp_y$ with accel.")
ax.set_xlabel("z (mm)")
ax.set_ylabel("Transverse Angle (mrad)")
ax.plot(accel_z / mm, vx / mm, label=r"$rp_x(s)$", c="k")
ax.plot(accel_z / mm, vy / mm, label=r"$rp_y(s)$", c="b")
ax.scatter(accel_z[-1] / mm, accel_target[2] / mrad, marker="*", c="k", s=90, alpha=0.6)
ax.scatter(accel_z[-1] / mm, accel_target[3] / mrad, marker="*", c="b", s=90, alpha=0.6)
ax.legend()

# Use coordinate vector to match from mathematica script
guess_accel = accel_scales
rp_norm = 15 * mrad
norms = np.array([1 / rp, 1 / rp, 1 / rp_norm, 1 / rp_norm])
parameters = {
    "emit": emit,
    "Q": Q,
    "gap centers": gap_centers,
    "Vg": Vg,
    "Lp": Lp,
    "phi_s": phi_s,
    "E": E_s,
}
opt = util.Optimizer(
    accel_x0, guess_accel, accel_target, norms, accel_fnames, parameters
)
opt.minimize_cost(opt.func_to_optimize_accel, max_iter=300)
errors = abs((target - solutions) / target)
print(f"Fraction Errors: {errors}")
