# Solve KV-envelope equations with given initial conditions and different
# lattice creations.

import numpy as np
import os
import scipy.optimize as sciopt
import itertools
import scipy.constants as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb

import matching_utility as util

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
amu = sc.atomic_mass
twopi = np.pi * 2

# ------------------------------------------------------------------------------
#    Field Data, Parameters, Initial Conditions
# Here the necessary fields (on-axis electric field and quadrupole gradient) are
# loaded. These fields are normalized and then scaled later through the proper
# function call. The fields are representative of one conductor object, are
# positive, and centered on z=0. Only the 'strength' or magnitude of the field
# data can be scaled. The shape, extent, etc. cannot and if this is needed, the
# fields must be regenerated and saved using an isolated conductor in Warp or some
# other software.
# Lastly, this section is also dedicated to establishing the parameters and
# initial conditions.
# ------------------------------------------------------------------------------
grad_q = np.load("normed_esq_gradient.npy")
grad_z = np.load("normed_esq_z.npy")
gap_Ez = np.load("normalized_iso_Ez.npy")
gap_z = np.load("normalized_iso_z.npy")

# Beam parameters
mass = 39.948 * amu * pow(sc.c, 2) / sc.elementary_charge  # eV
init_E = 7 * keV
init_Q = 6.986e-5
init_emit = 1.344 * mm * mrad
init_rx = 0.25 * mm
init_ry = 0.25 * mm
init_rxp = 2.7 * mrad
init_ryp = 2.7 * mrad

# Gap parameters
g = gap_z.max() - gap_z.min()  # Includes fringe length and 2*mm physical spacing.
phi_s = np.pi * np.array([0.0, 0.0, 0.0])
gap_mode = np.zeros(len(phi_s))

freq = 13.5 * MHz
Vg = 6 * kV * 1.007
Vaccel = abs(np.cos(phi_s))[:-1] * Vg

# Quad Parameters
V1 = 0.0
V2 = 0.0
grad_scale_factor = 6.40e6  # Found numerically
# grad_scale_factor = 6.613e6  # Found numerically, twostack
# grad_scale_factor = 6.62594e6  # Found numerically, threestack
# grad_scale_factor = 6.6275e6  # Found numerically, fourstack
# grad_scale_factor = 6.6275e6  # Found numerically, fivestack
lq = grad_z.max() - grad_z.min()
separation = 5 * um

# Geometric settings for lattice
aperture = 0.55 * mm
scheme = "g-g-q-q"
gap_centers = util.calc_gap_centers(init_E, mass, phi_s, gap_mode, freq, Vg)
gap_centers = gap_centers - gap_centers.min() + g / 2.0
zstart = 0.0
zend = gap_centers[-1] - g / 2.0
quad_centers = util.calc_quad_centers(gap_centers, lq, separation, g, "equal")
do_optimization = False

# Prepare the field data using the gap and quad centers.
quad_inputs = util.prepare_quad_inputs(
    quad_centers, (grad_z, grad_q), [V1, V2], scale_factor=grad_scale_factor
)
gap_inputs = util.prepare_gap_inputs(gap_centers[:-1], (gap_z, gap_Ez), Vaccel)
# ------------------------------------------------------------------------------
#    Lattice Building and KV Integration
# With all the details specified above and the quad and gap inputs created using
# the helper functions, the lattice can easily be built. Once the lattice is built
# the lattice object can be fed into the KV solver class and the equations
# integrated. The solutions and various statistics can then be accessed through
# the class methods and properties.
# Note the ordering of methods. First, the lattice class builds the necessary
# arrays by stitching the fields togther. Then the particle is advanced using the
# provided Ez fields and the scheme. Finally, with the energy as a function of z
# the kappa function can be evaluated.
# From here, all the necessary arrays are created and stored with the lattice.
# The lattice object then needs to be given to the KV integrator and the equation
# solver called with the initial coordinates and starting Q and emittance.
# ------------------------------------------------------------------------------
lattice = util.Lattice()
lattice.build_lattice(zstart, zend, scheme, quad_inputs, gap_inputs, res=20 * um)
lattice.adv_particle(init_E)
lattice.calc_lattice_kappa()

kv_solve = util.Integrate_KV_equations(lattice)
kv_solve.integrate_eqns(
    np.array([init_rx, init_ry, init_rxp, init_ryp]), init_Q, init_emit, verbose=True
)
# ------------------------------------------------------------------------------
#   Optimization
# Here we attempt to find a matched condition for the lattice. The optimization
# uses nelder mead, a robust searching algorithm that is easily usable through
# scipy's library.
# ------------------------------------------------------------------------------
if do_optimization:
    opt_params = {"emit": init_emit, "Q": init_Q}
    rnorm = 1.0 / aperture
    rp_max = 10 * mrad
    rpnorm = 1.0 / rp_max
    norms = np.array([rnorm, rnorm, rpnorm, rpnorm])
    opt_norms = np.array(norms)
    init_coords = np.array([init_rx, init_ry, init_rxp, init_ryp])
    opt = util.Optimizer(opt_params, lattice, opt_norms)
    opt.minimize_cost_fixed_voltage(
        opt.match_fixed_voltage,
        init_coords,
        max_iter=600,
    )


# ------------------------------------------------------------------------------
#   Plotting
# Plot final solutions. If the optimization was done, the KV equations will be
# integrated over the same lattice with the optimized solutions.
# ------------------------------------------------------------------------------

if do_optimization:
    kv_solve.integrate_eqns(opt.sol, init_Q, init_emit, verbose=True)

fig, ax = plt.subplots()
ax.plot(lattice.z / mm, kv_solve.rx / mm, c="k", label=r"$r_x$")
ax.plot(lattice.z / mm, kv_solve.ry / mm, c="b", label=r"$r_y$")
ax.scatter(lattice.z[-1] / mm, kv_solve.rx[0] / mm, marker="*", c="k", s=90)
ax.scatter(lattice.z[-1] / mm, kv_solve.ry[0] / mm, marker="*", c="b", s=90)
ax.set_xlabel("z (mm)")
ax.set_ylabel("Transverse Envelope Edge Radii (mm)")
for gc in gap_centers[:-1]:
    ax.axvline(x=gc / mm, c="k", lw=1, ls="--")
ax.legend()
plt.tight_layout()

fig, ax = plt.subplots()
ax.plot(lattice.z / mm, kv_solve.rxp / mrad, c="k", label=r"$r_x'$")
ax.plot(lattice.z / mm, kv_solve.ryp / mrad, c="b", label=r"$r_y'$")
ax.scatter(lattice.z[-1] / mm, kv_solve.rxp[0] / mrad, marker="*", c="k", s=90)
ax.scatter(lattice.z[-1] / mm, kv_solve.ryp[0] / mrad, marker="*", c="b", s=90)
ax.set_xlabel("z (mm)")
ax.set_ylabel("Transverse Envelope Edge Angle (mrad)")
for gc in gap_centers[:-1]:
    ax.axvline(x=gc / mm, c="k", lw=1, ls="--")
ax.legend()
plt.tight_layout()

fig, ax = plt.subplots()
ax.plot(lattice.z / mm, lattice.kappa_quad + lattice.kappa_gap)
ax.set_xlabel("z (mm)")
ax.set_ylabel(r"Focusing Strength $\kappa(z)$ $(m^{-2})$")
for gc in gap_centers[:-1]:
    ax.axvline(x=gc / mm, c="k", lw=1, ls="--")
plt.tight_layout()

fig, ax = plt.subplots()
ax.plot(lattice.z / mm, lattice.beam_energy / keV)
ax.set_xlabel("z (mm)")
ax.set_ylabel("Beam Energy (keV)")
for gc in gap_centers[:-1]:
    ax.axvline(x=gc / mm, c="k", lw=1, ls="--")
ax.axhline(
    lattice.beam_energy[-1] / keV,
    c="r",
    ls="--",
    lw=1,
    label=rf"Final $E_b$ = {lattice.beam_energy[-1]/keV:.2f} (keV)",
)
ax.legend()
plt.tight_layout()

fig, ax = plt.subplots()
ax.plot(lattice.z / mm, kv_solve.Q / init_Q)
ax.set_xlabel("z (mm)")
ax.set_ylabel(r"Normalized Perveance $Q(z)/Q_0$")
for gc in gap_centers[:-1]:
    ax.axvline(x=gc / mm, c="k", lw=1, ls="--")
plt.tight_layout()

fig, ax = plt.subplots()
ax.plot(lattice.z / mm, kv_solve.emit / init_emit)
ax.set_xlabel("z (mm)")
ax.set_ylabel(r"Normalized RMS-edge Emittance  $\epsilon(z)/\epsilon_0$")
for gc in gap_centers[:-1]:
    ax.axvline(x=gc / mm, c="k", lw=1, ls="--")
plt.tight_layout()

# Print out the percent difference in matching conditions
drx = abs(kv_solve.rx[-1] - kv_solve.rx[0]) / kv_solve.rx[0] * 100
dry = abs(kv_solve.ry[-1] - kv_solve.ry[0]) / kv_solve.ry[0] * 100
drxp = abs((kv_solve.rxp[-1] - kv_solve.rxp[0]) / kv_solve.rxp[0]) * 100
dryp = abs((kv_solve.ryp[-1] - kv_solve.ryp[0]) / kv_solve.ryp[0]) * 100

print(f"{'Perecent difference Drx, Dry:':<30} {drx:.2f}, {dry:.2f}")
print(f"{'Perecent difference Drxp, Dryp:':<30} {drxp:.2f}, {dryp:.2f}")

# ------------------------------------------------------------------------------
#    Data Saving and Export
# Here, the data is collected into a single array and exported to the csv file.
# It is recommended the csv file be created ahead of time to avoid having to
# hand create the headers here if not.
# ------------------------------------------------------------------------------
file_name = "env_data.csv"

# Create array of data. CSV file precreated with headers.
data = np.array(
    [
        init_E / keV,
        lattice.beam_energy[-1] / keV,
        lattice.z[-1] / mm,
        V1,
        V2,
        Vg / kV,
        init_Q,
        init_emit / mm / mrad,
        kv_solve.sigmax / np.pi * 180,
        kv_solve.sigmay / np.pi * 180,
        kv_solve.rx[0] / mm,
        kv_solve.ry[0] / mm,
        kv_solve.rxp[0] / mrad,
        kv_solve.ryp[0] / mrad,
        kv_solve.avg_rx / mm,
        kv_solve.avg_ry / mm,
        kv_solve.avg_rxp / mrad,
        kv_solve.avg_ryp / mrad,
        kv_solve.max_rx / mm,
        kv_solve.min_rx / mm,
        kv_solve.max_ry / mm,
        kv_solve.min_ry / mm,
        kv_solve.max_rxp / mrad,
        kv_solve.min_rxp / mrad,
        kv_solve.max_ryp / mrad,
        kv_solve.min_ryp / mrad,
        kv_solve.Fx,
        kv_solve.Fy,
        (gap_centers[-1] - gap_centers[-2] - g) / mm,
    ]
)
