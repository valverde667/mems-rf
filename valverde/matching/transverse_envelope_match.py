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

# The following parameters are used for the KV-solver. However, different
# values may want to be used with the matching section and acceleration
# section
match_Q = 6.986e-5
accel_Q = match_Q
match_emit = 1.336 * mm * mrad
accel_emit = match_emit
match_E_s = 7 * keV
accel_E_s = match_E_s
match_init_I = 10 * uA
accel_init_I = match_init_I
match_div_angle = 3.78 * mrad
accel_div_angle = match_div_angle

# System and Geometry settings
T_b = 0.1  # eV
Vg = 7 * kV
Ng = 4
phi_s = np.ones(Ng) * 0
f = 13.6 * MHz
lq = 0.695 * mm
lq_eff = 1.306 * mm
d = 3.0 * mm
g = 2 * mm
Vq = 0.6 * kV
match_Nq = 4
accel_Nq = 2
rp = 0.55 * mm
rsource = 0.25 * mm
G_hardedge = 2 * Vq / pow(rp, 2)
res = 10 * um

# Beam specifications
beam = wp.Species(type=wp.Argon, charge_state=1)
mass_eV = beam.mass * pow(SC.c, 2) / wp.jperev
beam.ekin = match_E_s
beam.ibeam = match_init_I
beam.a0 = rsource
beam.b0 = rsource
beam.ap0 = 0.0
beam.bp0 = 0.0
beam.vbeam = 0.0
vth = np.sqrt(T_b * wp.jperev / beam.mass)
wp.derivqty()

# Calculate Parameters
gap_centers = util.calc_gap_centers(accel_E_s, mass_eV, phi_s, f, Vg)
Lp = gap_centers[2] - g / 2

do_matching_section = False
do_accel_section = True
do_matching_section_optimization = False
do_accel_section_optimization = True

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
    match_scales = np.array([-0.6606901, 1.09060576, -0.97491688, 0.53503874])

    if user_input:
        # Instantiate the class and use the extracted fields to create the mesh.
        match_lattice = util.Lattice()
        match_file_names = ("iso_zgrad.npy", "iso_esq_grad.npy")
        match_lattice.user_input_match(match_file_names, match_Nq, scales=match_scales)
        match_z, match_grad = match_lattice.z, match_lattice.grad

    else:
        match_lattice = util.Lattice()
        # Create the hard edge equivalent. The max gradient is hardcoded and
        # corresponds to the gradient in a quadrupole with Vq=400 V.
        match_lattice.hard_edge_match(
            lq, lq_eff, d, Vq, match_Nq, rp, match_scales, max_grad=2566538624.836261
        )
        match_z, match_grad = match_lattice.z, match_lattice.grad

    match_dz = match_z[1] - match_z[0]
    match_kappa = wp.echarge * match_grad / 2.0 / match_E_s / wp.jperev

    # Solve KV equations for the lattice
    match_soln_matrix = np.zeros(shape=(len(match_z), 4))
    match_soln_matrix[0, :] = np.array(
        [rsource, rsource, match_div_angle, match_div_angle]
    )
    util.solver(match_soln_matrix, match_z, match_kappa, match_emit, match_Q)

    # Unpack solution arrays and save to data file.
    match_solutions = match_soln_matrix[-1, :]
    match_ux, match_uy = match_soln_matrix[:, 0], match_soln_matrix[:, 1]
    match_vx, match_vy = match_soln_matrix[:, 2], match_soln_matrix[:, 3]
    match_uxf, match_uyf = match_ux[-1], match_uy[-1]
    match_vxf, match_vyf = match_vx[-1], match_vy[-1]
    match_data = np.vstack((match_ux, match_uy, match_vx, match_vy, match_z))

    if user_input:
        np.save("matching_solver_data", match_data)
    else:
        np.save("matching_solver_data_hardedge", match_data)
        np.save("kappa_he", match_kappa)

    # Optimize for the voltage settings.
    if do_matching_section_optimization:
        match_x0 = np.array([rsource, rsource, match_div_angle, match_div_angle])
        match_rp_norm = 27 * mrad
        match_norms = np.array([1 / rp, 1 / rp, 1 / match_rp_norm, 1 / match_rp_norm])
        match_parameters = {
            "emit": match_emit,
            "Q": match_Q,
            "E": match_E_s,
            "Nq": match_Nq,
        }
        match_guess = match_scales

        match_opt = util.Optimizer(
            match_x0,
            match_guess,
            match_target,
            match_norms,
            match_file_names,
            match_parameters,
        )
        match_opt.optimize_matching = True
        match_opt.minimize_cost(match_opt.func_to_optimize_matching, max_iter=300)

if do_accel_section:
    accel_scales = np.array([0.85, 0.85])
    accel_fnames = ("accel_zmesh.npy", "accel_esq_grad.npy")
    accel_lattice = util.Lattice()
    accel_lattice.acceleration_lattice(
        gap_centers, accel_fnames, accel_scales, Lp, res=15 * um
    )
    accel_z, accel_grad = accel_lattice.z, accel_lattice.grad
    accel_dz = accel_z[1] - accel_z[0]

    accel_x0 = np.array([0.246 * mm, 0.314 * mm, 1.498 * mrad, -7.417 * mrad])
    accel_kappa = wp.echarge * accel_grad / 2.0 / accel_E_s / wp.jperev

    # Solve KV equations for the lattice
    accel_soln_matrix = np.zeros(shape=(len(accel_z), 4))
    accel_soln_matrix[0, :] = accel_x0
    history_Q, history_emit = util.solver_with_accel(
        accel_soln_matrix,
        accel_z,
        accel_kappa,
        accel_emit,
        accel_Q,
        accel_z,
        gap_centers,
        Vg,
        phi_s,
        accel_E_s,
        history=True,
    )

    # Unpack solution arrays and save to data file.
    accel_solutions = accel_soln_matrix[-1, :]
    accel_ux, accel_uy = accel_soln_matrix[:, 0], accel_soln_matrix[:, 1]
    accel_vx, accel_vy = accel_soln_matrix[:, 2], accel_soln_matrix[:, 3]
    accel_uxf, accel_uyf = accel_ux[-1], accel_uy[-1]
    accel_vxf, accel_vyf = accel_vx[-1], accel_vy[-1]
    accel_data = np.vstack((accel_ux, accel_uy, accel_vx, accel_vy, accel_z))

    np.save("acceleration_solver_data", accel_data)
    np.save("acceleration_z", accel_z)
    np.save("acceleration_kappa", accel_kappa)
    accel_target = accel_x0.copy()

    if do_accel_section_optimization:
        # Use coordinate vector to match from mathematica script
        accel_guess = accel_x0
        accel_target = accel_guess.copy()
        accel_rp_norm = 25 * mrad
        accel_norms = np.array([1 / rp, 1 / rp, 1 / accel_rp_norm, 1 / accel_rp_norm])
        accel_parameters = {
            "emit": accel_emit,
            "Q": accel_Q,
            "gap centers": gap_centers,
            "Vg": Vg,
            "Lp": Lp,
            "phi_s": phi_s,
            "E": accel_E_s,
        }
        accel_opt = util.Optimizer(
            accel_guess,
            accel_scales,
            accel_target,
            accel_norms,
            accel_fnames,
            accel_parameters,
        )
        accel_opt.optimize_acceleration = True
        accel_opt.z = accel_z
        accel_opt.grad = accel_grad
        # Create bounds for the coordinate position and angles
        rbound = (0.2 * mm, 0.8 * rp)
        rpbound = (-30 * mrad, 30 * mrad)
        accel_opt.bounds = [rbound, rbound, rpbound, rpbound]

        accel_opt.minimize_cost(accel_opt.func_to_optimize_acceleration, max_iter=1000)

# ------------------------------------------------------------------------------
#    Plot and Save
# Plot various quanities and save the data.
# ------------------------------------------------------------------------------
if do_matching_section:
    match_k0 = 1 * kV / match_E_s / pow(rp, 2)
    Fsc = 2 * match_Q / (match_ux + match_uy)
    Femitx = pow(match_emit, 2) / pow(match_ux, 3)
    Femity = pow(match_emit, 2) / pow(match_uy, 3)
    # match_kappa_he = np.load("matching_kappa_he.npy")
    # match_data_he = np.load("matching_solver_data_hardedge.npy")

    fig, ax = plt.subplots()
    ax.set_xlabel("z (mm)")
    ax.set_ylabel(r"$\kappa (z)/\hat{\kappa}$")
    plt.plot(match_z / mm, match_kappa / match_k0)
    ax.axhline(y=0, c="k", lw=0.5)
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title(r"Envelope Solutions for $r_x$ and $r_y$")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Transverse Position (mm)")
    ax.plot(match_z / mm, match_ux / mm, label=r"$r_x(s)$", c="k")
    ax.plot(match_z / mm, match_uy / mm, label=r"$r_y(s)$", c="b")
    ax.scatter(
        match_z[-1] / mm, match_target[0] / mm, marker="*", c="k", s=90, alpha=0.6
    )
    ax.scatter(
        match_z[-1] / mm, match_target[1] / mm, marker="*", c="b", s=90, alpha=0.6
    )
    ax.legend()

    fig, ax = plt.subplots()
    ax.set_title(r"Envelope Solutions for $rp_x$ and $rp_y$")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Transverse Angle (mrad)")
    ax.plot(match_z / mm, match_vx / mm, label=r"$rp_x(s)$", c="k")
    ax.plot(match_z / mm, match_vy / mm, label=r"$rp_y(s)$", c="b")
    ax.scatter(
        match_z[-1] / mm, match_target[2] / mrad, marker="*", c="k", s=90, alpha=0.6
    )
    ax.scatter(
        match_z[-1] / mm, match_target[3] / mrad, marker="*", c="b", s=90, alpha=0.6
    )
    ax.legend()

    fig, ax = plt.subplots()
    ax.plot(
        match_z / mm,
        2 * match_Q / (match_ux + match_uy),
        c="r",
        label=r"$F_\mathrm{SC}$",
    )
    ax.plot(
        match_z / mm,
        pow(match_emit, 2) / pow(match_ux, 3),
        c="k",
        label=r"$F_\mathrm{emit-x}$",
    )
    ax.plot(
        match_z / mm,
        pow(match_emit, 2) / pow(match_uy, 3),
        c="b",
        label=r"$F_\mathrm{emit-y}$",
    )
    ax.set_ylabel("Defocusing Term Strength (1/m)")
    ax.legend()

    Fig, ax = plt.subplots()
    ax.plot(
        match_z / mm, Fsc / Femitx, c="k", label=r"$F_\mathrm{sc} / F_\mathrm{x-emit}$"
    )
    ax.plot(
        match_z / mm, Fsc / Femity, c="b", label=r"$F_\mathrm{sc} / F_\mathrm{y-emit}$"
    )
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Ratio of Defocusing Terms")
    ax.legend()

    plt.show()

    print(f"Matching Section Final (rx, ry) mm: {match_uxf/mm:.4f}, {match_uyf/mm:.4f}")
    print(
        f"Matching Section Final (rpx, rpy) mrad: {match_vxf/mrad:.4f}, {match_vyf/mrad:.4f}"
    )

# ------------------------------------------------------------------------------
#    Acceleration Treatment
# Additional section to do acceleration matching
# ------------------------------------------------------------------------------
if do_accel_section:
    accel_k0 = 1 * kV / accel_E_s / pow(rp, 2)
    fig, ax = plt.subplots()
    ax.set_title(r"Initial (Non-optimized) $\kappa(z)$ ")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel(r"$\kappa (z)/\hat{\kappa}$")
    plt.plot(accel_z / mm, accel_kappa / accel_k0)
    ax.axhline(y=0, c="k", lw=0.5)
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title(r"Envelope Solutions for $r_x$ and $r_y$ with accel.")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Transverse Position (mm)")
    ax.plot(accel_z / mm, accel_ux / mm, label=r"$r_x(s)$", c="k")
    ax.plot(accel_z / mm, accel_uy / mm, label=r"$r_y(s)$", c="b")
    ax.scatter(accel_z[-1] / mm, accel_target[0] / mm, marker="*", c="k", s=90)
    ax.scatter(accel_z[-1] / mm, accel_target[1] / mm, marker="*", c="b", s=90)
    ax.legend()

    fig, ax = plt.subplots()
    ax.set_title(r"Envelope Solutions for $rp_x$ and $rp_y$ with accel.")
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Transverse Angle (mrad)")
    ax.plot(accel_z / mm, accel_vx / mm, label=r"$rp_x(s)$", c="k")
    ax.plot(accel_z / mm, accel_vy / mm, label=r"$rp_y(s)$", c="b")
    ax.scatter(accel_z[-1] / mm, accel_target[2] / mrad, marker="*", c="k", s=90)
    ax.scatter(accel_z[-1] / mm, accel_target[3] / mrad, marker="*", c="b", s=90)
    ax.legend()

    print(f"Accel. Section Final (rx, ry) mm: {accel_uxf/mm:.4f}, {accel_uyf/mm:.4f}")
    print(
        f"Accel. Section Final (rpx, rpy) mrad: {accel_vxf/mrad:.4f}, {accel_vyf/mrad:.4f}"
    )
