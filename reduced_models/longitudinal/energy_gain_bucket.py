# Script to simulate the advancement of ions advancing through acceleration gaps.
# The user can choose to load a flat-top field or a real field that was simulated
# in Warp and then exported to a numpy type file. There are a host of settings
# that can be modified early in the script. These variables are usually prefixed
# with 'dsgn_' to represent the design values used to construct the simulation
# lattice. Particle diagnostics are avaialble by setting the z-locations.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import scipy.constants as SC
import scipy.integrate as integrate
import scipy.optimize as opt
import periodictable
import seaborn as sns
import os
import pdb
import time

import energy_sim_utils as utils

mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.minor.top"] = True
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["ytick.right"] = True
mpl.rcParams["ytick.major.right"] = True
mpl.rcParams["ytick.minor.right"] = True
mpl.rcParams["figure.max_open_warning"] = 60

# different particle masses in eV
# amu in eV
amu_to_eV = SC.physical_constants["atomic mass constant energy equivalent in MeV"][0]
kV = 1000.0
keV = 1000.0
meV = 1.0e6
MHz = 1e6
cm = 1e-2
mm = 1e-3
um = 1e-6
us = 1e-6
ns = 1e-9  # nanoseconds
mA = 1e-3
uA = 1e-6
twopi = 2 * np.pi

# ------------------------------------------------------------------------------
#     Simulation Parameters
# This section is dedicated to naming and initializing design parameters that are
# to be used in the script. These names are maintained throughout the script and
# thus, if varied here are varied everywhere. In addition, some useful values
# such as the average DC electric field and initial RF wavelength are computed.
# ------------------------------------------------------------------------------
st = time.time()

# Design particle and beam parameters
ion = periodictable.argon
mass = ion.mass * amu_to_eV * meV
dsgn_initE = 7.0 * kV
Tb = 0.1
Np = int(1e4)

# Simulation parameters for gaps and geometries
phi_s = np.array([0.0, -1 / 6, -1 / 3, -1 / 2]) * np.pi
Ng = len(phi_s)
gap_width = 2.0 * mm
voltage_scale = 1 + 0.031
dsgn_gap_volt = 6.0 * voltage_scale * kV
dsgn_freq = 13.6 * MHz
real_gap_volt = dsgn_gap_volt
real_freq = dsgn_freq

# Calculate some useful quanitites
E_DC = real_gap_volt / gap_width
h_rf = SC.c / dsgn_freq
T_rf = 1.0 / dsgn_freq
E_spredicted = dsgn_initE + np.sum(
    np.array([dsgn_gap_volt * np.cos(phi) / voltage_scale for phi in phi_s])
)
E_max_attainable = dsgn_initE * dsgn_gap_volt / voltage_scale * Ng

# Energy analyzer parameters
Fcup_dist = 10 * mm

# Compute useful values
E_DC = real_gap_volt / gap_width / voltage_scale
dsgn_omega = twopi * dsgn_freq

# Fractions to mask particles in order to create a bucket for analysis.
alpha_E = 0.00
alpha_t = 0.40 * T_rf

# ------------------------------------------------------------------------------
#     Logical Flags
# Flags for different routines in the script.
# ------------------------------------------------------------------------------
l_use_flattop_field = False
l_use_Warp_field = True
l_mask_with_Fraction = True
l_mask_with_Hamiltonian = False
l_plot_diagnostics = False
l_plot_bucket_diagnostics = True
l_save_all_plots_pdf = False
l_plot_lattice = True
l_plot_RMS = False
l_plot_emit = False
plots_filename = "all-diagnostics.pdf"

# ------------------------------------------------------------------------------
#     Gap Centers
# Here, the gaps are initialized using the design values listed. The first gap
# is started fully negative to ensure the most compact structure. The design
# particle is initialized at z=0 t=0 so that it arrives at the first gap at the
# synchronous phase while the field is rising.
# ------------------------------------------------------------------------------
# Calculate additional gap centers if applicable. Here the design values should
# be used. The first gap is always placed such that the design particle will
# arrive at the desired phase when starting at z=0 with initial energy.
gap_mode = np.zeros(len(phi_s))
gap_centers = utils.calc_gap_centers(
    dsgn_initE, mass, phi_s, gap_mode, dsgn_freq, dsgn_gap_volt / voltage_scale
)
# ------------------------------------------------------------------------------
#    Mesh setup
# Here the mesh is created with some mesh design values.
# TODO:
#    - Create mesh refinement switch and procedure.
#    - Create logic switch for mesh refinement.
# ------------------------------------------------------------------------------
# Specify a mesh resolution
mesh_res = 10 * um
zmin = 0
zmax = gap_centers[-1] + gap_width / 2 + Fcup_dist
Nz = int((zmax - zmin) / mesh_res)
z = np.linspace(zmin, zmax, Nz)
dz = z[1] - z[0]
Ez0 = np.zeros(shape=z.shape)

# ------------------------------------------------------------------------------
#    Field Load
# The locations of the gaps are found in the z-mesh and using the gap thickness
# the flat-top field is loaded onto th emesh. The first gap  is maximally negative
# and the following gaps are 180º out of phase from the previous.
# The fields can either be loaded using a flat top or the real field which must
# extracted from the script 'fit_function_to_gap_field.py' This script reads in
# a field array that is generated in the "gap_field_function" script.
# TODO:
#    - Add mesh refinement.
# ------------------------------------------------------------------------------
# Instantiate the flat-top field values in the gap regions.
if l_use_flattop_field:
    for i, cent in enumerate(gap_centers):
        if i % 2 == 0:
            field_loc = np.where(
                (z >= cent - gap_width / 2) & (z <= cent + gap_width / 2)
            )
            Ez0[field_loc] = -real_gap_volt / gap_width
        else:
            field_loc = np.where(
                (z >= cent - gap_width / 2) & (z <= cent + gap_width / 2)
            )
            Ez0[field_loc] = real_gap_volt / gap_width

if l_use_Warp_field:
    # load isolated field
    z_iso = np.load("normalized_iso_z.npy")
    Ez_iso = abs(np.load("normalized_iso_Ez.npy"))

    # Scale electric field to match field-magnitude of Warp with input voltage.
    Ez_iso = Ez_iso * 657.3317 * dsgn_gap_volt

    # Find extent of field
    Ez_extent = z_iso[-1] - z_iso[0]

    # Patch already created zmesh with the isolated zmesh corresponding to the
    # isolated field
    z_patch_arrays = []
    Ez_patch_arrays = []
    for i, cent in enumerate(gap_centers):
        z_patch = z_iso + cent
        field_loc = np.where((z > cent - Ez_extent / 2) & (z < cent + Ez_extent / 2))
        patch_start = field_loc[0][0]
        patch_end = field_loc[0][-1]

        z_left = z[:patch_start]
        z_right = z[patch_end:]
        Ez0_left = Ez0[:patch_start]
        Ez0_right = Ez0[patch_end:]

        # Check for overlap between patched area and zmesh. If there is, remove
        # overlap and stitch together the patch.
        left_overlap = np.where((z_patch[0] - z_left) < 0)[0]
        if len(left_overlap) != 0:
            z_left = np.delete(z_left, left_overlap)
            Ez0_left = np.delete(Ez0_left, left_overlap)

        right_overlap = np.where((z_right - z_patch[-1]) < 0)[0]
        if len(right_overlap) != 0:
            z_right = np.delete(z_right, right_overlap)
            Ez0_right = np.delete(Ez0_right, right_overlap)

        # Stitch fields together and flip sign of the electric field
        z_patched = np.concatenate((z_left, z_patch, z_right))
        if i % 2 == 0:
            Ez0_patched = np.concatenate((Ez0_left, -Ez_iso, Ez0_right))
        else:
            Ez0_patched = np.concatenate((Ez0_left, Ez_iso, Ez0_right))

        # Rename previously defined meshs for continuity
        z = z_patched
        Nz = len(z)
        Ez0 = Ez0_patched
# ------------------------------------------------------------------------------
#    Particle Histories and Diagnostics
# The particle arrays are created here for the particle advancement. The design
# particle is the only particle with the history saved. The rest of the particle
# arrays are created and updated at each step.
# Diagnostics are placed at each gap, three diagnostics for gap: the entrance,
# gap center, and exit are all locations where the particle energy and time is
# recorded. Lastly, there are separate diagnostics placed at the midpoint between
# successive gaps where the Hamiltonian is computed.
# TODO:
#    - Add option to select particles to track and record history.
# ------------------------------------------------------------------------------
# Create design particle arrays. The design particle is always to begin at z=zmin,
# t=0 and given the design initial energy. The design particle is assume to be
# at z=0 at t=0 and is then moved backward in time to zmin
dsgn_pos = np.zeros(Nz)
dsgn_E = np.zeros(Nz)
dsgn_time = np.zeros(Nz)

dsgn_pos[0] = z.min()
dsgn_E[0] = dsgn_initE
dsgn_time[0] = z.min() / utils.beta(dsgn_E[0], mass) / SC.c

# Create particle arrays to store histories
parts_pos = np.zeros(Np)
parts_pos[:] = z.min()

parts_E = np.zeros(Np)
parts_E[:] = np.random.normal(loc=dsgn_initE, scale=Tb, size=Np)

parts_time = np.zeros(Np)

# Initialize particles be distributed around the synchronous particle's phase
init_time = np.linspace(-T_rf / 2, T_rf / 2, Np)

parts_pos[:] = z.min()
parts_time[:] = init_time

# Create arrays holding the zlocations of diagnostics and their respective index
# in the z-array
zdiagnostics = [z.min()]
idiagnostics = [0]
for loc in gap_centers:
    zdiagnostics.append(loc)
    this_ind = np.argmin(abs(z - loc))
    idiagnostics.append(this_ind)

zdiagnostics.append(z.max())
idiagnostics.append(len(z) - 1)
zdiagnostics = np.array(zdiagnostics)
idiagnostics = np.array(idiagnostics, dtype=int)

# Initialize diagnostic arrays. Find locations on mesh and use indexes to
# data from histories.
E_sdiagnostic = np.zeros(len(zdiagnostics))
t_sdiagnostic = np.zeros(len(zdiagnostics))
Ediagnostic = np.zeros(shape=(Np, len(zdiagnostics)))
tdiagnostic = np.zeros(shape=(Np, len(zdiagnostics)))
transmission_diagnostic = np.ones(len(zdiagnostics)) * Np

# Append initial diagnostics
E_sdiagnostic[0] = dsgn_E[0]
t_sdiagnostic[0] = dsgn_time[0]
Ediagnostic[:, 0] = parts_E
tdiagnostic[:, 0] = parts_time
Iavg = SC.e * Np / (parts_time[-1] - parts_time[0])

# ------------------------------------------------------------------------------
#    System Outputs
# Print some of the system parameters being used.
# ------------------------------------------------------------------------------
print("")
print("#----- Injected Beam")
print(f"{'Ion:':<30} {ion.name}")
print(f"{'Number of Particles:':<30} {Np:.0e}")
print(f"{'Injection Energy:':<30} {dsgn_E[0]/keV:.2f} [keV]")
print(f"{'Beam Temperature':<30} {Tb:.2f} [eV]")
print(f"{'Injected Average Current Iavg:':<30} {Iavg/mA:.4e} [mA]")
print(f"{'Predicted Final Design Energy:':<30} {dsgn_initE/keV:.2f} [keV]")

print("#----- Acceleration Lattice")
print(f"{'Number of Gaps':<30} {int(Ng)}")
print(f"{'Fcup Distance (from final plate):':<30} {Fcup_dist/mm:.2f} [mm]")
print(f"{'Gap Centers:':<30} {np.array2string(gap_centers/cm, precision=4)} [cm]")
print(
    f"{'Gap Distances:':<30} {np.array2string(np.diff(gap_centers/cm), precision=4)} [cm]"
)
print(f"{'System Length:':<30} {z.max()/cm:.3f} [cm]")
print(f"{'Gap Voltage:':<30} {dsgn_gap_volt/kV:.2f} [kV]")
print(f"{'Gap Width:':<30} {gap_width/mm:.2f} [mm]")
print(f"{'RF Frequency:':<30} {dsgn_freq/MHz:.2f} [MHz]")
print(f"{'RF Wavelength:':<30} {SC.c/dsgn_freq:.2f} [m]")
print(f"{'Sync Phi:':<30} {np.array2string(phi_s*180/np.pi,precision=3)} [deg]")

print("#----- Numerical Parameters")
print(f"{'Number of grid points:':<30} {len(z)}")
print(f"{'Grid spacing:':<30} {dz:>.4e} [m]")
print(f"{'Grid spacing in gap:':<30} {z_iso[1] - z_iso[0]:.4e} [m]")
print(
    f"{'Number of gridp points in gap:':<30} {int(np.floor(gap_width/(z_iso[1]-z_iso[0])))}"
)

# ------------------------------------------------------------------------------
#    Particle Advancement
# Particles are initialized to times corresponding to z=0. The advancement is
# then done evaluated the energy gain using the field value and mesh spacing dz.
# TODO:
#    Calculate Hamiltonian and tag particles in bucket.
# ------------------------------------------------------------------------------
# Main loop to advance particles. Real parameter settings should be used here.
idiagn_count = 1
for i in range(1, len(z)):
    # Do design particle
    this_dz = z[i] - z[i - 1]
    this_vs = utils.beta(dsgn_E[i - 1], mass) * SC.c

    # Evaluate the time for a half-step
    this_dt = this_dz / this_vs
    dsgn_time[i] = dsgn_time[i - 1] + this_dt
    Egain = Ez0[i] * utils.rf_volt(dsgn_time[i], freq=real_freq) * this_dz

    dsgn_E[i] = dsgn_E[i - 1] + Egain
    dsgn_pos[i] = dsgn_pos[i - 1] + this_dz

    # Do other particles
    mask = parts_E > 0
    this_v = utils.beta(parts_E[mask], mass) * SC.c
    this_dt = this_dz / this_v
    parts_time[mask] += this_dt

    Egain = Ez0[i - 1] * utils.rf_volt(parts_time[mask], freq=real_freq) * this_dz
    parts_E[mask] += Egain
    parts_pos[mask] += this_dz

    # Check diagnostic point
    if i == idiagnostics[idiagn_count]:
        E_sdiagnostic[idiagn_count] = dsgn_E[i]
        t_sdiagnostic[idiagn_count] = dsgn_time[i]
        Ediagnostic[:, idiagn_count] = parts_E[:]
        tdiagnostic[:, idiagn_count] = parts_time[:]
        transmission_diagnostic[idiagn_count] = np.sum(mask)

        idiagn_count += 1


# Convert nan values to 0
final_E = np.nan_to_num(parts_E[:])
final_t = np.nan_to_num(parts_time[:])

Ediagnostic = np.nan_to_num(Ediagnostic)
tdiagnostic = np.nan_to_num(tdiagnostic)

phase_sdiagnostic = twopi * dsgn_freq * t_sdiagnostic
phase_diagnostic = twopi * dsgn_freq * tdiagnostic
et = time.time()
print(f"End time: {et-st:.4f}")

# ------------------------------------------------------------------------------
#    Diagnostic Plots
# Plot phase space for each diagnostic location. The phase-space will be in terms
# of relative difference from the synchronous particle W-W_s and phi-phi_s at
# each location. Also plot the energy and time distribution at each diagnostic.
# ------------------------------------------------------------------------------
# Plot field with gaps
if l_plot_lattice:
    utils.make_lattice_plot(
        z, Ez0, gap_centers, phi_s, dsgn_gap_volt, gap_width, Fcup_dist
    )

# Plot the phase-space, energy and time distributions
if l_plot_diagnostics:
    for i, zloc in enumerate(zdiagnostics):
        # Grab energy and time
        this_E = Ediagnostic[:, i]
        this_t = tdiagnostic[:, i]
        this_Es = E_sdiagnostic[i]
        this_ts = t_sdiagnostic[i]
        dt = this_t - this_ts

        g = utils.make_dist_plot(
            dt / T_rf,
            this_E / keV,
            xlabel=r"Relative Time Difference $\Delta t / \tau_{rf}$",
            ylabel=r"Relative Energy Difference $\Delta {E}$ (keV)",
            auto_clip=True,
            xref=0.0,
            yref=this_Es / keV,
            levels=15,
            bins=40,
            weight=1 / Np_select,
            dx_bin=0.015,
            dy_bin=0.5,
        )


# ------------------------------------------------------------------------------
#    Bucket Analysis
# The percent_Edev variable finds the particles that are +/- the deviation in
# energy from the design particle. This can then be stored by the user and used
# to see how this bucket of particles evolves for gaps. The output plots will
# show the distribution of the energy relative to the design particle and
# distribution of phase relative to the design particle.
# ------------------------------------------------------------------------------
# Create mask for computing statistics. The negative energy particles are ignored
# when computing both the total beam statistics and the bucket statistics.

maskE = Ediagnostic[:, -1] > 0.0

if l_mask_with_Fraction:
    final_dt = tdiagnostic[:, -1] - t_sdiagnostic[-1]
    maskt = abs(final_dt) <= alpha_t
    mask = maskt & maskE

bucket_E = Ediagnostic[mask, -1]
bucket_time = tdiagnostic[mask, -1]

# Plot distribution of bucket relative to design
d_bucket_E = bucket_E - E_sdiagnostic[-1]
d_bucket_t = bucket_time - t_sdiagnostic[-1]

# Plot distributions
d_bucket_Ecounts, d_bucket_Eedges = np.histogram(d_bucket_E, bins=100)
d_bucket_tcounts, d_bucket_tedges = np.histogram(d_bucket_t, bins=100)

# Calculate percent of particles that in plot
percent_parts = np.sum(d_bucket_tcounts) / Np * 100
Np_select = np.sum(mask)

# Repeat previous plots for the selected particles
Np_zdiagnostic = np.zeros(len(zdiagnostics))
if l_plot_bucket_diagnostics:
    for i, zloc in enumerate(zdiagnostics):
        this_Np = len(Ediagnostic[mask, i])
        this_Np_select = np.sum(abs(tdiagnostic[:, i] - t_sdiagnostic[i]) <= alpha_t)
        Np_zdiagnostic[i] = this_Np_select
        if this_Np < int(1e4):
            rand_ints = np.random.randint(0, high=this_Np - 1, size=int(this_Np))
        else:
            rand_ints = np.random.randint(0, high=this_Np - 1, size=int(1e4))
        # Grab energy and time
        this_E = Ediagnostic[mask, i][rand_ints]
        this_t = tdiagnostic[mask, i][rand_ints]
        this_Es = E_sdiagnostic[i]
        this_ts = t_sdiagnostic[i]
        dt = this_t - this_ts

        g = utils.make_dist_plot(
            dt / T_rf,
            this_E / keV,
            xlabel=r"Relative Time Difference $\Delta t / \tau_{rf}$",
            ylabel=r"Relative Energy Difference $\Delta {E}$ (keV)",
            auto_clip=True,
            xref=0.0,
            yref=this_Es / keV,
            levels=15,
            bins=40,
            weight=1 / Np_select,
            dx_bin=0.015,
            dy_bin=0.5,
        )


# Plot design particle energy gain
fig, ax = plt.subplots()
ax.set_title("Design Particle Energy")
ax.set_xlabel("z[mm]")
ax.set_ylabel(r"$W_s$[keV]")
ax.plot(dsgn_pos / mm, dsgn_E / keV)
if Ng > 1:
    for cent in gap_centers:
        ax.axvline(cent / mm, c="grey", lw=1, ls="--")
else:
    ax.axvline(gap_centers[0] / mm, c="grey", lw=1, ls="--")
ax.axhline(
    dsgn_E[-1] / keV,
    c="r",
    ls="--",
    lw=1,
    label=rf"Final $W_s$ = {dsgn_E[-1]/keV:.2f}[keV]",
)
ax.legend()

# since these particles will throw off statistics.
rms_E = np.zeros(zdiagnostics.shape[-1])
rms_t = np.zeros(zdiagnostics.shape[-1])
rms_bucket_E = np.zeros(zdiagnostics.shape[-1])
rms_bucket_t = np.zeros(zdiagnostics.shape[-1])

emit = np.zeros(zdiagnostics.shape[-1])
emit_bucket = np.zeros(zdiagnostics.shape[-1])

for i in range(zdiagnostics.shape[-1]):
    # Calculate and plot mean RMS spread for energy and time
    ind = idiagnostics[i]
    ts = dsgn_time[ind]
    Es = dsgn_E[ind]
    t = tdiagnostic[:, i]
    E = Ediagnostic[:, i]

    # Mask out E<0 particles since these will skew calculations.
    rms_E[i] = np.mean(np.sqrt(pow(E[maskE] - Es, 2)))
    rms_t[i] = np.mean(np.sqrt(pow(t[maskE] - ts, 2)))
    rms_bucket_E[i] = np.mean(np.sqrt(pow(E[mask] - Es, 2)))
    rms_bucket_t[i] = np.mean(np.sqrt(pow(t[mask] - ts, 2)))

    # Calculate RMS emittance
    emit[i] = utils.calc_emittance(E[maskE] - Es, t[maskE] - ts)
    emit_bucket[i] = utils.calc_emittance(E[mask] - Es, t[mask] - ts)


if l_plot_RMS:
    fig = plt.figure(tight_layout=True, figsize=(8, 9))
    gs = gridspec.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    axx1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    axx2 = fig.add_subplot(gs[1, 1])

    ax1.set_title("RMS Energy Spread")
    ax1.set_xlabel("z[mm]")
    ax1.set_ylabel(r"$(\Delta E)_{rms}$ [keV]")
    ax1.text(
        0.5,
        2.18,
        f"Transmission %: {transmission_diagnostic[i]/Np*100:.2f}%",
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round", fc="lightgrey", ec="k", lw=1),
    )

    axx1.set_title("RMS Time Spread")
    axx1.set_ylabel(r"$(\Delta t)_{rms}$ [ns]")
    axx1.set_xlabel("z[mm]")
    ax1.yaxis.grid(True)
    axx1.yaxis.grid(True)

    # Plot gap centers
    for cent in gap_centers:
        ax1.axvline(cent / mm, c="grey", lw=1, ls="--")
        axx1.axvline(cent / mm, c="grey", lw=1, ls="--")

    ax1.scatter(zdiagnostics / mm, rms_E / keV, label="RMS Energy")
    axx1.scatter(zdiagnostics / mm, rms_t / ns, label="RMS Time")

    ax2.set_title("Bucket RMS Beam Energy Spread")
    ax2.set_xlabel("z[mm]")
    ax2.set_ylabel(r"$(\Delta E)_{rms}$ [keV]")
    ax2.text(
        0.5,
        0.97,
        f"% Particles in Bucket: {percent_parts:.2f}%",
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round", fc="lightgrey", ec="k", lw=1),
    )
    axx2.set_title("Bucket RMS Beam Time Spread")
    axx2.set_ylabel(r"$(\Delta t)_{rms}$ [ns]")
    axx2.set_xlabel("z[mm]")
    ax2.yaxis.grid(True)
    axx2.yaxis.grid(True)

    # Plot gap centers
    for cent in gap_centers:
        ax2.axvline(cent / mm, c="grey", lw=1, ls="--")
        axx2.axvline(cent / mm, c="grey", lw=1, ls="--")

    ax2.scatter(zdiagnostics / mm, rms_bucket_E / keV, label="RMS Energy")
    axx2.scatter(zdiagnostics / mm, rms_bucket_t / ns, label="RMS Time")
    plt.tight_layout()

if l_plot_emit:
    fig = plt.figure(tight_layout=True, figsize=(8, 9))
    gs = gridspec.GridSpec(2, 1)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    ax1.set_title("RMS Emittance")
    ax1.set_xlabel("z[mm]")
    ax1.set_ylabel(
        r"$\epsilon_{rms} = \sqrt{<\Delta E^2><\Delta t^2> - <\Delta E \Delta t>}$ [eV-s]"
    )
    ax1.text(
        0.5,
        2.18,
        f"Transmission %: {transmission_diagnostic[i]/Np*100:.2f}%",
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round", fc="lightgrey", ec="k", lw=1),
    )

    ax2.set_title("RMS Bucket Emittance")
    ax2.set_xlabel("z[mm]")
    ax2.set_ylabel(
        r"$\epsilon_{rms} = \sqrt{<\Delta E^2><\Delta t^2> - <\Delta E \Delta t>}$ [eV-s]"
    )
    ax2.text(
        0.5,
        0.97,
        f"% Particles in Bucket: {percent_parts:.2f}%",
        horizontalalignment="center",
        verticalalignment="top",
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round", fc="lightgrey", ec="k", lw=1),
    )
    ax1.yaxis.grid(True)
    ax2.yaxis.grid(True)

    # Plot gap centers
    for cent in gap_centers:
        ax1.axvline(cent / mm, c="grey", lw=1, ls="--")
        ax2.axvline(cent / mm, c="grey", lw=1, ls="--")

    ax1.scatter(zdiagnostics / mm, emit)
    ax2.scatter(zdiagnostics / mm, emit_bucket)

    plt.tight_layout()


if l_save_all_plots_pdf:
    pp = PdfPages(plots_filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pp, format="pdf")
    pp.close()
plt.show()

# ------------------------------------------------------------------------------
#    System Outputs for Bucket
# Print some of the system parameters being used.
# ------------------------------------------------------------------------------
print("")
print("#----- Bucket Characteristics ")
print(f"{'Final Design Energy:':<30} {dsgn_E[-1]/keV:.3f} [keV]")
print(f"{'Average Gain per Gap:':<30} {dsgn_E[-1]/keV/Ng:.2f} [keV]")
print(f"{'Time Selection Window:':<30} +/- {alpha_t/ns:.3f} [ns]")
print(f"{'Energy Selection Window:':<30} {alpha_E:.4f} [keV]")
print(f"{'Particles in Bucket:':<30} {percent_parts:.0f}%")
print(
    f"Fractional average current I/Iavg: {(np.sum(d_bucket_Ecounts)*SC.e/(d_bucket_tedges[-1] - d_bucket_tedges[0]))/Iavg:.4f} "
)
