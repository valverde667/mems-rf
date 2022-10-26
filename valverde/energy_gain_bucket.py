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
import os
import pdb

mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["ytick.minor.visible"] = True

# different particle masses in eV
# amu in eV
amu = SC.physical_constants["atomic mass constant energy equivalent in MeV"][0] * 1e6

Ar_mass = 39.948 * amu
He_mass = 4 * amu
p_mass = amu
kV = 1000
keV = 1000
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
#     Functions
# This section creates necessary functions for the script.
# ------------------------------------------------------------------------------


def getindex(mesh, value, spacing):
    """Find index in mesh for or mesh-value closest to specified value

    Function finds index corresponding closest to 'value' in 'mesh'. The spacing
    parameter should be enough for the range [value-spacing, value+spacing] to
    encompass nearby mesh-entries .

    Parameters
    ----------
    mesh : ndarray
        1D array that will be used to find entry closest to value
    value : float
        This is the number that is searched for in mesh.

    Returns
    -------
    index : int
        Index for the mesh-value closest to the desired value.
    """

    # Check if value is already in mesh
    if value in mesh:
        index = np.where(mesh == value)[0][0]

    else:
        index = np.argmin(abs(mesh - value))

    return index


def beta(E, mass=Ar_mass, q=1, nonrel=True):
    """Velocity of a particle with energy E."""
    if nonrel:
        sign = np.sign(E)
        beta = np.sqrt(2 * abs(E) / mass)
        beta *= sign
    else:
        gamma = (E + mass) / mass
        beta = np.sqrt(1 - 1 / gamma / gamma)

    return beta


def rf_volt(t, freq=13.6 * MHz):
    return np.cos(2 * np.pi * freq * t)


def calc_dipole_deflection(voltage, energy, length=50 * mm, g=11 * mm, drift=185 * mm):
    """Calculate an ion's deflection from a dipole field"""

    coeff = voltage * length / g / energy
    deflection = coeff * (length / 2 + drift)

    return deflection


def calc_Hamiltonian(W_s, phi_s, W, phi, f, V, g=2 * mm, m=Ar_mass, T=1):
    """Calculate the Hamiltonian.

    The non-linear Hamiltonian is dependent on the energy E, RF-frequency f,
    gap voltage V, gap width g, ion mass m and transit time factor T.

    Parameters
    ----------
    W_s : float
        Synchronous kinetic energy
    phi_s : float
        Synchronous phase
    W : float or array
        Kinetic energy for particles.
    phi : float or array
        phase of particles
    f : float
        Frequency of RF gaps.
    V : float
        Voltage applied on RF gaps.
    g : float
        Width of acceleration gap.
    m : float
        Ion mass.
    T : float
        Transit time factor.

    Returns
    -------
    H : float or array
        Hamiltonian values.

    """

    bs = beta(W_s, mass=m)
    hrf = SC.c / f
    A = twopi / hrf / pow(bs, 3)
    B = V * T / g / m

    term1 = 0.5 * A * pow((W - W_s) / m, 2)
    term2 = B * (np.sin(phi) - phi * np.cos(phi_s))
    H = term1 + term2

    return H


def calc_root_H(phi, phi_s=-np.pi / 2):
    """"Function for finding the 0-root for the Hamiltonian.

    The non-linear Hamiltonian has roots at phi=phi_s and phi_2.

    """
    term1 = np.sin(phi) + np.sin(phi_s)
    term2 = -np.cos(phi_s) * (phi - phi_s)

    return term1 + term2


# ------------------------------------------------------------------------------
#     Simulation Parameters
# This section is dedicated to naming and initializing design parameters that are
# to be used in the script. These names are maintained throughout the script and
# thus, if varied here are varied everywhere. In addition, some useful values
# such as the average DC electric field and initial RF wavelength are computed.
# ------------------------------------------------------------------------------
# Simulation Parameters for design particle
dsgn_phase = -np.pi / 2
dsgn_initE = 7 * kV
Np = int(1e5)

# Simulation parameters for gaps and geometries
Ng = 4
gap_width = 2.0 * mm
dsgn_gap_volt = 5.0 * kV
real_gap_volt = dsgn_gap_volt
dsgn_freq = 13.06 * MHz
real_freq = dsgn_freq
E_DC = real_gap_volt / gap_width
h_rf = beta(dsgn_initE, mass=Ar_mass) * SC.c / dsgn_freq

# Energy analyzer parameters
Fcup_dist = 10 * mm
dist_to_dipole = 25.0 * mm * 0
dipole_length = 50.0 * mm
dipole_gap_width = 11.0 * mm
dist_to_slit = 185.0 * mm
slit_width = 1.0 * mm
slit_center = 37 * mm

# Compute useful values
E_DC = real_gap_volt / gap_width
h_rf = beta(dsgn_initE, mass=Ar_mass) * SC.c / dsgn_freq
dsgn_omega = twopi * dsgn_freq

# Fractions to mask particles in order to create a bucket for analysis.
fraction_Edev = 0.05
fraction_tdev = 0.05

# ------------------------------------------------------------------------------
#     Logical Flags
# Flags for different routines in the script.
# ------------------------------------------------------------------------------
l_use_flattop_field = False
l_use_Warp_field = True
l_plot_diagnostics = True
l_plot_bucket_diagnostics = False
l_save_all_plots_pdf = True
l_plot_lattice = True
l_plot_RMS = True
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
# arrive at the desired phase when starting at z=0 with energy W.
phi_s = np.ones(Ng) * dsgn_phase
phi_s[1:] = np.array([-np.pi / 3, -np.pi / 4, -np.pi / 6])

gap_dist = np.zeros(Ng)
E_s = dsgn_initE
for i in range(Ng):
    this_beta = beta(E_s, mass=Ar_mass)
    this_cent = beta(E_s) * SC.c / 2 / dsgn_freq
    cent_offset = (phi_s[i] - phi_s[i - 1]) * this_beta * SC.c / dsgn_freq / twopi
    if i < 1:
        gap_dist[i] = (phi_s[i] + np.pi) * this_beta * SC.c / twopi / dsgn_freq
    else:
        gap_dist[i] = this_cent + cent_offset

    dsgn_Egain = dsgn_gap_volt * np.cos(phi_s[i])
    E_s += dsgn_Egain

gap_centers = np.array(gap_dist).cumsum()
# ------------------------------------------------------------------------------
#    Mesh setup
# Here the mesh is created with some mesh design values.
# TODO:
#    - Create mesh refinement switch and procedure.
#    - Create logic switch for mesh refinement.
# ------------------------------------------------------------------------------
# Specify a mesh resolution
mesh_res = 100 * um
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
    z_iso = np.load("z_isolated_5kV_2mm_10um.npy")
    Ez_iso = np.load("Ez_isolated_5kV_2mm_10um.npy")

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

        z_left = z[: patch_start + 1]
        z_right = z[patch_end:]
        Ez0_left = Ez0[: patch_start + 1]
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
dsgn_time[0] = z.min() / beta(dsgn_E[0]) / SC.c

# Create particle arrays to store histories
parts_pos = np.zeros(Np)
parts_pos[:] = z.min()

parts_E = np.zeros(Np)
parts_E[:] = dsgn_initE

parts_time = np.zeros(Np)

# Initialize particles be distributed around the synchronous particle's phase
phi_dev_plus = np.pi - dsgn_phase
phi_dev_minus = abs(-np.pi - dsgn_phase)
init_phase = np.linspace(dsgn_phase - phi_dev_minus, dsgn_phase + phi_dev_plus, Np)
init_time = init_phase / twopi / dsgn_freq

parts_pos[:] = z.min()
parts_time[:] = init_time

# Create diagnostic locations.
zdiagnostics = [z.min()]
for loc in gap_centers:
    zdiagnostics.append(loc)
zdiagnostics.append(z.max())
zdiagnostics = np.array(zdiagnostics)

idiagnostic = np.zeros(len(zdiagnostics), dtype=int)
for i, zloc in enumerate(zdiagnostics):
    ind = np.argmin(abs(z - zloc))
    idiagnostic[i] = ind

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

# Create Hamiltonianian diagnostics
z_Hdiagnostic = np.zeros(Ng)
for i in range(1, Ng):
    z_Hdiagnostic[i - 1] = (gap_centers[i - 1] + gap_centers[i]) / 2
z_Hdiagnostic[-1] = (gap_centers[-1] + z.max()) / 2

i_Hdiagnostic = np.zeros(len(z_Hdiagnostic), dtype=int)
for i, zloc in enumerate(z_Hdiagnostic):
    ind = np.argmin(abs(z - z_Hdiagnostic[i]))
    i_Hdiagnostic[i] = ind
Hdiagnostic = np.zeros(shape=(Np, Ng))
H_sdiagnostic = np.zeros(Ng)

# ------------------------------------------------------------------------------
#    Particle Advancement
# Particles are initialized to times corresponding to z=0. The advancement is
# then done evaluated the energy gain using the field value and mesh spacing dz.
# TODO:
#    Calculate Hamiltonian and tag particles in bucket.
# ------------------------------------------------------------------------------
# Main loop to advance particles. Real parameter settings should be used here.
idiagn_count = 1
i_Hdiagn_count = 0
for i in range(1, len(z)):

    # Do design particle
    this_dz = z[i] - z[i - 1]
    this_vs = beta(dsgn_E[i - 1]) * SC.c
    this_dt = this_dz / this_vs
    dsgn_time[i] = dsgn_time[i - 1] + this_dt

    Egain = Ez0[i - 1] * rf_volt(dsgn_time[i], freq=real_freq) * this_dz

    dsgn_E[i] = dsgn_E[i - 1] + Egain
    dsgn_pos[i] = dsgn_pos[i - 1] + this_dz
    dsgn_time[i] = dsgn_time[i - 1] + this_dt

    # Do other particles
    mask = parts_E > 0
    this_v = beta(parts_E[mask]) * SC.c
    this_dt = this_dz / this_v
    parts_time[mask] += this_dt

    Egain = Ez0[i - 1] * rf_volt(parts_time[mask], freq=real_freq) * this_dz
    parts_E[mask] += Egain
    parts_pos[mask] += this_dz

    # Check diagnostic point
    if i == idiagnostic[idiagn_count]:
        E_sdiagnostic[idiagn_count] = dsgn_E[i]
        t_sdiagnostic[idiagn_count] = dsgn_time[i]
        Ediagnostic[:, idiagn_count] = parts_E[:]
        tdiagnostic[:, idiagn_count] = parts_time[:]
        transmission_diagnostic[idiagn_count] = np.sum(mask)

        idiagn_count += 1

    # Check Hamiltonian diagnostic point
    if i <= i_Hdiagnostic[-1]:
        if i == i_Hdiagnostic[i_Hdiagn_count]:
            Hdiagnostic[:, i_Hdiagn_count] = calc_Hamiltonian(
                dsgn_E[i],
                twopi * dsgn_freq * dsgn_time[i],
                parts_E,
                twopi * dsgn_freq * parts_time[i],
                dsgn_freq,
                dsgn_gap_volt,
            )
            H_sdiagnostic[i_Hdiagn_count] = calc_Hamiltonian(
                dsgn_E[i],
                twopi * dsgn_freq * dsgn_time[i],
                dsgn_E[i],
                twopi * dsgn_freq * dsgn_time[i],
                dsgn_freq,
                dsgn_gap_volt,
            )

            i_Hdiagn_count += 1


# Convert nan values to 0
final_E = np.nan_to_num(parts_E[:])
final_t = np.nan_to_num(parts_time[:])

Ediagnostic = np.nan_to_num(Ediagnostic)
tdiagnostic = np.nan_to_num(tdiagnostic)

phase_sdiagnostic = twopi * dsgn_freq * t_sdiagnostic
phase_diagnostic = twopi * dsgn_freq * tdiagnostic

# ------------------------------------------------------------------------------
#    Diagnostic Plots
# Plot phase space for each diagnostic location. The phase-space will be in terms
# of relative difference from the synchronous particle W-W_s and phi-phi_s at
# each location. Also plot the energy and time distribution at each diagnostic.
# ------------------------------------------------------------------------------
# Plot field with gaps
if l_plot_lattice:
    fig, ax = plt.subplots()
    ax.set_title("Accel. Lattice with Applied Field at t=0", fontsize="large")
    ax.set_xlabel("z [mm]", fontsize="large")
    ax.set_ylabel(r"On-axis E-field $E(r=0, z)/E_{DC}$ [kV/mm]", fontsize="large")
    ax.plot(z / mm, Ez0 / E_DC)
    ax.axvline(gap_centers[0] / mm, c="grey", lw=1, ls="--", label="Gap Center")
    if Ng > 1:
        for i, cent in enumerate(gap_centers[1:]):
            ax.axvline(cent / mm, c="grey", lw=1, ls="--")

    ax.axvline(
        (gap_centers[-1] + gap_width / 2 + Fcup_dist) / mm,
        c="r",
        lw=1,
        ls="dashdot",
        label="Fcup",
    )
    ax.legend()
    plt.tight_layout()

# Plot the phase-space, energy and time distributions
if l_plot_diagnostics:
    for i, zloc in enumerate(zdiagnostics):
        # Plot phase space
        fig = plt.figure(tight_layout=True, figsize=(14, 12))
        gs = gridspec.GridSpec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        ax1.set_title(
            f"Longitudinal Phase-Space \n z={zloc/mm:.2f}[mm]", fontsize="x-large"
        )
        ax1.set_xlabel(r"Phase Deviation $\Delta \phi / \pi$", fontsize="x-large")
        ax1.set_ylabel(
            rf"Energy Deviation $\Delta W$[keV]", fontsize="x-large",
        )
        h = ax1.hist2d(
            np.modf((phase_diagnostic[:, i] - phase_sdiagnostic[i]) / np.pi)[0],
            (Ediagnostic[:, i] - E_sdiagnostic[i]) / keV,
            bins=[50, 50],
            cmin=0.01,
        )
        fig.colorbar(h[3], ax=ax1)

        # Plot the energy distribution at diagnostic
        Ecounts, Eedges = np.histogram(Ediagnostic[:, i], bins=100)
        ax2.set_title(
            f"Longitudinal Energy Distibution \n z={zloc/mm:.2f}[mm]",
            fontsize="x-large",
        )
        ax2.bar(
            Eedges[:-1] / keV,
            Ecounts[:] / Np,
            width=np.diff(Eedges[:] / keV),
            edgecolor="black",
            lw="1",
        )
        ax2.set_xlabel(r"Energy [keV]", fontsize="x-large")
        ax2.set_ylabel(r"Fraction of Particles", fontsize="x-large")
        ax2.text(
            0.5,
            0.99,
            f"Transmission %: {transmission_diagnostic[i]/Np*100:.2f}%",
            horizontalalignment="center",
            verticalalignment="top",
            transform=ax2.transAxes,
            bbox=dict(boxstyle="round", fc="lightgrey", ec="k", lw=1),
        )

        plt.tight_layout()

        # Plot the time distribution at diagnostic
        tcounts, tedges = np.histogram(tdiagnostic[:, i] - t_sdiagnostic[i], bins=100)
        ax3.bar(
            tedges[:-1] / us,
            tcounts / Np,
            width=np.diff(tedges / us),
            edgecolor="black",
            lw="1",
        )
        ax3.set_title(
            f"Longitudinal Time Distibution \n z={zloc/mm:.2f}[mm]", fontsize="x-large",
        )
        ax3.set_xlabel(r"$\Delta t$ [$\mu$s]", fontsize="x-large")
        ax3.set_ylabel(r"Fraction of Particles", fontsize="x-large")
        plt.tight_layout()


# ------------------------------------------------------------------------------
#    System Outputs
# Print some of the system parameters being used.
# ------------------------------------------------------------------------------
print("")
print("#----- Simulation Parameters")
print(f"Number of Gaps: {int(Ng)}")
print(f"Number of grid points: {len(z)}")
print(f"Grid spacing: {dz:.4e} [m]")
print(f"Grid spacing in gap: {z_iso[1] - z_iso[0]:.4e} [m]")
print(f"Steps in Gap: {int(np.floor(gap_width/(z_iso[1]-z_iso[0])))}")
print(f"Fcup Distance (from final plate): {Fcup_dist/mm:.2f} [mm]")
print(f"Gap Centers: {np.array2string(gap_centers/cm, precision=4)} [cm]")
print(f"Gap Distances:{np.array2string(np.diff(gap_centers/cm), precision=4)} [cm]")
print(f"Gap Voltage: {dsgn_gap_volt/kV:.2f} [kV]")
print(f"RF Frequency: {dsgn_freq/MHz:.2f} [MHz]")
print(f"Sync Phi:{np.array2string(phi_s*180/np.pi,precision=3)} [deg]")
print(f"Injection Energy: {dsgn_E[0]/keV:.2f} [keV]")
print(f"Final Design Energy: {dsgn_E[-1]/keV:.2f} [keV]")
print(f"Average Gain per Gap: {(dsgn_E[-1]-dsgn_initE)/keV/Ng:.2f} [keV]")
print(f"Average Current: {Iavg/mA:.4e} [mA]")

# ------------------------------------------------------------------------------
#    Bucket Analysis
# The percent_Edev variable finds the particles that are +/- the deviation in
# energy from the design particle. This can then be stored by the user and used
# to see how this bucket of particles evolves for gaps. The output plots will
# show the distribution of the energy relative to the design particle and
# distribution of phase relative to the design particle.
# ------------------------------------------------------------------------------
# Create mask using the desired percent deviation in energy
mask_E = (dsgn_initE >= dsgn_E[0] * (1 - fraction_Edev)) & (
    (final_E <= dsgn_initE * (1 + fraction_Edev))
)
mask_t = (init_time >= dsgn_time[0] * (1 - fraction_tdev)) & (
    (final_t <= dsgn_time[-1] * (1 + fraction_tdev))
)
# Use Hamiltonian to find phase-width and idenitfy bucket
phi2 = np.zeros(Ng)
for i in range(len(phi_s)):
    root = opt.root(calc_root_H, -0.75 * np.pi, args=phi_s[i])
    phi2[i] = root.x

mask = mask_E
bucket_E = final_E[mask]
bucket_time = final_t[mask]

# Plot distribution of bucket relative to design
d_bucket_E = bucket_E - dsgn_E[-1]
d_bucket_t = bucket_time - dsgn_time[-1]

# Plot distributions
d_bucket_Ecounts, d_bucket_Eedges = np.histogram(d_bucket_E, bins=100)
d_bucket_tcounts, d_bucket_tedges = np.histogram(d_bucket_t, bins=100)

# Calculate percent of particles that in plot
percent_parts = np.sum(d_bucket_Ecounts) / Np * 100

# Repeat previous plots for the selected particles
if l_plot_bucket_diagnostics:
    for i, zloc in enumerate(zdiagnostics):
        # Plot phase space
        fig = plt.figure(tight_layout=True, figsize=(14, 12))
        gs = gridspec.GridSpec(2, 2)
        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, 0])
        ax3 = fig.add_subplot(gs[1, 1])

        ax1.set_title(
            f"Longitudinal Bucket Phase-Space \n z={zloc/mm:.2f}[mm]",
            fontsize="x-large",
        )
        ax1.set_xlabel(r"Phase Deviation $\Delta \phi / \pi$")
        ax1.set_ylabel(
            rf"Energy Deviation $\Delta W$[keV]", fontsize="x-large",
        )
        h = ax1.hist2d(
            np.modf((phase_diagnostic[mask, i] - phase_sdiagnostic[i]) / np.pi)[0],
            (Ediagnostic[mask, i] - E_sdiagnostic[i]) / keV,
            bins=[50, 50],
            cmin=0.01,
        )
        fig.colorbar(h[3], ax=ax1)
        plt.tight_layout()

        # Plot the energy distribution at diagnostic
        Ecounts, Eedges = np.histogram(Ediagnostic[mask, i], bins=100)
        ax2.set_title(
            f"Longitudinal Bucket Energy Distribution \n z={zloc/mm:.2f}[mm] ",
            fontsize="x-large",
        )
        ax2.bar(
            Eedges[:-1] / keV,
            Ecounts[:] / Np,
            width=np.diff(Eedges[:] / keV),
            edgecolor="black",
            lw="1",
        )
        ax2.set_xlabel(r"Energy [keV]", fontsize="x-large")
        ax2.set_ylabel(r"Fraction of Particles", fontsize="x-large")
        plt.tight_layout()

        # Plot the time distribution at diagnostic
        tcounts, tedges = np.histogram(
            tdiagnostic[mask, i] - t_sdiagnostic[i], bins=100
        )
        ax3.bar(
            tedges[:-1] / us,
            tcounts / Np,
            width=np.diff(tedges / us),
            edgecolor="black",
            lw="1",
        )
        ax3.set_title(
            f"Longitudinal Bucket Time Distribution \n z={zloc/mm:.2f}[mm]",
            fontsize="x-large",
        )
        ax3.set_xlabel(r"$\Delta t$ [$\mu$s]", fontsize="x-large")
        ax3.set_ylabel(r"Fraction of Particles ", fontsize="x-large")
        plt.tight_layout()


# Plot design particle energy gain
fig, ax = plt.subplots()
ax.set_title("Design Particle Energy Gain")
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

# Plot RMS values for each diagnostic
rms_E = np.mean(Ediagnostic, axis=0)
rms_t = np.mean(tdiagnostic, axis=0)
rms_bucket_E = np.mean(Ediagnostic[mask, :], axis=0)
rms_bucket_t = np.mean(tdiagnostic[mask, :], axis=0)
if l_plot_RMS:
    fig = plt.figure(tight_layout=True, figsize=(8, 10))
    gs = gridspec.GridSpec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    axx1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    axx2 = fig.add_subplot(gs[1, 1])

    ax1.set_title("RMS Beam Energy \n at Diagnostic Locations")
    ax1.set_xlabel("z[mm]")
    ax1.set_ylabel("RMS Beam Energy [keV]")

    axx1.set_title("RMS Beam Time \n at Diagnostic Locations")
    axx1.set_ylabel(r"RMS Beam Time [$\mu$s]")
    axx1.set_xlabel("z[mm]")
    ax1.yaxis.grid(True)
    axx1.yaxis.grid(True)

    # Plot gap centers
    for cent in gap_centers:
        ax1.axvline(cent / mm, c="grey", lw=1, ls="--")
        axx1.axvline(cent / mm, c="grey", lw=1, ls="--")

    ax1.scatter(zdiagnostics / mm, rms_E / keV, label="RMS Energy")
    axx1.scatter(zdiagnostics / mm, rms_t / us, label="RMS Time")

    ax2.set_title("Bucket RMS Beam Energy \n at Diagnostic Locations")
    ax2.set_xlabel("z[mm]")
    ax2.set_ylabel("RMS Beam Energy [keV]")

    axx2.set_title("Bucket RMS Beam Time \n at Diagnostic Locations")
    axx2.set_ylabel(r"RMS Beam Time [$\mu$s]")
    axx2.set_xlabel("z[mm]")
    ax2.yaxis.grid(True)
    axx2.yaxis.grid(True)

    # Plot gap centers
    for cent in gap_centers:
        ax2.axvline(cent / mm, c="grey", lw=1, ls="--")
        axx2.axvline(cent / mm, c="grey", lw=1, ls="--")

    ax2.scatter(zdiagnostics / mm, rms_bucket_E / keV, label="RMS Energy")
    axx2.scatter(zdiagnostics / mm, rms_bucket_t / us, label="RMS Time")
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
print("Fractional energy selection: {fraction_Edev/keV:.2f} [keV]")
print("Fractional time selection: {fraction_tdev:.2e} [s]")
print(f"Percent Energy Deviation Selection: {fraction_Edev*100:.0f}%")
print(f"Particles in Bucket: {percent_parts:.0f}%")
print(
    f"Average Current: {np.sum(d_bucket_Ecounts)*SC.e/(d_bucket_tedges[-1] - d_bucket_tedges[0])/mA:.4e}[mA] "
)
