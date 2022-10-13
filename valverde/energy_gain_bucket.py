# Script to simulate the advancement of ions using a field (either flat-top
# or something simple) contained within the acceleration gap. Once ions are
# advanced through the acceleration gaps they are then advanced through a drift,
# a dipole field, and then analyzed to simulate the deflector plate diagnostic
# used in the lab.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import scipy.constants as SC
import scipy.integrate as integrate
import os
import pdb


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
    spacing : float
        Dictates the range of values that will fall into the region holding the
        desired value in mesh. Best to overshoot with this parameter and make
        a broad range.

    Returns
    -------
    index : int
        Index for the mesh-value closest to the desired value.
    """

    # Check if value is already in mesh
    if value in mesh:
        return np.where(mesh == value)[0][0]

    # Create array of possible indices
    indices = np.where((mesh > (value - spacing)) & (mesh < (value + spacing)))[0]

    # Compute differences of the indexed mesh-value with desired value
    difference = []
    for index in indices:
        diff = np.sqrt((mesh[index] ** 2 - value ** 2) ** 2)
        difference.append(diff)

    # Smallest element will be the index closest to value in indices
    i = np.argmin(difference)
    index = indices[i]

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


def calc_pires(energy, freq, mass=Ar_mass, q=1):
    """RF resonance condition in pi-mode"""
    beta_lambda = beta(energy, mass=mass, q=q) * SC.c / freq
    return beta_lambda / 2


def plot_phase(phi, E):
    fig, ax = plt.subplots()
    ax.scatter(phi, E, s=2)
    ax.set_xlabel(r"$\Delta \phi$ [rad]")
    ax.set_ylabel(r"$\Delta {\cal E}$ [kV]")
    plt.tight_layout()
    plt.show()


def rf_volt(t, freq=13.6 * MHz):
    return np.cos(2 * np.pi * freq * t)


def calc_dipole_deflection(voltage, energy, length=50 * mm, g=11 * mm, drift=185 * mm):
    """Calculate an ion's deflection from a dipole field"""

    coeff = voltage * length / g / energy
    deflection = coeff * (length / 2 + drift)

    return deflection


# ------------------------------------------------------------------------------
#     Simulation Parameters
# This section is dedicated to naming and initializing design parameters that are
# to be used in the script. These names are maintained throughout the script and
# thus, if varied here are varied everywhere.
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

# ------------------------------------------------------------------------------
#     Gap Centers
# Place gaps to be in RF-resonance with a design particle that receives an energy
# kick with some design phase on the RF acceleration gap. The first gap starts out
# with the field being negative to ensure the most compact structure.
# ------------------------------------------------------------------------------
# Calculate additional gap centers if applicable. Here the design values should
# be used. The first gap is always placed such that the design particle will
# arrive at the desired phase when starting at z=0 with energy W.
phi_s = np.ones(Ng) * dsgn_phase
phi_s[1:] = np.linspace(-np.pi / 3, np.pi / 8, Ng - 1)

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
# Here the mesh is setup to place a flat top field centered so that the design
# particle arrives in phase. The number of mesh points is paramterized by the
# mesh_res variable to represent a spacing resolution.
# ------------------------------------------------------------------------------
# Specify a mesh resolution
mesh_res = 50 * um
zmin = 0
zmax = gap_centers[-1] + gap_width / 2 + Fcup_dist
Nz = int((zmax - zmin) / mesh_res)
z = np.linspace(zmin, zmax, Nz)
dz = z[1] - z[0]
Ez0 = z.copy()

# ------------------------------------------------------------------------------
#    Field Load
# The locations of the gaps are found in the z-mesh and using the gap thickness
# the flat-top field is loaded onto th emesh. The first gap  is maximally negative
# and the following gaps are 180º out of phase from the previous.
# The fields can either be loaded using a flat top or the real field which must
# extracted from the script 'fit_function_to_gap_field.py'
# ------------------------------------------------------------------------------
# Instantiate the flat-top field values in the gap regions.
use_flattop = True
use_real_field = False
if use_flattop:
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

if use_real_field:
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

# Plot field with gaps
fig, ax = plt.subplots()
ax.set_xlabel("z [mm]")
ax.set_ylabel(r"On-axis E-field $E(r=0, z)/E_{DC}$ [kV/mm]")
ax.plot(z / mm, Ez0 / E_DC)
if Ng > 1:
    for cent in gap_centers:
        ax.axvline(cent / mm, c="grey", lw=1, ls="--")
else:
    ax.axvline(gap_centers[0] / mm, c="grey", lw=1, ls="--")
plt.tight_layout()

# ------------------------------------------------------------------------------
#    Particle Histories
# The particle arrays are created and the following quantities are tracked for the
# advancement: position, time, energy. This tracking will allow for various
# analysis such as relative differences from the design particle in time and
# energy.
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
init_time = np.linspace(dsgn_phase - phi_dev_minus, dsgn_phase + phi_dev_plus, Np)
init_time = init_time / twopi / dsgn_freq

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

# Append initial diagnostics
E_sdiagnostic[0] = dsgn_E[0]
t_sdiagnostic[0] = dsgn_time[0]
Ediagnostic[:, 0] = parts_E
tdiagnostic[:, 0] = parts_time
Iavg = SC.e * Np / (parts_time[-1] - parts_time[0])
# ------------------------------------------------------------------------------
#    Particle Advancement
# Particles are initialized to times corresponding to z=0. The advancement is
# then done evaluated the energy gain using the field value and mesh spacing dz.
# ------------------------------------------------------------------------------
# Main loop to advance particles. Real parameter settings should be used here.
idiagn_count = 1
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
    this_v = beta(parts_E[:]) * SC.c
    this_dt = this_dz / this_v
    parts_time[:] += this_dt

    Egain = Ez0[i - 1] * rf_volt(parts_time[:], freq=real_freq) * this_dz
    parts_E[:] += Egain
    parts_pos[:] += this_dz

    # Check diagnostic point
    if i == idiagnostic[idiagn_count]:
        E_sdiagnostic[idiagn_count] = dsgn_E[i]
        t_sdiagnostic[idiagn_count] = dsgn_time[i]
        Ediagnostic[:, idiagn_count] = parts_E[:]
        tdiagnostic[:, idiagn_count] = parts_time[:]

        idiagn_count += 1


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
# each location.
# ------------------------------------------------------------------------------
lplot_diagnostics = False
if lplot_diagnostics:
    for i, zloc in enumerate(zdiagnostics):
        # Plot phase space
        fig, ax = plt.subplots()
        if i < len(zdiagnostics) - 1:
            ax.set_title(f"Phase-Space, z={zloc/mm:.2f}[mm] ($N_g$ = {i})")
        else:
            ax.set_title(f"Phase-Space, z={zloc/mm:.2f}[mm]")
        ax.set_xlabel(r"$\phi / \pi$")
        ax.set_ylabel(rf"$\Delta W$, $W_s$ = {E_sdiagnostic[i]/keV:.2f}[keV] ")
        ax.hist2d(
            np.modf((phase_diagnostic[:, i] - phase_sdiagnostic[i]) / np.pi)[0],
            (Ediagnostic[:, i] - E_sdiagnostic[i]) / keV,
            bins=[100, 100],
        )
        plt.tight_layout()

        # Plot the energy distribution at diagnostic
        Ecounts, Eedges = np.histogram(Ediagnostic[:, i], bins=100)
        fig, ax = plt.subplots()
        if i < len(zdiagnostics) - 1:
            ax.set_title(f"Energy Distribution, z={zloc/mm:.2f}[mm] ($N_g$ = {i})")
        else:
            ax.set_title(f"Energy Distibution, z={zloc/mm:.2f}[mm]")
        ax.bar(
            Eedges[:-1] / keV,
            Ecounts[:] / Np,
            width=np.diff(Eedges[:] / keV),
            edgecolor="black",
            lw="1",
        )
        ax.set_xlabel(r"Energy [keV]")
        ax.set_ylabel(r"Fraction of Total Particles")
        plt.tight_layout()

        # Plot the time distribution at diagnostic
        tcounts, tedges = np.histogram(tdiagnostic[:, i], bins=100)
        fig, ax = plt.subplots()
        ax.bar(
            tedges[:-1] / us,
            tcounts / Np,
            width=np.diff(tedges / us),
            edgecolor="black",
            lw="1",
        )
        if i < len(zdiagnostics) - 1:
            ax.set_title(f"Time Distribution, z={zloc/mm:.2f}[mm] ($N_g$ = {i})")
        else:
            ax.set_title(f"Time Distibution, z={zloc/mm:.2f}[mm]")
        ax.set_xlabel(r"$\Delta t$ [$\mu$s]")
        ax.set_ylabel(r"Fraction of Total Particles")
        plt.tight_layout()

    plt.show()

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
plt.show()

# Plot RMS values for each diagnostic
rms_E = np.mean(Ediagnostic, axis=0)
rms_t = np.mean(tdiagnostic, axis=0)

fig, ax = plt.subplots()
ax.set_title("RMS Energy and Time at Diagnostic")
ax2 = ax.twinx()
ax.set_xlabel("z[mm]")
ax.set_ylabel("RMS Energy [keV]")
ax2.set_ylabel(r"RMS Time [$\mu$s]")
ax2.yaxis.label.set_color("blue")

# Plot gap centers
for cent in gap_centers:
    ax.axvline(cent / mm, c="grey", lw=1, ls="--")

ax.plot(zdiagnostics / mm, rms_E / keV, c="k", label="RMS Energy")
ax2.plot(zdiagnostics / mm, rms_t / us, c="b", ls="--", label="RMS Time")

plt.show()

# ------------------------------------------------------------------------------
#    System Outputs
# Print some of the system parameters being used.
# ------------------------------------------------------------------------------
print("")
print("#----- Simulation Parameters")
print(f"Number of Gaps: {int(Ng)}")
print(f"Gap Centers: {np.array2string(gap_centers/cm, precision=2)}[cm]")
print(f"Gap Voltage: {dsgn_gap_volt/kV:.2f}[kV]")
print(f"RF Frequency: {dsgn_freq/MHz:.2f}[MHz]")
print(f"Fcup Distance: {Fcup_dist/mm:.2f}[mm]")
print(f"Sync Phi:{np.array2string(phi_s*180/np.pi,precision=2)}[deg]")
print(f"Injection Energy: {dsgn_E[0]/keV:.2f}[keV]")
print(f"Final Design Energy: {dsgn_E[-1]/keV:.2f}[keV]")
print(f"Gain per Gap: {(dsgn_E[-1]-dsgn_initE)/keV/Ng:.2f}[keV]")
print(f"Average Current: {Iavg/mA:.4e}[mA]")

# ------------------------------------------------------------------------------
#    Bucket Analysis
# The percent_Edev variable finds the particles that are +/- the deviation in
# energy from the design particle. This can then be stored by the user and used
# to see how this bucket of particles evolves for gaps. The output plots will
# show the distribution of the energy relative to the design particle and
# distribution of phase relative to the design particle.
# ------------------------------------------------------------------------------
# Create mask using the desired percent deviation in energy
fraction_Edev = 0.15
mask = (final_E >= dsgn_E[-1] * (1 - fraction_Edev)) & (
    (final_E <= dsgn_E[-1] * (1 + fraction_Edev))
)
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
fig, ax = plt.subplots()
ax.bar(
    d_bucket_Eedges[:-1] / keV,
    d_bucket_Ecounts / Np,
    width=np.diff(d_bucket_Eedges / keV),
    edgecolor="black",
    lw="1",
    label=f"Percent Transmission: {percent_parts:.2f}%",
)
ax.set_title(rf"Final Energy Distribution Within {fraction_Edev*100}% $\Delta W$")
ax.set_xlabel(rf"$\Delta W$ [keV], Predicted Final W: {dsgn_E[-1]/keV:.2f} [keV]")
ax.set_ylabel(r"Fraction of Total Particles")
ax.legend()

fig, ax = plt.subplots()
ax.bar(
    d_bucket_tedges[:-1] / us,
    d_bucket_tcounts / Np,
    width=np.diff(d_bucket_tedges) / us,
    edgecolor="black",
    lw="1",
    label=f"Percent Transmission: {percent_parts:.2f}%",
)
ax.set_title(rf"Final Time Distribution Within {fraction_Edev*100}% $\Delta W$")
ax.set_xlabel(r"$\Delta t$[$\mu$s]")
ax.set_ylabel(r"Fraction of Total Particles")
ax.legend()
plt.show()

# Repeat previous plots for the selected particles
lplot_bucket_diagnostics = True
if lplot_bucket_diagnostics:
    for i, zloc in enumerate(zdiagnostics):
        # Plot phase space
        fig, ax = plt.subplots()
        if i < len(zdiagnostics) - 1:
            ax.set_title(f"Bucket Phase-Space, z={zloc/mm:.2f}[mm] ($N_g$ = {i})")
        else:
            ax.set_title(f"Bucket Phase-Space, z={zloc/mm:.2f}[mm]")
        ax.set_xlabel(r"$\phi / \pi$")
        ax.set_ylabel(rf"$\Delta W$, $W_s$ = {E_sdiagnostic[i]/keV:.2f}[keV] ")
        ax.hist2d(
            np.modf((phase_diagnostic[mask, i] - phase_sdiagnostic[i]) / np.pi)[0],
            (Ediagnostic[mask, i] - E_sdiagnostic[i]) / keV,
            bins=[100, 100],
        )
        plt.tight_layout()

        # Plot the energy distribution at diagnostic
        Ecounts, Eedges = np.histogram(Ediagnostic[mask, i], bins=100)
        fig, ax = plt.subplots()
        if i < len(zdiagnostics) - 1:
            ax.set_title(
                f"Bucket Energy Distribution, z={zloc/mm:.2f}[mm] ($N_g$ = {i})"
            )
        else:
            ax.set_title(f"Bucket Energy Distibution, z={zloc/mm:.2f}[mm]")
        ax.bar(
            Eedges[:-1] / keV,
            Ecounts[:] / Np,
            width=np.diff(Eedges[:] / keV),
            edgecolor="black",
            lw="1",
        )
        ax.set_xlabel(r"Energy [keV]")
        ax.set_ylabel(r"Fraction of Total Particles")
        plt.tight_layout()

        # Plot the time distribution at diagnostic
        tcounts, tedges = np.histogram(tdiagnostic[mask, i], bins=100)
        fig, ax = plt.subplots()
        ax.bar(
            tedges[:-1] / us,
            tcounts / Np,
            width=np.diff(tedges / us),
            edgecolor="black",
            lw="1",
        )
        if i < len(zdiagnostics) - 1:
            ax.set_title(f"Bucket Time Distribution, z={zloc/mm:.2f}[mm] ($N_g$ = {i})")
        else:
            ax.set_title(f"Bucket Time Distibution, z={zloc/mm:.2f}[mm]")
        ax.set_xlabel(r"$\Delta t$ [$\mu$s]")
        ax.set_ylabel(r"Fraction of Total Particles")
        plt.tight_layout()

    plt.show()

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
plt.show()

# Plot RMS values for each diagnostic
rms_E = np.mean(Ediagnostic[mask, :], axis=0)
rms_t = np.mean(tdiagnostic[mask, :], axis=0)

fig, ax = plt.subplots()
ax.set_title("RMS Energy and Time at Diagnostic")
ax2 = ax.twinx()
ax.set_xlabel("z[mm]")
ax.set_ylabel("RMS Energy [keV]")
ax2.set_ylabel(r"RMS Time [$\mu$s]")
ax2.yaxis.label.set_color("blue")

# Plot gap centers
for cent in gap_centers:
    ax.axvline(cent / mm, c="grey", lw=1, ls="--")

ax.plot(zdiagnostics / mm, rms_E / keV, c="k", label="RMS Energy")
ax2.plot(zdiagnostics / mm, rms_t / us, c="b", ls="--", label="RMS Time")

plt.show()
# ------------------------------------------------------------------------------
#    System Outputs for Bucket
# Print some of the system parameters being used.
# ------------------------------------------------------------------------------
print("")
print("#----- Bucket Characteristics ")
print(f"Percent Energy Deviation Selection: {fraction_Edev*100:.0f}%")
print(f"Particles in Bucket: {percent_parts:.0f}%")
print(
    f"Average Current: {np.sum(d_bucket_Ecounts)*SC.e/(d_bucket_tedges[-1] - d_bucket_tedges[0])/mA:.4e}[mA] "
)
