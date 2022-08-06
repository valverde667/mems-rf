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
mm = 1e-3
um = 1e-6
us = 1e-6
ns = 1e-9  # nanoseconds
twopi = 2 * np.pi

# ------------------------------------------------------------------------------
#     Functions
# This section creates necessary functions for the script.
# ------------------------------------------------------------------------------
def beta(E, mass=Ar_mass, q=1, nonrel=True):
    """Velocity of a particle with energy E."""
    if nonrel:
        beta = np.sqrt(2 * E / mass)
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
design_phase = -0
dsgn_initE = 7 * kV
Np = 10000
gap_width = 2 * mm

# Simulation parameters for gaps and geometries
design_gap_volt = 7.0 * kV
real_gap_volt = design_gap_volt
design_freq = 13.06 * MHz
real_freq = design_freq
design_omega = 2 * np.pi * design_freq
real_omega = design_omega
E_DC = real_gap_volt / gap_width
Ng = 2
Fcup_dist = 10 * mm
dsgn_finE = dsgn_initE + Ng * design_gap_volt * np.cos(design_phase)

# Energy analyzer parameters
dist_to_dipole = 25.0 * mm * 0
dipole_length = 50.0 * mm
dipole_gap_width = 11.0 * mm
dist_to_slit = 185.0 * mm
slit_width = 1.0 * mm
slit_center = 37 * mm

# ------------------------------------------------------------------------------
#     Gap Centers
# Place gaps to be in RF-resonance with a design particle that receives an energy
# kick with some design phase on the RF acceleration gap. The first gap starts out
# with the field being negative to ensure the most compact structure.
# ------------------------------------------------------------------------------
# Calculate additional gap centers if applicable. Here the design values should
# be used.
gap_dist = np.zeros(Ng)
for i in range(Ng):
    dsgn_Egain = design_gap_volt * np.cos(-design_phase)
    E = dsgn_initE + i * dsgn_Egain
    this_cent = beta(E) * SC.c / 2 / design_freq
    gap_dist[i] = this_cent

gap_centers = np.array(gap_dist).cumsum()


# ------------------------------------------------------------------------------
#    Mesh setup
# Here the mesh is setup to place a flat top field centered so that the design
# particle arrives in phase. The number of mesh points is paramterized by the
# mesh_res variable to represent a spacing resolution.
# ------------------------------------------------------------------------------
# Specify a mesh resolution
mesh_res = 5 * um
zmin = 0.0
zmax = gap_centers[-1] + gap_width / 2 + Fcup_dist
Nz = int((zmax - zmin) / mesh_res)
z = np.linspace(zmin, zmax, Nz)
dz = z[1] - z[0]
Ez0 = z.copy()

# ------------------------------------------------------------------------------
#    Particle Histories
# The particle arrays are created and the following quantities are tracked for the
# advancement: position, time, energy. This tracking will allow for various
# analysis such as relative differences from the design particle in time and
# energy.
# ------------------------------------------------------------------------------
# Create design particle arrays. The design particle is always to begin at z=0,
# t=0 and given the design initial energy.
dsgn_pos = np.zeros(Nz)
dsgn_E = np.zeros(Nz)
dsgn_time = np.zeros(Nz)

dsgn_pos[0] = 0.0
dsgn_E[0] = dsgn_initE
dsgn_time[0] = 0.0

# Calculate full DC beam length and start position of design particle. This will
# give a CW injection.
DC_length = SC.c / design_freq
particle_dist = np.linspace(-DC_length / 2, DC_length / 2, Np)

# Create particle arrays to store histories
parts_pos = np.zeros(shape=(Np, Nz))
parts_pos[:, 0] = particle_dist
parts_E = np.zeros(shape=(Np, Nz))
parts_time = np.zeros(shape=(Np, Nz))
parts_E[:, 0] = dsgn_initE

# initialize particles to z=0 along with times
vparts = np.sqrt(2 * parts_E[:, 0] / Ar_mass) * SC.c
time = (parts_pos[:, 0]) / vparts
parts_pos[:, 0] = 0
parts_time[:, 0] = time

# ------------------------------------------------------------------------------
#    Field Load
# The locations of the gaps are found in the z-mesh and using the gap thickness
# the flat-top field is loaded onto th emesh. The first gap  is maximally negative
# and the following gaps are 180º out of phase from the previous.
# The fields can either be loaded using a flat top or the real field which must
# extracted from the script 'fit_function_to_gap_field.py'
# ------------------------------------------------------------------------------
# Instantiate the flat-top field values in the gap regions.
use_flattop = False
use_real_field = True
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
    z_iso = np.load("z_isolated_7kV_2mm_20um.npy")
    Ez_iso = np.load("Ez_isolated_7kV_2mm_20um.npy")

    # Find extent of field
    Ez_extent = z_iso[-1] - z_iso[0]

    # Patch already created zmesh with the isolated zmesh corresponding to the
    # isolated field
    z_patch_arrays = []
    Ez_patch_arrays = []
    for i, cent in enumerate(gap_centers):

        z_patch = z_iso.copy() + cent
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
# The mesh sizing will have changed after adding patches so the arrays need to
# redone before advancing.
if use_real_field:
    Nz = len(z)
    dsgn_pos = np.zeros(Nz)
    dsgn_E = np.zeros(Nz)
    dsgn_time = np.zeros(Nz)

    dsgn_pos[0] = 0.0
    dsgn_E[0] = dsgn_initE
    dsgn_time[0] = 0.0

    # Create particle arrays to store histories
    parts_pos = np.zeros(shape=(Np, Nz))
    parts_pos[:, 0] = particle_dist
    parts_E = np.zeros(shape=(Np, Nz))
    parts_time = np.zeros(shape=(Np, Nz))
    parts_E[:, 0] = dsgn_initE

    # initialize particles to z=0 along with times
    vparts = np.sqrt(2 * parts_E[:, 0] / Ar_mass) * SC.c
    time = (parts_pos[:, 0]) / vparts
    parts_pos[:, 0] = 0
    parts_time[:, 0] = time


# ------------------------------------------------------------------------------
#    Particle Advancement
# Particles are initialized to times corresponding to z=0. The advancement is
# then done evaluated the energy gain using the field value and mesh spacing dz.
# ------------------------------------------------------------------------------
# Main loop to advance particles. Real parameter settings should be used here.
for i in range(1, len(z)):
    # Do design particle
    this_vs = beta(dsgn_E[i - 1]) * SC.c
    this_dt = dz / this_vs
    dsgn_time[i] = dsgn_time[i - 1] + this_dt

    Egain = Ez0[i - 1] * rf_volt(dsgn_time[i], freq=real_freq) * dz

    dsgn_E[i] = dsgn_E[i - 1] + Egain
    dsgn_pos[i] = dsgn_pos[i - 1] + dz
    dsgn_time[i] = dsgn_time[i - 1] + this_dt

    # Do other particles
    this_v = beta(parts_E[:, i - 1]) * SC.c
    this_dt = dz / this_v
    parts_time[:, i] = parts_time[:, i - 1] + this_dt

    Egain = Ez0[i - 1] * rf_volt(parts_time[:, i], freq=real_freq) * dz
    parts_E[:, i] = parts_E[:, i - 1] + Egain
    parts_pos[:, i] = parts_pos[:, i - 1] + dz

# Convert nan values to 0
final_E = np.nan_to_num(parts_E[:, -1])
final_t = np.nan_to_num(parts_time[:, -1])

# Plot the final energy and time but ignore the 0-bin since this will be overly
# large and drown out the distribution. This isn't done for time since doing so
# is more trouble then its worth. Do be sure to zoom in on the plot and make sure
# the distribution on either side of zero is relatively uniform since this is what
# it should be for a CW beam.
Ecounts, Eedges = np.histogram(final_E, bins=100)
tcounts, tedges = np.histogram(final_t, bins=100)

# Calculate percent of particles that in plot
percent_parts = np.sum(Ecounts[1:]) / Np * 100
fig, ax = plt.subplots()
ax.bar(
    Eedges[1:-1] / keV,
    Ecounts[1:] / Np,
    width=np.diff(Eedges[1:] / keV),
    edgecolor="black",
    lw="1",
    label=f"Percent Parts: {percent_parts:.2f}%",
)
ax.set_xlabel(r"Energy [keV]")
ax.set_ylabel(r"Fraction of Total Particles")
ax.legend()

fig, ax = plt.subplots()
ax.bar(
    tedges[:-1] / us,
    tcounts / Np,
    width=np.diff(tedges / us),
    edgecolor="black",
    lw="1",
)
ax.set_xlabel(r"Time [$\mu$s]")
ax.set_ylabel(r"Fraction of Total Particles")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------------------------
#    Bucket Analysis
# The percent_Edev variable finds the particles that are +/- the deviation in
# energy from the design particle. This can then be stored by the user and used
# to see how this bucket of particles evolves for gaps. The output plots will
# show the distribution of the energy relative to the design particle and
# distribution of phase relative to the design particle.
# ------------------------------------------------------------------------------
# Create mask using the desired percent deviation in energy
percent_Edev = 0.15
mask = (final_E >= dsgn_finE * (1 - percent_Edev)) & (
    (final_E <= dsgn_finE * (1 + percent_Edev))
)
bucket_E = final_E[mask]
bucket_time = final_t[mask]

# Plot distribution of bucket relative to design
d_bucket_E = bucket_E - dsgn_finE
d_bucket_phase = design_omega * (bucket_time - dsgn_time[-1])

# Plot distributions
d_bucket_Ecounts, d_bucket_Eedges = np.histogram(d_bucket_E, bins=100)
d_bucket_phasecounts, d_bucket_phaseedges = np.histogram(d_bucket_phase, bins=100)

# Calculate percent of particles that in plot
percent_parts = np.sum(d_bucket_Ecounts) / Np * 100
fig, ax = plt.subplots()
ax.bar(
    d_bucket_Eedges[:-1] / keV,
    d_bucket_Ecounts / Np,
    width=np.diff(d_bucket_Eedges / keV),
    edgecolor="black",
    lw="1",
    label=f"Percent Parts: {percent_parts:.2f}%",
)
ax.set_xlabel(r"$\Delta E$ [keV]")
ax.set_ylabel(r"Fraction of Total Particles")
ax.legend()

fig, ax = plt.subplots()
dsgn_final_phase = design_omega * dsgn_time[-1]
ax.bar(
    d_bucket_phaseedges[:-1] / dsgn_final_phase,
    d_bucket_phasecounts / Np,
    width=np.diff(d_bucket_phaseedges / dsgn_final_phase),
    edgecolor="black",
    lw="1",
)
ax.set_xlabel(r"$\Delta \phi$ in units of $\phi_{s,f}$")
ax.set_ylabel(r"Fraction of Total Particles")
ax.legend()
plt.show()
