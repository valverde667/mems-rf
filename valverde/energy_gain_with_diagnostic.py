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
E_DC = real_gap_volt * kV / gap_width
Ng = 16
Fcup_dist = 30 * mm
Emax = dsgn_initE + Ng * design_gap_volt * np.cos(design_phase)

# Energy analyzer parameters
dist_to_dipole = 25.0 * mm
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
mesh_res = 50 * um
Nz = int((gap_centers[-1] + dist_to_dipole) / mesh_res)
z = np.linspace(0.0, gap_centers[-1] + dist_to_dipole, Nz)
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
#    Field Load and Advancement
# The locations of the gaps are found in the z-mesh and using the gap thickness
# the flat-top field is loaded onto th emesh. The first gap  is maximally negative
# and the following gaps are 180º out of phase from the previous.
# ------------------------------------------------------------------------------
# Instantiate the flat-top field values in the gap regions.
for i, cent in enumerate(gap_centers):
    if i % 2 == 0:
        field_loc = np.where((z >= cent - gap_width / 2) & (z <= cent + gap_width / 2))
        Ez0[field_loc] = -real_gap_volt / gap_width
    else:
        field_loc = np.where((z >= cent - gap_width / 2) & (z <= cent + gap_width / 2))
        Ez0[field_loc] = real_gap_volt / gap_width


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

# Bin the final energies on its own plot for later comparison using Warp fields.
fig, axes = plt.subplots(figsize=(10, 2))
axes.hist(final_E / keV, bins=100)

# ------------------------------------------------------------------------------
#    Dipole Selector
# In experiment, an energy analyzer is used to measure the final energy of the
# ions. A dipole consisting of two parallel plates is used to deflect the ions.
# The dipole voltage bias is varied and readings in a faraday cup are recorded
# for ions passing through a slit acting as a selector.
# This portion is meant to simulate this by first using a simple deflection
# equation. An array of voltage values is sampled and ions are selected based on
# their final deflected position using the slit width as a filter.
# ------------------------------------------------------------------------------
dipole_bias = np.linspace(0 * kV, 5 * kV, 100)

# Create an array to record the selected particles for each bias used
parts_selected = np.zeros(len(dipole_bias))

# Loop through biases and evaluate the deflection. Count how many particles
# land within the slit and store value.
filtered_E = final_E[np.where(final_E > 0.0)[0]]
slit_select_min = slit_center - slit_width / 2
slit_select_max = slit_center + slit_width / 2
# pdb.set_trace()
for i, bias in enumerate(dipole_bias):
    deflect = calc_dipole_deflection(bias, filtered_E)

    slit_selection = (deflect > slit_select_min) & (deflect < slit_select_max)
    selected = np.sum(deflect[slit_selection[0]])
    parts_selected[i] = selected

fig, ax = plt.subplots()
selected_inds = np.where(parts_selected > 0)[0]
ax.scatter(dipole_bias, parts_selected / Np, s=3)
ax.plot(dipole_bias, parts_selected / Np)
ax.set_xlabel("Dipole Bias [V]")
ax.set_ylabel("Fraction of Particles Selected")
plt.show()
