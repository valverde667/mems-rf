# This script is for understanding and testing that understanding of the phase
# space evolution of longitudinal dynamics. The present goals are to develop
# a routine that updates phase and kinetic energy using the finite difference
# equations. From here, the results will be compared to simulating a CW beam.
# The results will most likely not be identical but present similar phase-space
# plots. In particular, the eye, fish, and golf-club structures should be
# reproduced in both methods.

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import matplotlib.pyplot as plt
import scipy.constants as SC
import time
import pdb
import os


# ------------------------------------------------------------------------------
#     Constants and definitions section
# Establish some useful constants that will be used as units. Also, establish
# variable names that will be repeatedly used like mass, or 2 * np.pi, etc.
# This section will also contain function calls that are necessary to script.
# Some functions are defined for convenience, like calculating values or making
# plots; while others are more nuanced and have an increased level of
# sophistication.
# ------------------------------------------------------------------------------
# different particle masses in eV
# amu in eV
amu = SC.physical_constants["atomic mass constant energy equivalent in MeV"][0] * 1e6
Ar_mass = 39.948 * amu
He_mass = 4 * amu
p_mass = amu
kV = 1e3
keV = 1e3
MHz = 1e6
mm = 1e-3
ns = 1e-9  # nanoseconds
twopi = 2 * np.pi

# Function definitions start.
def calc_beta(E, mass=Ar_mass, q=1, nonrel=True):
    """Velocity of a particle with energy E."""
    if nonrel:
        sign = np.sign(E)
        beta = np.sqrt(2 * abs(E) / mass) * sign
    else:
        gamma = (E + mass) / mass
        beta = np.sqrt(1 - 1 / gamma / gamma)

    return beta


def calc_pires(energy, freq, mass=Ar_mass, q=1):
    """RF resonance condition in pi-mode"""
    beta_lambda = beta(energy, mass=mass, q=q) * SC.c / freq
    return beta_lambda / 2


# ------------------------------------------------------------------------------
#     Simulation Parameters/Settings
# This section sets various simulation parameters. In this case, initial kinetic
# energies, gap geometries, design settings such as frequency or phase, etc.
# ------------------------------------------------------------------------------
init_dsgn_E = 7 * keV
init_E = 7 * keV
init_dsgn_phi = -np.pi / 2
init_phi = -np.pi / 4
q = 1
Np = 10

Ng = 30
dsgn_freq = 13.6 * MHz
dsgn_gap_volt = 7 * kV * 0.01
dsgn_gap_width = 2 * mm
dsgn_DC_Efield = dsgn_gap_volt / dsgn_gap_width
transit_tfactor = 1.0

# ------------------------------------------------------------------------------
#     Naive Simulation and particle advancement
# The phase and kinetic energy arrays for both design and non-design particles
# are initialized and initial conditions included. The particles are then
# advanced for the specefied numbrer of gaps while updating the phases first
# and then using the updated phase to update the energy. Plots are made by taking
# the difference for phase - design_phase and E - dsgn_E for each gap.
# ------------------------------------------------------------------------------
dsgn_phase = np.zeros(Ng)
dsgn_E = np.zeros(Ng)
dsgn_phase[0], dsgn_E[0] = init_dsgn_phi, init_dsgn_E
phase = np.zeros(shape=(Np, Ng))
E = np.zeros(shape=(Np, Ng))
phase[:, 0], E[:, 0] = np.linspace(-np.pi, np.pi, Np), init_E * np.ones(Np)

# Main loop. Advance particles through rest of gaps and update arrays.
for i in range(1, Ng):
    beta_s = calc_beta(dsgn_E[i - 1])
    beta = calc_beta(E[:, i - 1])
    phase[:, i] = phase[:, i - 1] + np.pi * beta_s / beta + np.pi
    dsgn_phase[i] = dsgn_phase[i - 1] + twopi

    # Calculate coefficient in energy update for better organization
    coeff = q * dsgn_gap_volt * transit_tfactor
    E[:, i] = E[:, i - 1] + coeff * np.cos(phase[:, i])
    dsgn_E[i] = dsgn_E[i - 1] + coeff * np.cos(dsgn_phase[i])

# Make phase space plots for each gap. Do 10 gaps to keep small
# for i in range(Ng):
#     fig, ax = plt.subplots()
#     dphi = phase[:, i] - dsgn_phase[i]
#     dE = E[:, i] - dsgn_E[i]
#
#     # Shift dphi to be between -pi and pi
#     shift_dphi = twopi - dphi % twopi - np.pi
#     ax.scatter(shift_dphi / np.pi, dE / kV)
#     ax.set_title(f"Gap {i + 1}")
#     plt.show()

# ------------------------------------------------------------------------------
#     Simulation and particle advancement of differences
# The differences in phase and differences in kinetic energies between the design
# particle and not are incremented rather than computing the inidividual phases
# and energy then taking the difference. This is to see if there is any
# difference (I'd imagine not) and check understanding.
# ------------------------------------------------------------------------------
phi = np.zeros(shape=(Np, Ng))
dW = np.zeros(shape=(Np, Ng))
W_s = np.zeros(Ng)
init_phi = init_dsgn_phi
init_dW = np.linspace(-1.5, 1.5, Np) * kV

phi[:, 0] = init_phi
dW[:, 0] = init_dW
W_s[0] = init_dsgn_E

for i in range(1, Ng):
    beta_s = calc_beta(W_s[i - 1])
    phi[:, i] = phi[:, i - 1] - np.pi * dW[:, i - 1] / pow(beta_s, 2) / Ar_mass
    coeff = q * dsgn_gap_volt * transit_tfactor
    dW[:, i] = dW[:, i - 1] + coeff * (np.cos(phi[:, i]) - np.cos(init_dsgn_phi))

    W_s[i] = W_s[i - 1] + coeff * np.cos(init_dsgn_phi)

# Create dynamic plotting to visualize individual particle trajectories
do_dynamic_plot = False
if do_dynamic_plot:
    fig, ax = plt.subplots()
    ax.set_xlabel(fr"$\phi$ [rad], $\phi_s =$ {init_dsgn_phi/np.pi:.3f} $\pi$")
    ax.set_ylabel(r"$\Delta {{\cal E}}$ [keV]")

    plt.ion()
    plt.show()

    # Loop through the particles. For each particle loop through the gaps and plot
    # the particle's position in phase space.
    for i in range(0, Np - 3, 3):
        for j in range(Ng):
            ax.scatter(phi[i : i + 3, j], dW[i : i + 3, j] / kV, c="k", s=3)
            plt.draw()
            plt.pause(0.0001)
    fig.savefig(f"phase-space_{Np}Np{Ng}Ng", dpi=400)

    input("Press [enter] to continue.")

fig, ax = plt.subplots()
ax.set_title(f"Phase Space Trajectories for {Np} Particles and {Ng} gaps")
for i in range(Np):
    ax.scatter(phi[i, :] - init_dsgn_phi, dW[i, :] / kV, s=4, c="k")
ax.axhline(y=0, c="k", ls="--", lw=1)
ax.axvline(x=0, c="k", ls="--", lw=1)
ax.set_xlabel(fr"$\Delta \phi$ [rad], $\phi_s =$ {init_dsgn_phi/np.pi:.3f} $\pi$")
ax.set_ylabel(r"$\Delta {{\cal E}}$ [keV]")
plt.show()
