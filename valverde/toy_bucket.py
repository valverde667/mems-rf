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


def calc_synch_ang_freq(f, V, phi_s, W_s, T=1, g=1 * mm, m=Ar_mass, q=1):
    """Calculate synchrotron angular frequency for small acceleration and phase diff.

    The formula assumes that the velocity of the design particle is
    approximately constants (beta_s and gamma_s ≈ const.) and that other
    particles only slightly deviate from the design particle in phase.
    This leads to a linear second order ODE and a conservation of phase space.
    The wavenumber and hence frequency is a function of the input parameters
    listed below.

    Parameters:
    -----------
    f : float
        RF frequency for the system.
    V : float
        The voltage amplitude applied to the RF gap.
    phi_s : float
        Design frequency. For the harmonic expression of the 2nd Order ODE, this
        value should be negative.
    W_s : float
        The kinetic energy of the design particle. This is assumed to remain
        constant throughout the acceleration gaps.
    T : float
        Transit time factor. Can be treated as a parameter to be varied but
        initialized to 1. Values are between [0,1].
    g : float
        Thickness of accelerating gap.
    """
    # Evaluate a few variables before hand
    beta_s = calc_beta(W_s, mass=m, q=q)
    E0 = V / g
    lambda_rf = SC.c / f

    # Evaluate chunks of the expression for the wave number k_s
    energy_chunk = q * E0 * T * np.sin(-phi_s) / m
    wave_chunk = 2 * np.pi / lambda_rf / pow(beta_s, 3)

    # Compute wavenumber. Since this approximation assumes small phase deviation
    # and small acceleration beta ≈ beta_s so omega = k_s * beta_s * c
    wave_number = np.sqrt(wave_chunk * energy_chunk)
    omega_synch = wave_number * beta_s * SC.c

    return omega_synch


def convert_to_spatial(dW, W_s, dphi, phi_s, f, gap_centers):
    """Take dW and dphi coordinates and convert to dphi and z

    The main script evaluates the relative change in phase and kinetic energy
    from the synchronous particle gap-to-gap. It is also useful to see the
    change in phase and energy as the particles progress down the beamline in
    z. Especially in the contant energy and small phase change approximation
    scheme where the difference equations can be analytically evaluated giving
    sinusoid formulations. This can be plotted along with the finite difference
    evaluations for comparison and checking.

    Parameters
    ----------
    dW : ndarray
        This array represents the relative change from the synchronous energy W_s
        from gap-to-gap. Thus, the dimensions are #particles by # gaps (Np, Ng)
    W_s : ndarray
        This array gives the design kinetic for the design particle for each
        gaps and is a 1 by Ng array.
    dphi : ndarray
        This array gives the relative phase difference gap-to-gap from the
        design particle and is of shape (Np, Ng).
    phi_s : float
        The design phase used in the simulation.
    f : float
        The RF frequency used in the simulation.
    gap_centers : ndarray
        The locations of the gap centers. Shape of (1, Ng).

    Returns
    -------
    z : ndarray
        This array gives the position of the particles corresponding to the
        phasing down the beamline.
    """
    # Assign the number of particles and number of gaps used
    Np, Ng = dW.shape

    # Calculate beta for each particle after each gap.
    beta = np.zeros(shape=(Np, Ng))
    for i in range(Ng):
        # Use relative energy difference to calc particle energy and corresponding
        # beta
        beta[:, i] = calc_beta(dW[:, i] + W_s[i])

    # Calculate z-coordinates for each particle when design particle is at gap
    # center.
    coeff = SC.c / 2 / np.pi / f
    z = np.zeros(shape=(Np, Ng))
    for i in range(Ng):
        zs_i = gap_centers[i]
        dz_i = beta[:, i] * dphi[:, i] + (beta[:, i] - beta_s[i]) * phi_s

        z[:, i] = zs_i + coeff * dz_i

    return z


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
dsgn_gap_volt = 7 * kV
dsgn_gap_width = 2 * mm
dsgn_DC_Efield = dsgn_gap_volt / dsgn_gap_width
transit_tfactor = 1.0
omega_s = calc_synch_ang_freq(dsgn_freq, dsgn_gap_volt, init_dsgn_phi, init_dsgn_E)

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
beta_s = np.zeros(Ng)
init_phi = init_dsgn_phi
init_dW = np.linspace(-1.5, 1.5, Np) * kV

phi[:, 0] = init_phi
dW[:, 0] = init_dW
W_s[0] = init_dsgn_E
beta_s[0] = calc_beta(init_dsgn_E)

for i in range(1, Ng):
    this_beta_s = calc_beta(W_s[i - 1])
    phi[:, i] = phi[:, i - 1] - np.pi * dW[:, i - 1] / pow(this_beta_s, 2) / Ar_mass
    coeff = q * dsgn_gap_volt * transit_tfactor
    dW[:, i] = dW[:, i - 1] + coeff * (np.cos(phi[:, i]) - np.cos(init_dsgn_phi))

    W_s[i] = W_s[i - 1] + coeff * np.cos(init_dsgn_phi)
    beta_s[i] = this_beta_s


# Use relative phase and energy differences to find the longitudinal positions
# of the particles when the design particle hits the gap centers.
gaps = np.array([b * SC.c / 2 / dsgn_freq for b in beta_s]).cumsum()
pos = convert_to_spatial(dW, W_s, phi, init_dsgn_phi, dsgn_freq, gaps)

# ------------------------------------------------------------------------------
#    Plotting/Visualization
# The main figure plots the phase-space for the energy difference and phase
# difference for all particles relative to the design particle.
# A useful plot to be added is the phase over time. This would show the
# synchrotron oscillations as the particles progress through successive gaps.
# The phase space plots can be viewed through the dynamic plots by setting the
# switch to True. This is a quick and dirty animation and for many gaps and
# particles the dynamic plotting will get slow very fast. Use this for quick
# analysis and looking at the phase space trajectory.
# ------------------------------------------------------------------------------
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
