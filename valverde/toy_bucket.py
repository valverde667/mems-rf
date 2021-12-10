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


def calc_synch_ang_freq(f, V, phi_s, W_s, T=1, g=2 * mm, m=Ar_mass, q=1):
    """Calculate synchrotron angular frequency for small acceleration and phase diff.

    The formula assumes that the velocity of the design particle is
    approximately constants (beta_s and gamma_s ≈ const.) and that other
    particles only slightly deviate from the design particle in phase.
    This leads to a linear second order ODE and a conservation of phase space.
    The wavenumber and hence frequency is a function of the input parameters
    listed below.

    Parameters:
    -----------
    f [Hz units] : float
        RF frequency for the system.
    V [Volts units] : float
        The voltage amplitude applied to the RF gap.
    phi_s : float
        Design frequency. For the harmonic expression of the 2nd Order ODE, this
        value should be negative.
    W_s [eV units] : float
        The kinetic energy of the design particle. This is assumed to remain
        constant throughout the acceleration gaps.
    T : float
        Transit time factor. Can be treated as a parameter to be varied but
        initialized to 1. Values are between [0,1].
    g [mm units] : float
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


def calc_Hamiltonian(
    phi_s, W_s, dW, phi, f, V, T=1, g=2 * mm, m=Ar_mass, q=1, giveAB=True
):
    """Calculate Hamiltonian for longitudinal motion.

    The Hamiltonian for the longitudinal motion that doesn't assume small angle
    excursions about the synchronous particle is
        H = Aw^2/2 + B [sin(phi) - phi * cos(phi_s)]
    where w = dW/mc^2, A = 2pi / lambda_rf / (gamma_s^3 beta_s^3) and
    B = qE0T/mc^2.

    Parameters:
    -----------
    phi_s : float
        The design frequency used. Assumed not to change.
    W_s [eV units] : float
        The kinetic energy of the design particle. Assumed to be in units of eV.
    dphi : float
        The initil phase deviation from the design particle.
    dW [eV units] : float
        The initial kinetic energy deviation form the design particle. Assumed
        to be in units of eV.
    f [Hz units] : float
        RF frequency for the system. Units of Hz
    V [Volts units] : float
        The voltage amplitude applied to the RF gap.
    T : float
        Transit time factor. Can be treated as a parameter to be varied but
        initialized to 1. Values are between [0,1].
    g [mm units] : float
        Thickness of accelerating gap.

    Return:
    -------
    max_excursion : tuple
        The max deviations in kinetic energy (first entry) and phase (second).
        The last enetry is the Hamiltonian.
    """

    # Evaluate and initialize some easy variables
    beta_s = calc_beta(W_s, mass=m, q=q)
    lambda_rf = SC.c / f
    E0 = V / g

    # Evaluate A-coefficient for energy assuming non-rel (gamma ≈ 1)
    A = 2 * np.pi / lambda_rf / beta_s / beta_s / beta_s
    B = q * E0 * T / m

    # Evaluat parts of Hamiltonian
    potential = B * (np.sin(phi) - phi * np.cos(phi_s))
    kinetic = A * pow(dW / m, 2) / 2

    # Calculate Hamiltonian
    H = kinetic + potential

    if giveAB:
        return H, A, B
    else:
        return H


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


def multiple_formatter(denominator=4, number=np.pi, latex="\pi"):
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den * x / number))
        com = gcd(num, den)
        (num, den) = (int(num / com), int(den / com))
        if den == 1:
            if num == 0:
                return r"$0$"
            if num == 1:
                return r"$%s$" % latex
            elif num == -1:
                return r"$-%s$" % latex
            else:
                return r"$%s%s$" % (num, latex)
        else:
            if num == 1:
                return r"$\frac{%s}{%s}$" % (latex, den)
            elif num == -1:
                return r"$\frac{-%s}{%s}$" % (latex, den)
            else:
                return r"$\frac{%s%s}{%s}$" % (num, latex, den)

    return _multiple_formatter


def plot_initial_bucket(phases, phi_s, plot_ax, format=multiple_formatter()):
    """Show particle locations along RF phase curve from -pi to pi."""
    x = np.linspace(-np.pi, np.pi, 100)
    plot_ax.plot(x, np.cos(x))
    plot_ax.scatter(phases, np.cos(phases), c="k")
    plot_ax.scatter(
        phi_s, np.cos(phi_s), c="g", label=fr"$\phi_s$ = {phi_s/np.pi:.4f}$\pi$"
    )
    plot_ax.grid(alpha=0.5, ls="--")
    plot_ax.xaxis.set_major_locator(plt.MultipleLocator(np.pi / 4))
    plot_ax.xaxis.set_minor_locator(plt.MultipleLocator(np.pi / 12))
    plot_ax.xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
    plot_ax.legend()


def calc_transit_factor(W_s, f, g=2 * mm, q=1, mass=Ar_mass):
    """ "Calculate transit time factor for even-symmetric gap"""

    # Calculate the design particle beta and RF-wavelength
    beta_s = calc_beta(W_s, mass=m, q=q)
    lambda_rf = SC.c / f

    # Calculate argument of sine that is also denominator in expression
    argument = np.pi * g / beta_s / lambda_rf

    transit_factor = np.sin(argument) / argument

    return transit_factor


# ------------------------------------------------------------------------------
#     Simulation Parameters/Settings
# This section sets various simulation parameters. In this case, initial kinetic
# energies, gap geometries, design settings such as frequency or phase, etc.
# ------------------------------------------------------------------------------
init_dsgn_E = 7 * keV
init_E = 7 * keV
init_dsgn_phi = -np.pi / 6
phi_dev = np.pi / 20
W_dev = 0.5 * kV
q = 1
Np = 25

Ng = 200
dsgn_freq = 13.6 * MHz
dsgn_gap_volt = 7 * kV * 0.001
dsgn_gap_width = 2 * mm
dsgn_DC_Efield = dsgn_gap_volt / dsgn_gap_width
transit_tfactor = 1.0

# Advance particles using initial conditions
phi = np.zeros(shape=(Np, Ng))
dW = np.zeros(shape=(Np, Ng))
W_s = np.zeros(Ng)
beta_s = np.zeros(Ng)
# init_phi = np.linspace(init_dsgn_phi - phi_dev, init_dsgn_phi + phi_dev, Np)
init_phi = np.linspace(-np.pi, np.pi, Np)
init_dW = np.linspace(-W_dev, W_dev, Np)

phi[:, 0] = init_phi
dW[:, 0] = init_dW
W_s[0] = init_dsgn_E
beta_s[0] = calc_beta(init_dsgn_E)

# Switch to control whether or not the synchronous beta should be updated.
# Setting True will evaluate beta_s at each gap and the phase-space can no
# longer be considered conserved
update_beta_s = True

# ------------------------------------------------------------------------------
#     Simulation and particle advancement of differences
# The differences in phase and differences in kinetic energies between the design
# particle and not are incremented rather than computing the inidividual phases
# and energy then taking the difference. This is to see if there is any
# difference (I'd imagine not) and check understanding.
# ------------------------------------------------------------------------------

# Loop through each gap and calculate energy difference and phase difference.
for i in range(1, Ng):
    if update_beta_s:
        this_beta_s = calc_beta(W_s[i - 1])
    else:
        this_beta_s = calc_beta(W_s[0])

    phi[:, i] = (
        phi[:, i - 1] - np.pi * dW[:, i - 1] / Ar_mass / this_beta_s / this_beta_s
    )

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
# Create phase-space plots without accounting for RF cycles
make_pdfs = False
if make_pdfs:
    today = datetime.datetime.today()
    date_string = today.strftime("%m-%d-%Y_%H-%M-%S_")
    with PdfPages(f"phase-space-plots_{date_string}.pdf") as pdf:
        plt.figure()
        plt.axis("off")
        plt.text(0.5, 1.0, "Simulation Characteristics")
        plt.text(0.5, 0.9, f"Injection Energy: {init_dsgn_E/kV} [keV]")
        plt.text(0.5, 0.8, fr"Synchronous Phase: varied")
        plt.text(0.5, 0.7, f"Gap Voltage: {dsgn_gap_volt/kV:.2f} [kV]")
        plt.text(0.5, 0.6, f"RF Frequency: {dsgn_freq/MHz:.2f} [MHz]")
        plt.text(0.5, 0.5, r"Vary $\phi_s$ with approx const acceleration.")
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        phi_s_list = np.array([-1 / 2, -1 / 4, -1 / 6, -1 / 8, 0]) * np.pi
        for init_dsgn_phi in phi_s_list:

            phi = np.zeros(shape=(Np, Ng))
            dW = np.zeros(shape=(Np, Ng))
            W_s = np.zeros(Ng)
            beta_s = np.zeros(Ng)
            init_phi = init_dsgn_phi * np.linspace(0.99, 1.01, Np)
            init_dW = np.linspace(-1.5, 1.5, Np) * kV * 0

            phi[:, 0] = init_phi
            dW[:, 0] = init_dW
            W_s[0] = init_dsgn_E
            beta_s[0] = calc_beta(init_dsgn_E)

            for i in range(1, Ng):
                this_beta_s = calc_beta(W_s[0])
                phi[:, i] = (
                    phi[:, i - 1] - np.pi * dW[:, i - 1] / pow(this_beta_s, 2) / Ar_mass
                )
                coeff = q * dsgn_gap_volt * transit_tfactor
                dW[:, i] = dW[:, i - 1] + coeff * (
                    np.cos(phi[:, i]) - np.cos(init_dsgn_phi)
                )

                W_s[i] = W_s[i - 1] + coeff * np.cos(init_dsgn_phi)
                beta_s[i] = this_beta_s

            # Make panel plots. First plot is visual aid of initial bucket on the
            # RF phase diagram. The second plot will be the initial condition in
            # dphi and dW. The last plot is the phase-space.
            fig, ax = plt.subplots(nrows=3, figsize=(8, 12))
            plot_initial_bucket(init_phi, init_dsgn_phi, ax[0])
            ax[1].scatter(phi[:, 0] - init_dsgn_phi, dW[:, 0] / kV)
            ax[1].set_title("Initial Conditions for All Particles")
            ax[1].set_ylabel(r"$\Delta W$ [keV]")
            ax[1].set_xlabel(
                fr"$\Delta \phi$ [rad], $\phi_s =$ {init_dsgn_phi/np.pi:.3f} $\pi$"
            )
            for i in range(Np):
                ax[2].scatter(phi[i, :] - init_dsgn_phi, dW[i, :] / kV)
            ax[2].set_ylabel(r"$\Delta W$ [keV]")
            ax[2].set_xlabel(
                fr"$\Delta \phi$ [rad], $\phi_s =$ {init_dsgn_phi/np.pi:.3f} $\pi$"
            )
            ax[2].axhline(y=0, c="k", ls="--", lw=1)
            ax[2].axvline(x=0, c="k", ls="--", lw=1)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

# Create dynamic plotting to visualize individual particle trajectories. Should
# select a few particles and a few gaps to plot since plotting will start to
# run long.
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

Hi, A, B = calc_Hamiltonian(
    init_dsgn_phi, W_s[0], dW[:, 0], phi[:, 0], dsgn_freq, dsgn_gap_volt
)
Hf = calc_Hamiltonian(
    init_dsgn_phi,
    W_s[0],
    dW[:, -1],
    phi[:, -1],
    dsgn_freq,
    dsgn_gap_volt,
    giveAB=False,
)
# Plot the phase space using the max deviations found from the initial conditions.
fig, ax = plt.subplots()
ax.axhline(y=0, c="k", ls="--", lw=1)
ax.axvline(x=0, c="k", ls="--", lw=1)
ax.set_xlim(-1.1, 1.1)
max_dW = (
    np.sqrt(4 * B / A * (init_dsgn_phi * np.cos(init_dsgn_phi) - np.sin(init_dsgn_phi)))
    * Ar_mass
)
ax.axhline(y=max_dW / kV, c="r", ls="--", lw=1)
ax.axhline(y=-max_dW / kV, c="r", ls="--", lw=1)
# max_dW = np.max(abs(max_dev_dW))
# max_dphi = np.max(abs(max_dev_dphi))
# ax.set_xlim(-1.0 * max_dphi / np.pi, 1.0 * max_dphi / np.pi)
# ax.set_ylim(-1.0 * max_dW / keV, 1.0 * max_dW / keV)

for i in range(0, Np, 1):
    ax.scatter((phi[i, :] - init_dsgn_phi) / np.pi, dW[i, :] / keV, c="k", s=1)

ax.scatter([0.0], [0.0], c="k", s=10)
ax.set_xlabel(fr"$\Delta \phi/$$\pi$, $\phi_s =$ {init_dsgn_phi/np.pi:.3f}$\pi$")
ax.set_ylabel(r"$\Delta W$ [keV]")
ax.legend()
plt.tight_layout()
plt.savefig("/Users/nickvalverde/Desktop/bucket", dpi=400)
plt.show()

# Make plot of the energy difference over time to see evolutuion
# fig, ax = plt.subplots()
# ax.set_title(
#     fr"Energy Deviations From Design Particle at Each Gap for $\phi_s =$ {init_dsgn_phi/np.pi:.3f}$\pi$"
# )
# ax.axhline(y=0, c="k", ls="--", lw=1)
# # ax.set_ylim(-1.0 * max_dW / keV, 1.0 * max_dW / keV)
# gap_ind = np.array([i + 1 for i in range(Ng)], dtype=int)
# for i in range(0, Np, 1):
#     ax.scatter(gap_ind, dW[i, :] / keV, c="k", s=1)
#     ax.plot(gap_ind, dW[i, :] / keV, c="k", lw=0.8)
#
# ax.set_xlabel("Acceleration Gap")
# ax.set_ylabel(r"$\Delta W$ [keV]")
# plt.tight_layout()
# plt.show()
garbage
# Check Hamiltonian conservation
per_diff = abs((Hf - Hi) / Hi) * 100
per_diff_W = abs((dW[:, -1] - dW[:, 0]) / dW[:, 0])

fig, ax = plt.subplots()
ax.plot(phi[:, 0] / np.pi, per_diff)
ax.set_title("Percent Difference Between Initial and Final H by Initial Phase Choice")
ax.set_ylabel(r"Percent Difference ($H_f - H_i$)")
ax.set_xlabel(r"Initial Phase Choice [$\pi$-units]")
plt.show()

fig, ax = plt.subplots()
ax.plot(phi[:, 0] / np.pi, per_diff_W)
ax.set_title(
    "Percent Difference Between Initial and Final Energy by Initial Phase Choice"
)
ax.set_ylabel(r"Percent Difference ($\Delta W_f - \Delta W_i$)")
ax.set_xlabel(r"Initial Phase Choice [$\pi$-units]")
plt.show()
garbage
# Plot changes in Hamiltonian for farthest (left) particle
fig, ax = plt.subplots()
H_at_gaps = np.zeros(Ng)
for i in range(Ng):
    this_H = calc_Hamiltonian(
        init_dsgn_phi, W_s[i], dW[0, i], phi[0, i], dsgn_freq, dsgn_gap_volt
    )
    H_at_gaps[i] = this_H

ax.scatter(np.arange(Ng) + 1, H_at_gaps, c="k", s=1)
ax.set_xlabel("Gap Index")
ax.set_ylabel(r"$H_\phi$ at Each Gap")
ax.set_title(
    fr"Evolution of the Hamiltonian for a Selected Particle at $\Delta \phi_i$ = {(phi[0,0] - init_dsgn_phi)/np.pi:.3f}"
)
plt.show()
