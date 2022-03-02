# Simulate injection of particles and model longitudinal phase-space in Energy
# and phase.

import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.constants as SC
import time
import itertools
import pdb
import os

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
ns = 1e-9  # nanoseconds
twopi = 2 * np.pi


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


# Simulation Parameters for design particle
design_phase = -0
dsgn_initE = 7 * kV
Np = 4000

# Simulation parameters for gaps and geometries
design_gap_volt = 7 * kV
design_freq = 13.6 * MHz
design_omega = 2 * np.pi * design_freq
Ng = 12

# ------------------------------------------------------------------------------
#     Initial Setup
# Start design particle with design phase at z=0. Place first gap so that design
# particle arrives at design phase. Since gaps are initialized to have peak
# voltage at t=0, the first gap is placed such that the field oscillates one full
# period before arrival of design particle. Other particles can be centered on
# the design particle, or distributed from the gap center towards z=0 or z<0.
# ------------------------------------------------------------------------------

# Initialize simulation by setting up first gap commensurate with the design
# particle. Gaps are initialized to have peak output at t=0
dsgn_pos = np.zeros(Ng + 1)
dsgn_E = np.zeros(Ng + 1)
dsgn_time = np.zeros(Ng + 1)

dsgn_pos[0] = 0.0
dsgn_E[0] = dsgn_initE
dsgn_time[0] = 0.0

# Calculate full DC beam length and start position of design particle. The
# convention here is to start with a gaining gap voltage. Since the design particle
# convention enters the gap when the fielding is going from neg -> pos, the
# first gap needs to be placed a RF cycle away
coeff = np.sqrt(2 * dsgn_initE / Ar_mass)

tDC = 1.0 / design_freq
that = 0.5 / design_freq
ts = (np.pi - design_phase) / 2 / np.pi / design_freq

DC_length = coeff * SC.c * tDC
zhat = coeff * SC.c * that
zs = coeff * SC.c * ts

init_gap = zhat + zs


# Instantiate the design particle metrics to first gap
vs_start = coeff * SC.c
ts_start = init_gap / vs_start
dsgn_time[1] = ts_start
dsgn_pos[1] = init_gap
Egain = design_gap_volt * np.cos(design_omega * ts_start)
dsgn_E[1] = dsgn_E[0] + Egain

# Create simulation particles and initialize data arrays
beta_lambda = vs_start / design_freq
particle_dist = np.linspace(-DC_length / 2, DC_length / 2, Np)

# Create particle arrays to store histories
parts_pos = np.zeros(shape=(Np, Ng + 1))
parts_pos[:, 0] = particle_dist
parts_E = np.zeros(shape=(Np, Ng + 1))
parts_time = np.zeros(shape=(Np, Ng + 1))
parts_E[:, 0] = dsgn_initE

# Advance particles to first gap
vparts = np.sqrt(2 * parts_E[:, 0] / Ar_mass) * SC.c
time = (init_gap - parts_pos[:, 0]) / vparts
parts_Egain = design_gap_volt * np.cos(design_omega * time)
parts_pos[:, 1] = init_gap
parts_time[:, 1] = time
parts_E[:, 1] = parts_E[:, 0] + parts_Egain

# Advance through the rest of the gaps
for i in range(1, Ng):
    newz = calc_pires(dsgn_E[i], design_freq)

    # Update design particle
    dsgn_dv = np.sqrt(2 * dsgn_E[i] / Ar_mass) * SC.c
    dsgn_dt = newz / dsgn_dv

    # Update other particles
    direction = np.sign(parts_E[:, i])
    parts_dv = np.sqrt(2 * abs(parts_E[:, i]) / Ar_mass) * SC.c * direction
    parts_dt = newz / parts_dv

    if i % 2 == 0:
        dsgn_Egain = design_gap_volt * np.cos(design_omega * (dsgn_dt + dsgn_time[i]))
        parts_Egain = design_gap_volt * np.cos(
            design_omega * (parts_dt + parts_time[:, i])
        )
    else:
        dsgn_Egain = -design_gap_volt * np.cos(design_omega * (dsgn_dt + dsgn_time[i]))
        parts_Egain = -design_gap_volt * np.cos(
            design_omega * (parts_dt + parts_time[:, i])
        )

    dsgn_pos[i + 1] = newz
    dsgn_E[i + 1] = dsgn_E[i] + dsgn_Egain
    dsgn_time[i + 1] = dsgn_time[i] + dsgn_dt

    parts_pos[:, i + 1] = newz
    parts_E[:, i + 1] = parts_E[:, i] + parts_Egain
    parts_time[:, i + 1] = parts_time[:, i] + parts_dt


# ------------------------------------------------------------------------------
#     Simulation with Warp Found Fields
# Here the electric field Ez is calculated over the entire simulation mesh for 12
# gaps (≈91 keV at end). The particles are initialized to start at the minimum
# z-point and the advanced backward in time. All particles are then advanced
# along to each gridpoint. The electric field is modulated with a cosine function
# to simulate the time variation, i.e. E(z,t) = E(z) cos(wt). The particles are
# advanced a dz each time and gain an energy E(z,t)dz
# ------------------------------------------------------------------------------
# Load arrays
phi0 = np.load("potential_array.npy")
Ez0 = np.load("field_array.npy")
z = np.load("zmesh.npy")
gap_centers = np.load("gap_centers.npy")
dz = z[1] - z[0]

# Define the cosine energy modulation
rfvolt = lambda t: np.cos(design_omega * t)

# initialze arrays to hold particle information
pos = np.zeros(shape=(Np, len(z)))
energy = np.zeros(shape=(Np, len(z)))
time = np.zeros(shape=(Np, len(z)))
energy[:, 0] = dsgn_initE

# Calculate time to reverse particles all to minimum z coordinate which is
# the lambda_rf / 2
vstart = beta(dsgn_initE) * SC.c
tstart = (particle_dist - z.min()) / vstart
time[:, 0] = tstart
Egain_maxs = np.zeros(len(z))

# Advance particles through the zmesh and add energy gains along the way
# pdb.set_trace()
for i in range(1, len(z)):
    mask = energy[:, i - 1] > 0
    vi = beta(energy[mask, i - 1]) * SC.c
    dt = dz / vi

    time[mask, i] = time[mask, i - 1] + dt
    pos[mask, i] = pos[mask, i - 1] + dz

    Egain = Ez0[i - 1] * rfvolt(time[mask, i]) * dz
    Egain_maxs[i - 1] = Egain.max()
    energy[mask, i] = energy[mask, i - 1] + Egain

# ------------------------------------------------------------------------------
#     Analysis/Plotting Section
# Make plots and analyze phase space distribution of particles after simulation.
# The analysis sets the y-axis in any plots to be change in energy. However,
# on the x-axis it can be change in phase w.r.t. the design particle, or it
# can be the phase over time through multiple RF cycles. The latter case makes
# it easier to see the difference buckets forming and particles dropping out.
# ------------------------------------------------------------------------------
# Initialize empty delta arrays. Loop through data arrays and populate deltas.
delta_E = parts_E.copy()
delta_time = parts_time.copy()
delta_phase = np.zeros(shape=(Np, Ng + 1))
parts_phase = design_omega * parts_time
for i in range(Ng + 1):
    delta_E[:, i] = parts_E[:, i] - dsgn_E[i]
    delta_time[:, i] = parts_time[:, i] - dsgn_time[i]
    dphi = design_omega * delta_time[:, i]
    delta_phase[:, i] = dphi
dsgn_phase = dsgn_time * design_omega


# Set bounds on plotting using output data from toybucket program
min_cross = -1.4899 * np.pi
max_cross = 0.4899 * np.pi
bucket_init_phase = 0.4899
bucket_mask = (delta_phase[:, 1] >= min_cross) & (delta_phase[:, 1] <= max_cross)
separatrix_ind = np.argmax(abs(particle_dist[bucket_mask]))
parts_survived = np.sum(bucket_mask)
max_dW = delta_E[separatrix_ind, :].max()
min_dW = delta_E[separatrix_ind, :].min()

fig, ax = plt.subplots()
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(min_dW / keV, max_dW / keV)
for i in range(Np):
    plt.plot(delta_phase[i, :], delta_E[i, :] / keV, c="k", lw=1)

garbage
# Create and save plots to pdf
today = datetime.datetime.today()
date_string = today.strftime("%m-%d-%Y_%H-%M-%S_")
with PdfPages(f"continuous_phase-space-plots_{date_string}.pdf") as pdf:
    plt.figure()
    plt.axis("off")
    plt.text(0.5, 1.0, "Simulation Characteristics")
    plt.text(0.5, 0.9, f"Injection Energy: {dsgn_initE/kV} [keV]")
    plt.text(0.5, 0.8, fr"Synchronous Phase: {design_phase/np.pi:.3f} $\pi$")
    plt.text(0.5, 0.7, f"Gap Voltage: {design_gap_volt/kV:.2f} [kV]")
    plt.text(0.5, 0.6, f"RF Frequency: {design_freq/MHz:.2f} [MHz]")
    plt.text(
        0.5, 0.5, fr"Beam Length: {(particle_dist[-1] - particle_dist[0])/mm:.2f} [mm]"
    )
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Plot phase-space trajectory of each particle
    for i in range(Np):
        fig, ax = plt.subplots()
        ax.set_title(
            fr"Phase Space for Gap {i}, ${{\cal E}}_s$ = {dsgn_E[i]/kV:.2f} [keV]"
        )
        ax.set_xlabel(r"Time Progression of $\phi/2\pi$")
        ax.set_ylabel(r"$\Delta {\cal E}$ [keV]")

        ax.scatter((delta_phase[i, :]) / np.pi, delta_E[i, :] / kV, s=2)
    black_proxy = plt.Rectangle((0, 0), 0.5, 0.5, fc="k")
    # ax.legend([black_proxy], [f"Np Frac Remaining: {part_frac:.2f}"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

# Create phase-space plots without accounting for RF cycles
today = datetime.datetime.today()
date_string = today.strftime("%m-%d-%Y_%H-%M-%S_")
with PdfPages(f"phase-space-plots_{date_string}.pdf") as pdf:
    plt.figure()
    plt.axis("off")
    plt.text(0.5, 1.0, "Simulation Characteristics")
    plt.text(0.5, 0.9, f"Injection Energy: {dsgn_initE/kV} [keV]")
    plt.text(0.5, 0.8, fr"Synchronous Phase: {design_phase/np.pi:.3f} $\pi$")
    plt.text(0.5, 0.7, f"Gap Voltage: {design_gap_volt/kV:.2f} [kV]")
    plt.text(0.5, 0.6, f"RF Frequency: {design_freq/MHz:.2f} [MHz]")
    plt.text(
        0.5, 0.5, fr"Beam Length: {(particle_dist[-1] - particle_dist[0])/mm:.2f} [mm]"
    )
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    # Save phase-space plot for each gap to pdf
    for i in range(1, 20):
        fig, ax = plt.subplots()
        ax.set_title(
            fr"Phase Space for Gap {i}, ${{\cal E}}_s$ = {dsgn_E[i]/kV:.2f} [keV]"
        )
        ax.set_xlabel(r"$\Delta \phi/2\pi$")
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylabel(r"$\Delta {\cal E}$ [keV]")
        ax.axhline(y=0, ls="--", c="k", lw=1)
        ax.axvline(x=0, ls="--", c="k", lw=1)

        Eselect = abs(delta_E[:, 0:i] / dsgn_E[0:i]) < 0.3
        part_frac = np.sum(Eselect) / Np

        # Transform phase coordinates
        dphi = np.cos(delta_phase.copy())

        ax.scatter((dphi[:, 0:i] - np.pi / 2) / twopi, delta_E[:, 0:i] / kV, s=2)
        black_proxy = plt.Rectangle((0, 0), 0.5, 0.5, fc="k")
        ax.legend([black_proxy], [f"Np Frac Remaining: {part_frac:.2f}"])
        plt.tight_layout()
        pdf.savefig()
        plt.close()
