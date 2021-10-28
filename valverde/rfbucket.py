""" simulate some of the experimental data """

import concurrent.futures
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
design_phase = -np.pi / 6
dsgn_initE = 7 * kV
Np = 10000

# Simulation parameters for gaps and geometries
design_gap_volt = 7 * kV * 0.1
design_freq = 13.6 * MHz
design_omega = 2 * np.pi * design_freq
Ng = 50
gap_pos = []

# Initialize simulation by setting up first gap commensurate with the design
# particle. Gaps are initialized to have peak output at t=0
dsgn_pos = np.zeros(Ng + 1)
dsgn_E = np.zeros(Ng + 1)
dsgn_time = np.zeros(Ng + 1)

dsgn_pos[0] = 0.0
dsgn_E[0] = dsgn_initE
dsgn_time[0] = 0.0

vstart = np.sqrt(2 * dsgn_initE / Ar_mass) * SC.c
init_gap = vstart / design_omega * (2 * np.pi - design_phase)
t1 = init_gap / vstart
dsgn_time[1] = t1
dsgn_pos[1] = init_gap
Egain = design_gap_volt * np.cos(design_omega * t1)
dsgn_E[1] = dsgn_E[0] + Egain

# Distribute uniformly for one centered on design particle.
beta_lambda = vstart / design_freq
particle_dist = np.linspace(init_gap - beta_lambda, init_gap - beta_lambda / 2, Np)

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


# Make phase space plots using delta E and phase from design particle
delta_E = parts_E.copy()
delta_time = parts_time.copy()
delta_phase = np.zeros(shape=(Np, Ng + 1))
parts_phase = design_omega * parts_time

for i in range(len(dsgn_E)):
    delta_E[:, i] = parts_E[:, i] - dsgn_E[i]
    delta_time[:, i] = parts_time[:, i] - dsgn_time[i]
    dphi = design_omega * delta_time[:, i]
    delta_phase[:, i] = dphi

dsgn_phase = dsgn_time * design_omega

# Create and save plots to pdf
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
    for i in range(1, Ng):
        fig, ax = plt.subplots()
        ax.set_title(
            fr"Phase Space for Gap {i}, ${{\cal E}}_s$ = {dsgn_E[i]/kV:.2f} [kV]"
        )
        ax.set_xlabel(r"$\phi/2\pi$")
        ax.set_ylabel(r"$\Delta {\cal E}$ [keV]")

        Eselect = abs(delta_E[:, i] / dsgn_E[i]) < 0.20
        part_frac = np.sum(Eselect) / Np

        ax.scatter(parts_phase[Eselect, i] / twopi, delta_E[Eselect, i] / kV, s=2)
        black_proxy = plt.Rectangle((0, 0), 0.5, 0.5, fc="k")
        ax.legend([black_proxy], [f"Np Frac Remaining: {part_frac:.2f}"])
        plt.tight_layout()
        pdf.savefig()
        plt.close()
