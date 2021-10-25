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


def setup_gaps(
    synch_phase,
    init_synch_energy,
    rf_freq,
    rf_volt,
    Ngaps=1,
    g=2 * mm,
    mass=Ar_mass,
    q=1,
    Fcup_dist=50 * mm,
):
    """ "Place RF-Gap centers for acceleration

    The design particle is initialized at z=0 with t=0. The gaps are then placed
    such that the design particle arrives at each gap at the set synchronous
    phase. At each gap, the design particle's energy is incremented based on
    the RF-voltage and synchronous phase. """

    gap_centers = []
    start_pos = [0.0]
    Fcup = [50 * mm]
    const = SC.c / 2 / rf_freq
    E = init_synch_energy
    for i in range(Ngaps - 1):
        # First gap is at z=0. This loop starts at placing next gap.
        E_gain = rf_volt * np.cos(synch_phase)
        E += E_gain

        this_gap = const * beta(E, mass, q)
        gap_centers.append(this_gap)

    pos = np.array(start_pos + gap_centers + Fcup)
    pos = pos.cumsum()

    return pos


def setup_beam(E, num_parts, pulse_length, mass=Ar_mass, q=1):
    """Starting conditions for a beam of a given pulse length and energy"""

    # store information in a 2d array
    StartingCondition = np.zeros(shape=(num_parts, 3))

    xmin = beta(E, mass, q) * SC.c * pulse_length  # beam pulse is [-xmim, 0]

    # Set starting conditions. Columns are position, energy, and time/phase
    StartingCondition[: int(num_parts / 2), 0] = np.linspace(
        -xmin / 2, 0, int(num_parts / 2)
    )
    StartingCondition[int(num_parts / 2) :, 0] = np.linspace(
        0, xmin / 2, int(num_parts / 2)
    )
    StartingCondition[:, 1] = E
    StartingCondition[:, 2] = 0.0

    return StartingCondition


def trace_particles(
    pos,
    f,
    V,
    StartingCondition,
    phase_offset=0.0,
    mass=Ar_mass,
    q=1,
    d=0.75e-3,
    steps=1,
):
    """Simulate particles acclerated through RF-gaps.

    The rf-wafers will be at the position given in pos, the last entry
    in pos will be assumed to be a F-cup, so no acceleration will be
    calculated for this one.
    The simulation will assume a frequency f and an amplitude V.

    d      is the thickness of the acceleration gap and
    steps gives the number of steps inside the acceleration gaps

    In case the wavelength is not large compared to d, steps needs to be increased!

    Returns
    -------
    particles : ndarray
        rows represent the particles. columns represent initial position, final
        position, energy, and time.
    energy_history : ndarray
        rows represent particles. Columns represent the energy gain per step in
        the acceleration gap. 0th column is the initial energy.
    phase_history : ndarray
        rows represent particles. Columns represent the phase at which particle
        recieved acceleration. 0th column set to 0.

    """

    # Create history arrays for particle energy and phase
    particles = StartingCondition.copy()

    # Create history and phase arrays
    energy_history = [particles.copy()[:, 1]]
    phase_history = []

    for i, x in enumerate(pos):
        mask = particles[:, 1] > 0
        particles[mask, 2] += (x - d / 2 - particles[mask, 0]) / (
            beta(particles[mask, 1], mass, q) * SC.c
        )
        particles[mask, 0] = x - d / 2
        # last element is the F-cup, no change in E
        if x != pos[-1]:
            # the rf-gaps alternate in pull and push
            for a in range(steps):
                dx = d / (steps + 1)
                dt = dx / beta(particles[mask, 1], mass, q) / SC.c
                particles[mask, 2] += dt
                if i % 2 == 0:
                    dE = (
                        V
                        * np.cos(2 * np.pi * f * particles[mask, 2] + phase_offset)
                        / steps
                    )
                else:
                    dE = (
                        -V
                        * np.cos(2 * np.pi * f * particles[mask, 2] + phase_offset)
                        / steps
                    )
                particles[mask, 0] += dx
                particles[mask, 1] += dE
                mask = particles[:, 1] > 0

                phase = 2 * np.pi * f * particles.copy()[:, 2] + phase_offset
                phase %= np.pi
                energy_history.append(particles.copy()[:, 1])
                phase_history.append(phase)
        else:
            # do another d/2 step to get to the final position
            dx = d / 2
            dt = dx / beta(particles[mask, 1], mass, q) / SC.c
            particles[mask, 2] += dt
            particles[mask, 0] += dx

    energy_history = np.array(energy_history)
    phase_history = np.array(phase_history)

    return particles, energy_history.T, phase_history.T


def calc_pires(energy, freq, mass=Ar_mass, q=1):
    beta_lambda = beta(energy, mass=mass, q=q) * SC.c / freq
    return beta_lambda / 2


# Simulation parameters
# beam_length_sets = np.linspace(0.1, 1, 10) / 13.6 / MHz
# for beam_length in beam_length_sets:
#     init_energy = 3 * kV
#     rf_volt = 7 * kV
#     rf_offset = 0
#     rf_freq = 13.6 * MHz
#     synch_phase = np.pi / 4
#     Ngaps = 10
#     Nparticles = 10000
#     centers = setup_gaps(synch_phase, init_energy, rf_freq, rf_volt, Ngaps=Ngaps)
#     beam = setup_beam(init_energy, Nparticles, beam_length)
#
#     # Simulate particles
#     particles, energies, phases = trace_particles(
#         centers, rf_freq, rf_volt, beam, phase_offset=synch_phase
#     )
#
#     # Grab design particle characterisitcs
#     design_ind = int(Nparticles / 2)
#     design_part = particles[design_ind, :]
#     design_energies = energies[design_ind, :]
#     design_phases = phases[design_ind, :]
#
#     # Find differences in phase and energy from design particles
#     delta_phases = phases - design_phases
#     delta_energies = energies[:, 1:] - design_energies[1:]
#
#     # Create phase-space plots in delta phi and delta E. Save to pdfs.
#     today = datetime.datetime.today()
#     date_string = today.strftime("%m-%d-%Y_%H-%M-%S_")
#
#     with PdfPages(f"phase-space-plots_{date_string}.pdf") as pdf:
#         plt.figure()
#         plt.axis("off")
#         plt.text(0.5, 1.0, "Simulation Characteristics")
#         plt.text(0.5, 0.9, f"Injection Energy: {init_energy/kV} [kV]")
#         plt.text(0.5, 0.8, fr"Synchronous Phase: {synch_phase/np.pi:.3f} $\pi$")
#         plt.text(0.5, 0.7, f"Gap Voltage: {rf_volt/kV:.2f} [kV]")
#         plt.text(0.5, 0.6, f"RF Frequency: {rf_freq/MHz:.2f} [MHz]")
#         plt.text(0.5, 0.5, fr"Beam Length: {beam_length*rf_freq:.2f} $\tau_{{rf}}$")
#         plt.tight_layout()
#         pdf.savefig()
#         plt.close()
#
#         # Save phase-space plot for each gap to pdf
#         for i in range(delta_phases.shape[-1]):
#             fig, ax = plt.subplots()
#             ax.set_title(f"Phase Space for Gap {i+1}")
#             ax.set_xlabel(r"$\Delta \phi$ [rad]")
#             ax.set_ylabel(r"$\Delta {\cal E}$ [kV]")
#             ax.scatter(delta_phases[:, i], delta_energies[:, i] / kV, s=2)
#             pdf.savefig()
#             plt.close()

# Simulation Parameters for design particle
design_phase = -np.pi / 6
dsgn_initE = 7 * kV
mass = 39.948 * amu  # [eV]
Np = 1000

# Simulation parameters for gaps and geometries
design_gap_volt = 7 * kV
design_freq = 13.6 * MHz
design_omega = 2 * np.pi * design_freq
Ng = 3
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

# Distribute particles around design particle. Model CW beam with full RF load
design_period = 1 / design_freq
rf_lambda = vstart * design_period
particle_dist = np.linspace(-rf_lambda / 2, rf_lambda / 2, Np)

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
    print(newz / mm)
    newv = np.sqrt(dsgn_E[i] * 2 / Ar_mass) * SC.c
    newt = newz / newv
    if i % 2 == 0:
        Egain = design_gap_volt * np.cos(design_omega * (newt + dsgn_time[i]))
    else:
        Egain = -design_gap_volt * np.cos(design_omega * (newt + dsgn_time[i]))

    dsgn_pos[i + 1] = newz
    dsgn_E[i + 1] = dsgn_E[i] + Egain
    dsgn_time[i + 1] = dsgn_time[i] + newt
