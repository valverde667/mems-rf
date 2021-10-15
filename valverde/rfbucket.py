""" simulate some of the experimental data """

import concurrent.futures
import numpy as np
import matplotlib

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


# Simulation parameters
init_energy = 3 * kV
rf_volt = 7 * kV
rf_offset = 0
rf_freq = 13.6 * MHz
Ngaps = 3
Nparticles = 100
beam_length = 1 / rf_freq
synch_phase = np.pi / 4
centers = setup_gaps(synch_phase, init_energy, rf_freq, rf_volt, Ngaps=Ngaps)
beam = setup_beam(init_energy, Nparticles, beam_length)

# Simulate particles
particles, energies, phases = trace_particles(
    centers, rf_freq, rf_volt, beam, phase_offset=synch_phase
)

# Grab design particle characterisitcs
design_ind = int(Nparticles / 2)
design_part = particles[design_ind, :]
design_energies = energies[design_ind, :]
design_phases = phases[design_ind, :]
