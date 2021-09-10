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


def beta(E, mass=Ar_mass, q=1):
    """Velocity of a particle with energy E."""
    gamma = (E + mass) / mass
    beta = np.sqrt(1 - 1 / gamma / gamma)
    return beta


def beam_setup(E, packages, pulse_length, mass=Ar_mass, q=1):
    """Starting conditions for a beam of a given pulse length and energy"""

    # store information in a 2d array
    StartingCondition = np.zeros(shape=(packages, 3))

    xmin = beta(E, mass, q) * SC.c * pulse_length  # beam pulse is [-xmim, 0]

    # set starting conditions
    StartingCondition[:, 0] = np.linspace(-xmin, 0, packages)  # x-coordinate
    StartingCondition[:, 1] = E  # energy
    StartingCondition[:, 2] = 0.0  # time

    return StartingCondition


def trace_particles(
    pos, f, V, StartingCondition, mass=Ar_mass, q=1, d=0.75e-3, steps=1
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
    # Create vector of initial positions
    initial_pos = particles.copy()[:, 0]
    initial_pos = initial_pos[:, np.newaxis]
    # Add vector to particle matrix as final column
    particles = np.hstack((particles, initial_pos))
    # Create history and phase arrays
    energy_history = [particles.copy()[:, 1]]
    phase_history = [0 * particles.copy()[:, 1]]

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
                    dE = V * np.sin(2 * np.pi * f * particles[mask, 2]) / steps
                else:
                    dE = -V * np.sin(2 * np.pi * f * particles[mask, 2]) / steps
                particles[mask, 0] += dx
                particles[mask, 1] += dE
                mask = particles[:, 1] > 0

                phase = 2 * np.pi * f * particles.copy()[:, 2] % (2 * np.pi)
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


def wafer_setup(E=10e3, V=1e3, f=15e6, N=1, Fcup_dist=5e-2, mass=Ar_mass, q=1):
    """Return RF wafer position, always add a F-cup at the end"""

    assert N >= 1

    startpos = [0]

    const = SC.c / (2 * f)
    wafers = [const * beta(E + (i + 1) * V, mass, q) for i in range(N - 1)]

    Fcup = [Fcup_dist]

    pos = np.array(startpos + wafers + Fcup)

    # the actual position are the sums of the previous gaps
    pos = pos.cumsum()

    return pos


initial_energy = 7 * kV
design_voltage = 7 * kV
design_frequency = 14.86 * MHz
beam_length = 5e-6
real_energy = 7 * kV
real_voltage = 3 * kV
real_frequency = 14.86 * MHz
d = 2 * mm

packages = 1
steps = 1
gaps = 1
beam = beam_setup(initial_energy, packages, beam_length)
pos = wafer_setup(E=initial_energy, V=design_voltage, f=design_frequency, N=gaps)
# pdb.set_trace()
t, energies, phases = trace_particles(
    pos=pos, f=real_frequency, V=real_voltage, StartingCondition=beam, d=d, steps=steps
)
