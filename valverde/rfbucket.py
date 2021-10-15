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

    # Set starting conditions. Columns are position, energy, and time
    StartingCondition[: int(num_parts / 2), 0] = np.linspace(
        -xmin / 2, 0, int(num_parts / 2)
    )
    StartingCondition[int(num_parts / 2) :, 0] = np.linspace(
        0, xmin / 2, int(num_parts / 2)
    )
    StartingCondition[:, 1] = E
    StartingCondition[:, 2] = 0.0

    return StartingCondition


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
