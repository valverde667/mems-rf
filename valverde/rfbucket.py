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


def beta(E, mass=Ar_mass, q=1, nonrel=True):
    """Velocity of a particle with energy E."""
    if nonrel:
        beta = np.sqrt(2 * E / mass)
    else:
        gamma = (E + mass) / mass
        beta = np.sqrt(1 - 1 / gamma / gamma)

    return beta


def load_lattice(
    synch_phase,
    init_synch_energy,
    rf_freq,
    rf_volt,
    Ngaps=1,
    g=2 * mm,
    mass=Ar_mass,
    q=1,
    Fcup_dist=50 * mm,
    verbose=False,
):
    """ "Place RF-Gap centers for acceleration

    The design particle is initialized at z=0 with t=0. The gaps are then placed
    such that the design particle arrives at each gap at the set synchronous
    phase. At each gap, the design particle's energy is incremented based on
    the RF-voltage and synchronous phase. """

    # Calculate start position for synchronous particle
    bc = beta(init_synch_energy, mass, q) * SC.c

    gap_centers = []
    Fcup = [50 * mm]
    const = SC.c / 2 / rf_freq
    E = init_synch_energy
    for i in range(Ngaps):
        this_gap = const * beta(E, mass, q)
        gap_centers.append(this_gap)

        E_gain = rf_volt * np.cos(synch_phase)
        E += E_gain

    pos = np.array(gap_centers + Fcup)
    pos = pos.cumsum()

    return pos


def load_particles(synch_phase, synch_energy, num_parts):
    """Create injection of particles based on synchronous particle"""


# Simulation paramters
init_energy = 3 * kV
rf_volt = 7 * kV
rf_offset = 0
rf_freq = 13.6 * MHz
Ngaps = 3
synch_phase = np.pi / 4
centers = load_lattice(synch_phase, init_energy, rf_freq, rf_volt, Ngaps=Ngaps)
print(centers / mm)


#
