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
