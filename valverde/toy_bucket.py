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
#            Constants and definitions section
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


# ------------------------------------------------------------------------------
#           Simulation Parameters/Settings
# This section sets various simulation parameters. In this case, initial kinetic
# energies, gap geometries, design settings such as frequency or phase, etc.
# ------------------------------------------------------------------------------
init_dsgn_E = 7 * keV
init_E = 7 * keV
init_dsgn_phi = -np.pi / 2
init_phi = -np.pi / 4

dsgn_freq = 13.6 * MHz
dsgn_gap_volt = 7 * kV
dsgn_gap_width = 2 * mm
dsgn_DC_Efield = dsgn_gap_volt / dsgn_gap_width
transit_tfactor = 1.0
