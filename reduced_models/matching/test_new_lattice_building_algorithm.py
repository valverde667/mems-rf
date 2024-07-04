# Model the transverse dynamics by solving the KV-envelope equations. This model
# uses the extracted fields from both the acceleration gaps and the optimized
# ESQ models to integrate the KV equations. Additionally, there are options to
# find the optimized voltage or 4D coordinates by using the Nelder-Mead search
# algorithm.

import numpy as np
import os
import scipy.optimize as sciopt
import itertools
import scipy.constants as sc
import matplotlib.pyplot as plt
import matplotlib as mpl
import pdb

import matching_utility as util

mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.minor.top"] = True
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["ytick.right"] = True
mpl.rcParams["ytick.major.right"] = True
mpl.rcParams["ytick.minor.right"] = True

# Define useful constants
mm = 1e-3
mrad = 1e-3
um = 1e-6
kV = 1e3
mrad = 1e-3
keV = 1e3
uA = 1e-6
MHz = 1e6
amu = sc.atomic_mass
twopi = np.pi * 2

grad_q = np.load("normed_esq_gradient.npy")
grad_z = np.load("normed_esq_z.npy")
gap_Ez = np.load("normalized_iso_Ez.npy")
gap_z = np.load("normalized_iso_z.npy")

# Beam parameters
mass = 39.948 * amu * pow(sc.c, 2) / sc.elementary_charge  # eV
init_E = 7 * keV

# Gap parameters
g = gap_z.max() - gap_z.min()  # Includes fringe length and 2*mm physical spacing.
phi_s = np.pi * np.array([0.0, 0.0, 0.0])
gap_mode = np.zeros(len(phi_s))

freq = 13.5 * MHz
Vg = 6 * kV * 1.007
Vaccel = abs(np.cos(phi_s))[:-1] * Vg

gap_centers = util.calc_gap_centers(init_E, mass, phi_s, gap_mode, freq, Vg)
scheme = "gg-qq"


def build_lattice(zgap, zfield, centers, dz=10 * um):
    # Calculate some key variables
    zext = zgap.max() - zgap.min()
    z = []
    data = []

    for gc in centers:
        # Calculate stitch points
        z1 = gc - zgap[-1] - dz
        z2 = gc + zgap[-1] - dz

        znew = np.linspace(0, z1 + dz, dz)
        newdata = np.zeros(len(znew))

        # Attach data
        np.hstack((znew, zgap + gc))
        np.hstack((newdata, zfield))

        z.append(znew)
        data.append(newdata)

    return z, data


z, data = build_lattice(gap_z, gap_Ez, gap_centers)
#
#
# # Geometric settings for lattice
# aperture = 0.55 * mm
# scheme = "g-g-q-q"
# gap_centers = util.calc_gap_centers(init_E, mass, phi_s, gap_mode, freq, Vg)
# gap_centers = gap_centers - gap_centers.min() + g / 2.0
# zstart = 0.0
# zend = gap_centers[-1] - g / 2.0
# quad_centers = util.calc_quad_centers(gap_centers, lq, separation, g, "equal")
