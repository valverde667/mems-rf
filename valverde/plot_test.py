import numpy as np
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.animation as animation
import datetime
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy import integrate
import scipy.constants as SC
import scipy.optimize as optimize
import time
import pdb
import os

plt.style.use("seaborn-deep")


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


def elliptical_load(Np, xmax, ymax):
    """Create Np particles within ellipse with axes xmax and ymax"""
    keep_x = np.zeros(Np)
    keep_y = np.zeros(Np)
    kept = 0
    while kept < Np:
        x = np.random.uniform(-xmax, xmax)
        y = np.random.uniform(-ymax, ymax)
        coord = np.sqrt(pow(x / xmax, 2) + pow(y / ymax, 2))
        if coord <= 1:
            keep_x[kept] = x
            keep_y[kept] = y
            kept += 1

    return (keep_x, keep_y)


Np = int(1e6)
xmax = np.pi
ymax = 3 * keV
x, y = elliptical_load(Np, xmax, ymax)

fig, ax = plt.subplots()
hist = ax.hist2d(x / np.pi, y / keV, bins=[150, 150])
fig.colorbar(hist[3], ax=ax)
ax.set_title("Control Plot")
ax.set_xlabel(fr"$\phi / \pi$")
ax.set_ylabel(r"Kinetic Energy $W$ [keV]")


# Use modulo
# inds = np.where(np.sign(x) < 0)[0]
xmod = x % np.pi
# xmod[inds] -= np.pi

fig, ax = plt.subplots()
hist = ax.hist2d(xmod / np.pi, y / keV, bins=[150, 150])
fig.colorbar(hist[3], ax=ax)
ax.set_title("Taking Modulo")
ax.set_xlabel(fr"$\phi / \pi$")
ax.set_ylabel(r"Kinetic Energy $W$ [keV]")
plt.show()
