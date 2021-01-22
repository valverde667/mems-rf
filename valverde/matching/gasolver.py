"""Script uses a genetic algorithm to find the optimized ESQ voltages that will
give the best solution to the KV envelope equations. The best solution is such
that, given the initial coordinates r0 and r0', the coordinates one lattice
period later r and r' will be close to the initial coordinates, i.e.
     r0 ≈ r
     r0' ≈ r'.

Currently the the two ESQs are distributed in the drift space such that there
is equal drift between all elemtns. The drift space is set to 9.3mm (this value
was pulled from the 1D simulation in the MEQALAC code which placed the first
optimzed RF gap for ions of Argon 9.3mm from the origin). The ESQs are .695mm
in length and so the centers are placed at 2.3175 and 4.9825mm. This gives equal
drift lengths between elements of d=1.97mm.

The genetic algorithm is imported from the PYPI library with documentation found
at https://pypi.org/project/geneticalgorithm/."""

import numpy as np
import matplotlib.pyplot as plt
import pdb

import warp as wp

import parameters
from solver import hard_edge_kappa

# Set useful contants
mm = wp.mm
um = 1e-6
kV = wp.kV

# Set parameter values
param_dict = parameters.main()
Q = param_dict["Q"]
k = 5.7e4
emittance = param_dict["emittance"]
ux_initial = param_dict["inj_radius"]
uy_initial = param_dict["inj_radius"]
vx_initial = 5 * mm
vy_initial = -5 * mm


# Set up geometric parameters
d = 9.3 * mm  # Distance from first RF gap center to second
g = 2 * mm  # Gap distance
lq = 0.695 * mm  # Physical length of quadrupoles
N = 501  # Number of grid points
L = 2 * d  # Total length of mesh
ds = L / N

# Create hard edge array for ESQs and plot for visual check.
pos_arr = np.linspace(0, L, N)
karr = hard_edge_kappa(k, pos_arr)
fig, ax = plt.subplots()
ax.plot(pos_arr / mm, karr)
ax.set_xlabel("s [mm]")
ax.set_ylabel(r"$\kappa (s)$ $[m]^{-2}$")
plt.show()
