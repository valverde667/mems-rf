"""Script will search the 4D parameter space for the ESQ section in the
acceleration lattice. This is done by varying the r and r' in x-y for set
voltages V1 and V2. In other words, at some fixed V1 and V2 a specified cost
fucntion is minimized and this minimum value corresponds to the set V1 and V2.
Then, for example, V1 is incremented and the minimum parameter settings are
found once more."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb

import parameters as params
from solver import hard_edge_kappa

# Useful constants
kV = 1e3
mm = 1e-3
mrad = 1e-3

# Parameters (See fodo streamlit script)
d = 9.3 * mm  # Distance between RF gap centers
g = 2 * mm  # RF gap length
lq = 0.695 * mm  # ESQ length
space = (d - g - 2 * lq) / 3  # Spacing between ESQs
Lp = 2 * d  # Lattice period
Vbrkdwn = 3 * kV / mm
s = np.linspace(0, Lp, 500)
ds = s[1] - s[0]
param_dict = params.main()

# Max settings
maxBias = Vbrkdwn * space
maxR = 0.55 * mm
maxDR = maxR / Lp

# Hyperparameter settings
Niter = 1000
lrng_rate = 0.0001
Vsteps = 10
threshold = 0.01  # Cost function most likely wont approach 0 exactly


# Sample gradient descent run. Keeping here for future refernce.
# x = np.linspace(-3, 3, 101)
# y = np.linspace(-3, 3, 101)
# z = x ** 2 + y ** 2
# init = np.array([2, 1.4])
#
# hist = np.zeros(Niter)
# for i in range(Niter):
#     this_x, this_y = init[0], init[1]
#     this_hist = this_x**2 + this_y**2
#     hist[i] = this_hist
#
#     grad = 2*this_x + 2*this_y
#     init -= rate * grad
#
# plt.plot([i for i in range(Niter)], hist)
# plt.show()

# Create meshgrid of Voltages for V1 (focusing) and V2 (defocusing).
V1range = np.linspace(0.0, maxBias, Vsteps)
V2range = np.linspace(-maxBias, 0.0, Vsteps)
V1, V2 = np.meshgrid(V1range, V2range)

# ===============================================================================
# Test algorithm for seeking extremum using gradient of cost function. Using the
# the created meshgrid is too complicated at the moment for troubleshooting. So
# I'll start by making this in chunks.
# ===============================================================================
# Solver Chunk.
V1 = 0.4 * kV
V2 = -0.4 * kV


def solve_KV(init, s, ksolve, params=param_dict, ret_hist=False):
    """Solve KV equation for initial positions.

    Function solves KV envelope equation without acceleartion give as:
    r_x'' + kappa * r_x - 2Q/(r_x + r_y) - emit^2/r_x^3
    where the sign of kappa alternates for r_y.

    Paramters
    ---------
    init : ndarray
        Initial positions r_x, r_y, r'_x, r_y'.

    s : ndarray
        Longitudinal mesh to do the solve on.

    ksolve : ndarray
        Mesh holding the kappa values for each point in s. Note that this should
        be the same size as s.

    params : dict
        Dictionary holding parameters of the system such as perveance 'Q',
        emittance 'emittance'.

    ret_hist : bool
        Boolean switch. When set to true, the histories for the solver are
        returned that way plots can be made.

    Returns
    -------
    soln : ndarray
        Final r_x, r_y, r'_x, and r'_y.

    history: ndarray
        If ret_hist set to true then this returns a len(s) by 4 matrix holding
        r_x, r_y, r'_x, r'_y at each point on the mesh.
        """

    # Initial values and allocate arrays for solver.
    ds = s[1] - s[0]
    soln = init.copy()
    if ret_hist:
        history = np.zeros((len(s), len(init)))
        history[0, :] = soln

    Q = params["Q"]
    emittance = params["emittance"]

    for n in range(1, len(s)):
        # Grab values from soln array
        ux, uy, vx, vy = soln[0], soln[1], soln[2], soln[3]

        # Evalue terms in update equation
        term = 2 * Q / (ux + uy)
        term1x = pow(emittance, 2) / pow(ux, 3) - ksolve[n - 1] * ux
        term1y = pow(emittance, 2) / pow(uy, 3) + ksolve[n - 1] * uy

        # Evaluate updated u and v
        newvx = (term + term1x) * ds + vx
        newvy = (term + term1y) * ds + vy
        newux = newvx * ds + ux
        newuy = newvy * ds + uy

        # Update soln
        soln[:] = newux, newuy, newvx, newvy
        if ret_hist:
            history[n, :] = soln[:]

    if ret_hist:
        return soln, history
    else:
        return soln
