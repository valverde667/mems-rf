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

# Parameters for lattice (See fodo streamlit script)
d = 9.3 * mm  # Distance between RF gap centers
g = 2 * mm  # RF gap length
lq = 0.695 * mm  # ESQ length
space = (d - g - 2 * lq) / 3  # Spacing between ESQs
Lp = 2 * d  # Lattice period
Vbrkdwn = 3 * kV / mm
N = 10
s = np.linspace(0, Lp, N + 1)
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

# Weights used in cost function and gradient. Note, that the weights are
# squared here as dictated by the function definition below.
position_weight = pow(1 / maxR, 2)
position_angle = pow(s[-1] / maxR, 2)

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

# ==============================================================================
#     Utility Function
# Here the functions necessary for the script are defined. These include the
# the solver function, the cost function for minimizing, and the gradient of
# the cost function.
# ==============================================================================


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


def cost_func(init_param, fin_params, weights):
    """Cost function for minimizing.

    Here, the cost function is the squared differences between the final
    paramters and initial paramters after solving.
        C = w1 * (rf - r0)^2 + w2 * (r'f - r'0)^2
    where w1 and w2 are the weights to be used and the cost function is summed
    over each component r_x and r_y.

    Paramters
    ---------
    init_params : ndarray
        Array holding initial values r_x, r_y, r'_x, r'_y

    fin_params : ndarray
        Array holding final values from solver.

    weights : ndarray
        Array holding the weights to be used for the position and angle
        contributions to the cost.

    Returns
    -------
    cost : float
        The value of the cost function.
    """

    # Initialize/grab values
    init_pos = init_params[0:2]
    fin_pos = fin_params[0:2]
    init_angle = init_params[-2:]
    fin_angle = fin_params[-2:]

    w_pos = weights[0]
    w_angle = weights[1]

    # Evaluate cost function in chunks to minimize error
    cost_pos = w_pos * (fin_pos - init_pos) ** 2
    cost_angle = w_angle * (fin_angle - init_angle) ** 2
    cost = cost_pos + cost_angle

    return cost


def gd_cost(init_parms, fin_parms, weights):
    """Cost function for minimizing.

    Here, the cost function is the squared differences between the final
    paramters and initial paramters after solving.
        C = w1 * (rf - r0)^2 + w2 * (r'f - r'0)^2
    where w1 and w2 are the weights to be used and the cost function is summed
    over each component r_x and r_y.

    Paramters
    ---------
    init_params : ndarray
        Array holding initial values r_x, r_y, r'_x, r'_y

    fin_params : ndarray
        Array holding final values from solver.

    weights : ndarray
        Array holding the weights to be used for the position and angle
        contributions to the cost.

    Returns
    -------
    gd_cost : float
        The value of the gradient of the cost function.
    """

    # Initialize/grab values
    init_pos = init_params[0:2]
    fin_pos = fin_params[0:2]
    init_angle = init_params[-2:]
    fin_angle = fin_params[-2:]

    w_pos = weights[0]
    w_angle = weights[1]

    # Evaluate cost function in chunks to minimize error
    gd_cost_pos = 2 * w_pos * (fin_pos - init_pos)
    gd_cost_angle = 2 * w_angle * (fin_angle - init_angle)
    gd_cost = cost_pos + cost_angle

    return gd_cost


# ==============================================================================
#     Optimization for single voltage setting.
# Here the optimization routine is tested for a single voltage setting. This
# portion acts as a testing arena for solvability and tuning of hyperparamters.
# ==============================================================================

# Initialize hard edge kappa array
voltages = [0 * kV, 0 * kV]
inj_energy = param_dict["inj_energy"]
hard_kappa, __ = hard_edge_kappa(
    voltage=voltages,
    pos_array=s,
    drift=d,
    gap=g,
    length_quad=lq,
    N=len(s),
    injection_energy=inj_energy,
    rp=maxR,
)

# Initial position and angle parameters
init_rx, init_ry = 0.5 * mm, 0.5 * mm
init_rpx, init_rpy = 5 * mrad, -5 * mrad
init = np.array([init_rx, init_ry, init_rpx, init_rpy])
sol, h = solve_KV(init, s, hard_kappa, ret_hist=True)
