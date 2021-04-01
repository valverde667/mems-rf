"""Script to perform 1D gradient descent on ESQ optimization in the lattice.
Here, only the r_x r'_x coordinates will be varied given the bias settings for
ESQ1 and ESQ2."""

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
N = 2000
s = np.linspace(0, Lp, N + 1)
ds = s[1] - s[0]
param_dict = params.main()
inj_energy = param_dict["inj_energy"]
# Max settings
maxBias = Vbrkdwn * space
maxR = 0.55 * mm
maxDR = maxR / Lp

# Hyperparameter settings
Niter = 1000
lrng_rate = 0.0001
Vsteps = 10
threshold = 0.01  # Cost function most likely wont approach 0 exactly


# ==============================================================================
#     Utility Function
# Here the functions necessary for the script are defined. These include the
# the solver function, the cost function for minimizing, and the gradient of
# the cost function.
# ==============================================================================


def OneD_solve_KV(init, s, ksolve, params=param_dict, ret_hist=False):
    """Solve KV equation for initial positions.

    Function solves KV envelope equation without acceleartion give as:
    r_x'' + kappa * r_x - 2Q/(r_x + r_y) - emit^2/r_x^3.

    Paramters
    ---------
    init : ndarray
        Initial positions r_x and r'_x

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
        Final r_x and r'_x

    history: ndarray
        If ret_hist set to true then this returns a len(s) by 2 matrix holding
        r_x and r'_x at each point on the mesh.
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
        ux, uy = soln[:, 0], soln[:, 1]
        vx, vy = soln[:, 2], soln[:, 3]

        # Evalue terms in update equation
        term = 2 * Q / (ux + uy)

        term1x = pow(emittance, 2) / pow(ux, 3) - ksolve[:, n - 1] * ux
        term1y = pow(emittance, 2) / pow(uy, 3) + ksolve[:, n - 1] * uy

        # Evaluate updated u and v
        newvx = (term + term1x) * ds + vx
        newvy = (term + term1y) * ds + vy

        newux = newvx * ds + ux
        newuy = newvy * ds + uy

        # Update soln
        soln[:, 0], soln[:, 1] = newux, newuy
        soln[:, 2], soln[:, 3] = newvx, newvy
        if ret_hist:
            history[n, :] = soln[:]

    if ret_hist:
        return soln, history
    else:
        return soln


def cost_func(init_params, fin_params, weights):
    """Cost function for minimizing.

    Here, the cost function is the squared differences between the final
    paramters and initial paramters after solving.
        C = w1 * (r_0 - rf)^2 + w2 * (r'_0 - r'_f)^2
    where w1 and w2 are the weights to be used and the cost function is summed
    over each component r_x.

    Paramters
    ---------
    init_params : ndarray
        Array holding initial values r_x and r'_x

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

    # Evaluate cost function in chunks to minimize error
    cost = weights * (init_params - fin_params) ** 2

    return np.sum(cost, axis=1)


def OneD_gd_cost(init_params, fin_params, weights):
    """Cost function for minimizing.

    Here, the cost function is the squared differences between the final
    paramters and initial paramters after solving.
        C = 2w1 * (r_0 - rf) + 2w2 * (r'_0 - r'_f)
    where w1 and w2 are the weights to be used and the cost function is summed
    over each component r_x and r_y.

    Paramters
    ---------
    init_params : ndarray
        Array holding initial values r_x and r'_x

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

    # Evaluate cost function in chunks to minimize error
    gd_cost = 2 * weights * (init_params - fin_params)
    return gd_cost


# ==============================================================================
#     Optimization for single voltage setting.
# Here the optimization routine is tested for a single voltage setting. This
# portion acts as a testing arena for solvability and tuning of hyperparamters.
# ==============================================================================

do_one_setting = False
if do_one_setting:
    # Initialize hard edge kappa array
    voltages = [0.4 * kV, -0.4 * kV]
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
    init_rx = 0.5 * mm
    init_rpx = 5 * mrad
    init = np.array([init_rx, init_rpx])

    position_weight, angle_weight = 1, 1
    weights = np.array([position_weight, angle_weight])
    lrn_rate = 10 ** -3
    epochs = 2000
    params = init.copy()
    param_traj = np.zeros([epochs + 1, len(params)])

    for i in range(epochs):
        sol = OneD_solve_KV(init, s, hard_kappa)
        dW = OneD_gd_cost(init, sol, weights=weights)
        params = params - lrn_rate * dW
        param_traj[i, :] = params


# ==============================================================================
#     Optimization for grid of voltage settings
# Create a meshgrid of voltage settings and then perform gradient descent on each
# setting all at once. This will require running the solver and gradient loop
# MxM times where M is the number of voltage settings. Care should be taken in
# this part to do it right.
# ==============================================================================
V1 = np.linspace(-0.4, 0.4, 100) * kV
V2 = np.linspace(-0.4, 0.4, 100) * kV
Vgrid = np.array(np.meshgrid(V1, V2))
Vsets = Vgrid.T.reshape(-1, 2)
# Identify interval for +/- ESQ according to position. The first chunk
# represents the drift region between two stacks and is not field free. The
# second chunk is the field free region and has the ESQs.
chnk1 = d
esq1_lowbnd = chnk1 + g / 2 + space
esq1_highbnd = esq1_lowbnd + lq

esq2_lowbnd = esq1_highbnd + space
esq2_highbnd = esq2_lowbnd + lq

# Find indices that correspond to the ESQs
splus = np.where((s >= esq1_lowbnd) & (s <= esq1_highbnd))[0]
sminus = np.where((s >= esq2_lowbnd) & (s <= esq2_highbnd))[0]

kappa_array = np.zeros([Vsets.shape[0], len(s)])
voltage_array = kappa_array.copy()

V1_array = Vsets[:, 0]
V2_array = Vsets[:, 1]

voltage_array[:, splus] = V1_array[:, np.newaxis]
voltage_array[:, sminus] = V2_array[:, np.newaxis]

kappa_array = voltage_array / inj_energy / maxR / maxR
# Create initial position and angles. Use preset x=.5mm and x' = 5 mrad for all
# values and optimize all at once.
init = np.ones(shape=(kappa_array.shape[0], 4))
init[:, :] = 0.5 * mm, 0.5 * mm, 5 * mrad, -5 * mrad

# Perform gradient descent using the above routine.
position_weight, angle_weight = maxR, maxDR
weights = np.array([position_weight, position_weight, angle_weight, angle_weight])
lrn_rate = 10 ** 0
epochs = 5
params = init.copy()
for i in range(epochs):
    sol = OneD_solve_KV(params, s, kappa_array)
    dW = OneD_gd_cost(params, sol, weights=weights)
    params = params - lrn_rate * dW

lrn_rate = 10 ** -1
for i in range(epochs):
    sol = OneD_solve_KV(params, s, kappa_array)
    dW = OneD_gd_cost(params, sol, weights=weights)
    params = params - lrn_rate * dW

lrn_rate = 10 ** -2
for i in range(int(epochs * 3)):
    sol = OneD_solve_KV(params, s, kappa_array)
    dW = OneD_gd_cost(params, sol, weights=weights)
    params = params - lrn_rate * dW

costs = cost_func(init, params, weights)
costs = costs / max(costs)
cost_grid = costs.reshape(len(V1), len(V2)).T
np.save("cost_array", costs)
np.save("param_array", params)
np.save("voltage_settings", Vsets)
# Create contour plot
fig, ax = plt.subplots()
cont = ax.contourf(Vgrid[1] / kV, Vgrid[0] / kV, cost_grid, levels=50)
ax.set_xlabel(r"$V_2$ [kV]")
ax.set_ylabel(r"$V_1$ [kV]")
fig.colorbar(cont)
plt.savefig("costcontour.pdf", dpi=400)

plt.show()

# ==============================================================================
#     Optimization on Beales
# A simple gradient descent on Beales function.
# ==============================================================================
def beales_function(x, y):
    return (
        np.square(1.5 - x + x * y)
        + np.square(2.25 - x + x * y * y)
        + np.square(2.625 - x + x * y ** 3)
    )
    return f


def grad_beales_function(params):
    x = params[0]
    y = params[1]
    grad_x = (
        2 * (1.5 - x + x * y) * (-1 + y)
        + 2 * (2.25 - x + x * y ** 2) * (-1 + y ** 2)
        + 2 * (2.625 - x + x * y ** 3) * (-1 + y ** 3)
    )
    grad_y = (
        2 * (1.5 - x + x * y) * x
        + 4 * (2.25 - x + x * y ** 2) * x * y
        + 6 * (2.625 - x + x * y ** 3) * x * y ** 2
    )
    return [grad_x, grad_y]


def gd(grad, init, n_epochs=1000, eta=10 ** -4, noise_strength=0):
    # This is a simple optimizer
    params = np.array(init)
    param_traj = np.zeros([n_epochs + 1, 2])
    param_traj[0,] = init
    dW = 0
    for j in range(n_epochs):
        noise = noise_strength * np.random.randn(params.size)
        dW = np.array(grad(params)) + noise
        params = params - eta * dW
        param_traj[j + 1,] = params
    return param_traj


do_beales = False
if do_beales:
    init = [1, -3]
    eta = 10 ** -3
    epochs = 5000
    dw = 0
    params = np.array(init)
    param_traj = np.zeros([epochs + 1, 2])
    param_traj[0,] = init
    for i in range(epochs):
        dW = np.array(grad_beales_function(params))
        params = params - eta * dW
        param_traj[i + 1,] = params
