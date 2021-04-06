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

# ==============================================================================
#     Utility Functions
# Here the functions necessary for the script are defined. These include the
# the solver function, the cost function for minimizing, and the gradient of
# the cost function.
# ==============================================================================
def OneD_solve_KV(init, s, ksolve, Q=5.7e-5, emittance=1.1e-5, ret_hist=False):
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


def gradient_descent(
    params,
    s,
    kappa_array,
    weights=[1, 1, 1, 1],
    lrn_rate=1.0,
    epochs=1,
    Q=5.72e-5,
    emittance=1.12e-5,
    verbose=False,
):
    """Function to perform gradient descent and update parameters."""

    # Run descent for number of epochs and update parameters
    if verbose:
        print("--Epochs: {}".format(epochs))
        print("--Learning Rate: {}".format(lrn_rate))

    for iter in range(epochs):
        if iter % 50 == 0:
            if verbose:
                print("--Epoch {}".format(iter, epochs))
        kv_soln = OneD_solve_KV(params, s, kappa_array, Q=Q, emittance=emittance)
        dW = OneD_gd_cost(params, kv_soln, weights)
        params = params - lrn_rate * dW

    return params


# ==============================================================================
#     Simplified one volt setting
# This section sets up the routine for one voltage setting. This is primarily
# used for testing the current setups to see if solutions are found. The
# voltage and input settings are from hand-tuned solutions using the streamlit
# script.
# ==============================================================================
# Parameters from the cell
Q = 0
emittance = 0

V1 = 0.210 * kV
V2 = -0.230 * kV
chnk1 = d
esq1_lowbnd = chnk1 + g / 2 + space
esq1_highbnd = esq1_lowbnd + lq

esq2_lowbnd = esq1_highbnd + space
esq2_highbnd = esq2_lowbnd + lq

# Find indices that correspond to the ESQs
splus = np.where((s >= esq1_lowbnd) & (s <= esq1_highbnd))[0]
sminus = np.where((s >= esq2_lowbnd) & (s <= esq2_highbnd))[0]

kappa_array = np.zeros(len(s))
voltage_array = kappa_array.copy()
voltage_array[splus] = V1
voltage_array[sminus] = V2
kappa_array = voltage_array / inj_energy / maxR / maxR

# Create initial position and angles. Use preset x=.5mm and x' = 5 mrad =
init = np.ones(4)
init[:] = 0.5 * mm, 0.5 * mm, 5 * mrad, -5 * mrad

# Arrays must be in the shape (MxN) in order to work with the solver functions.
# This is remedied by adding a new row-axis. The row axis represents the differnt
# settings where here there is one.
kappa_array = kappa_array[np.newaxis, :]
voltage_array = voltage_array[np.newaxis, :]
init = init[np.newaxis, :]

# Create weights and use scale factor to scale the position weight so that the
# gradients are of same magnitude.
scale_fact = 200
position_weight = scale_fact * maxDR
angle_weight = maxDR
weights = np.array([position_weight, position_weight, angle_weight, angle_weight])
params = init.copy()

# Set hyperparameters for gradient descent. Learning rate affects the magnitude
# of the gradient. Epochs sets how man steps to take
lrn_rate = 10 ** -2
epochs = 100
cost_tol = 1e-6
param_hist = []
param_hist.append(params[0, :])
cost_hist = []
rhist = []

for iter in range(epochs):
    if iter % 50 == 0:
        print("--Epoch {}".format(iter))
    sol = gradient_descent(
        params,
        s,
        kappa_array,
        weights=weights,
        lrn_rate=lrn_rate,
        epochs=1,
        Q=0,
        emittance=0,
    )
    this_cost = cost_func(params, sol, weights)[0]
    excursion = np.sum(np.sqrt(sol[0, :2] ** 2))
    cost_hist.append(this_cost)
    rhist.append(excursion)
    param_hist.append(sol[0, :])

    params = sol

    cond = (this_cost < cost_tol) & (excursion < maxR)
    if cond:
        print("Solution found")
        break
param_hist = np.array(param_hist)
cost_hist = np.array(cost_hist)
rhist = np.array(rhist)
ddd
# ==============================================================================
#     Create grid of voltage settings
# The voltages will be creating by using the meshgrid routine in numpy. From
# here, for each voltage setting there is a corresponding kappa array that
# must be created. This will give a matrix where each row represents an array
# of kappa values corresponding to the voltage settings.
# ==============================================================================
V1 = np.linspace(0.0, 0.350, 20) * kV
V2 = np.linspace(-0.350, 0.0, 20) * kV
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

# ==============================================================================
#     Gradient descent section
# Here, the initial arrays are generated for the routine. Each array will
# simultaneously optimized. This is done by collecting all initializations into
# a matrix and then performing the solver.
## Todo
#    - Put constraints in GD updates to reduce computational cost.
#    - Perform detailed analysis on hyperparameters (lrn_rate, gradients, etc.)
# ==============================================================================
# Create initial position and angles. Use preset x=.5mm and x' = 5 mrad for all
# values and optimize all at once.
init = np.ones(shape=(kappa_array.shape[0], 4))
# init = np.random.random((kappa_array.shape[0], 4))
init[:, :] = 0.5 * mm, 0.5 * mm, 5 * mrad, -5 * mrad

# Perform gradient descent using the above routine.
scale_fact = 200
position_weight = scale_fact * maxDR
angle_weight = maxDR
weights = np.array([position_weight, position_weight, angle_weight, angle_weight])
params = init.copy()

lrn_rate = 10 ** -1
epochs = 2
params = gradient_descent(
    params, s, kappa_array, weights=weights, lrn_rate=lrn_rate, epochs=epochs
)
lrn_rate = 10 ** -2
epochs = 2
params = gradient_descent(
    params, s, kappa_array, weights=weights, lrn_rate=lrn_rate, epochs=epochs
)

lrn_rate = 10 ** -3
epochs = 2
params = gradient_descent(
    params, s, kappa_array, weights=weights, lrn_rate=lrn_rate, epochs=epochs
)

costs = cost_func(init, params, weights)
costs = costs / max(costs)

# Find constraints and set normed cost equal to 1
mask_rx = params[:, 0] >= maxR
mask_ry = params[:, 1] >= maxR
mask_r = mask_rx | mask_ry
costs[mask_r] = 1

mask_rxp = (params[:, 2] >= maxDR) | (params[:, 2] <= -maxDR)
mask_ryp = (params[:, 3] >= maxDR) | (params[:, 3] <= -maxDR)
mask_rp = mask_rxp | mask_ryp
costs[mask_rp] = 1

cost_grid = costs.reshape(len(V1), len(V2)).T
np.save("cost_array", costs)
np.save("param_array", params)
np.save("voltage_settings", Vsets)
# Create contour plot
fig, ax = plt.subplots()
cont = ax.contourf(Vgrid[1] / kV, Vgrid[0] / kV, cost_grid, levels=30)
ax.set_xlabel(r"$V_2$ [kV]")
ax.set_ylabel(r"$V_1$ [kV]")
fig.colorbar(cont)
plt.savefig("costcontour.pdf", dpi=400)

plt.show()
