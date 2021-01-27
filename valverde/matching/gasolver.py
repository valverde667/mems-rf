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
from geneticalgorithm import geneticalgorithm as ga

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
s = np.linspace(0, L, N)

# Define cost function with decision variables as inputs. In this case, the
# decision variables will be 6 total: 4 positional for x, y, vx, and vy and 2
# for the ESQ voltage settings.
def cost_func(decsn_vars):
    """Cost function for the genetic algorithm

    The cost function takes as an input a 1D array of decision variables. In this
    case, the varying parameters are the 4 coordinate variables x, y, vx, and vy
    along with the two bias settings on the ESQs. The goal is to minimize the
    difference between the initial coordinate variables such that, after one
    lattice period later,
        x0 ≈ x
        y0 ≈ y
        x0' ≈ x'
        y0' ≈ y'.

    The cost is computing by calculated the sum of the root of the square of the
    differences cost = sqrt((x - x0)^2) + sqrt((y - y0)^2) + ...

    Parameters
    ----------
    decsn_vars : ndarray
        1D array giving the decision variables. The decision variables are, in
        order, (x, y, vx, vy, V1, V2) where V1 and V2 are the ESQ voltage
        settings.

    cost : float
        The computed cost of the input settings.

    """

    # Make global assignments for KV solver
    global Q, N, s, ds, emittance

    # Assignments from decsn_vars
    solver_arr = decsn_vars[0:4]
    init_arr = solver_arr.copy()
    ux0, uy0 = init_arr[0], init_arr[1]
    vx0, vy0 = init_arr[2], init_arr[3]
    V1, V2 = decsn_vars[4], decsn_vars[5]

    # Create focusing strength array with hard edge model
    karray, _ = hard_edge_kappa([V1, V2], s)

    # Main loop. Loop through updated array and solver KV equation
    for n in range(0, N):
        ux, uy = solver_arr[0], solver_arr[1]
        vx, vy = solver_arr[2], solver_arr[3]
        term = 2 * Q / (ux + uy)

        term1x = pow(emittance, 2) / pow(ux, 3) - karray[n] * ux
        term1y = pow(emittance, 2) / pow(uy, 3) + karray[n] * uy

        vx = (term + term1x) * ds + vx
        vy = (term + term1y) * ds + vy

        ux = vx * ds + ux
        uy = vy * ds + uy

        solver_arr = np.array([ux, uy, vx, vy])

    # Compute cost
    costx = np.sqrt((ux - ux0) ** 2) + np.sqrt((vx - vx0) ** 2)
    costy = np.sqrt((uy - uy0) ** 2) + np.sqrt((vy - vy0) ** 2)

    # Create a constrain by defining a penalty. The constraint is that the
    # total radius cannot exceed the aperture radius of .55mm
    rend = np.sqrt(ux ** 2 + uy ** 2)
    if rend > 0.55 * mm:
        pen = 500.0
    else:
        pen = 0.0

    cost = costx + costy + pen

    return cost


# Creat variable boundaries for the GA solver. From the docs, the boundaries
# must be an array for each variable. Thus, for 6 variables, I need a 6x# array.
ux_bound = np.array([0.2 * mm, 0.4 * mm])
uy_bound = np.array([0.2 * mm, 0.4 * mm])
vx_bound = np.array([3 * mm, 5 * mm])
vy_bound = np.array([-5 * mm, -3 * mm])
V1_bound = np.array([-0.7 * kV, 0.7 * kV])
V2_bound = np.array([-0.7 * kV, 0.7 * kV])
var_bound = np.array([ux_bound, uy_bound, vx_bound, vy_bound, V1_bound, V2_bound])

# Set GA parameters and initialize model.
algorithm_param = {
    "max_num_iteration": 2000,
    "population_size": 250,
    "mutation_probability": 0.45,
    "elit_ratio": 0.01,
    "crossover_probability": 0.5,
    "parents_portion": 0.3,
    "crossover_type": "uniform",
    "max_iteration_without_improv": None,
}
model = ga(
    function=cost_func,
    dimension=6,
    variable_type="real",
    variable_boundaries=var_bound,
    algorithm_parameters=algorithm_param,
)
model.run()
