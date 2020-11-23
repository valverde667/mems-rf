"""Script for solivng KV-envelope equation."""

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as const

import parameters


def hard_edge_kappa(kappa, pos_array):
    """Function to populate kappa(s) with kappa assuming hard edge model.

    The function locates ESQs in an array and sets values to +/- kappa. This
    is to approximate an ESQ doublet by hard edge. Thus, the entries in the
    kappa_array will be in those positions where the ESQ lives on according to
    the position array.

    Parameters
    ----------
    kappa : float
        Value of kappa that is to be input in the array.

    pos_array : ndarray
        One dimensional array giving the discretized space values that the KV
        enevlope equation is to be solved one.

    Returns
    -------
    kappa_array : ndarray
        The array representing two ESQs via a hard edge model. This array will
        be of same size as the position array.

    """

    # Grab global values
    global d, lq, g, k

    # Identify interval for +/- ESQ according to position
    plus_int = d / 2
    minus_int = plus_int + lq + d + g + d

    # Find indices that correspond to the ESQs
    splus = np.where((pos_array >= plus_int) & (pos_array <= plus_int + lq))[0]
    sminus = np.where((pos_array >= minus_int) & (pos_array <= minus_int + lq))[0]

    # Create kappa array and assign values
    kappa_array = np.zeros(len(pos_array))

    # Assign plus and minus ESQ values
    kappa_array[splus] = k
    kappa_array[sminus] = -k

    return kappa_array


def solve_KV():
    """Differential equation solver for KV equation

    Function will use Euler-Cromer method to solve the KV-envelope equation.
    Denoting derivative by primes, the coupled equation is

    r_x'' + k(s)r_x - 2Q/(r_x + r_y) - emittance_x^2/r_x^3 = 0
    r_y'' + k(s)r_y - 2Q/(r_x + r_y) - emittance_y^2/r_y^3 = 0

    The method transforms the two second-order equations into four first-order
    equations by defining u_x = r_x, v_x = u'_x and similarly for r_y. The
    input parameter is assumed to be an Nx4 matrix where the columns reprsent,
    in order: u_x, u_y, v_x, v_y. These arrays are then updated and the soln
    soln matrix is returned. The initial conditions should be initialized in the
    matrix calling the function. Else, the ICs will be zero.

    Parameters
    ----------
    soln_matrix : ndarray
        An Nx4 matrix where N is the number of steps to take in the solver
        and the four columns are u_x, u_y, v_x, v_y respectively.

    Returns
    -------
    kv_soln : ndarray
        The Nx4 matrix where the columns are now the solutions to the KV-envelope
        equation.

    """

    # Use global variables
    global Q, karray, ds, emittance, soln_marix

    # Identify arrays
    ux = soln_matrix[:, 0]
    uy = soln_matrix[:, 1]
    vx = soln_matrix[:, 2]
    vy = soln_matrix[:, 3]

    # Main loop to update equation. Loop through matrix and update entries.
    for n in range(1, len(soln_matrix)):
        # Evaluate term present in both equations
        term = 2 * Q / (ux[n - 1] + uy[n - 1])

        # Evaluate terms for x and y
        term1x = pow(emittance, 2) / pow(ux[n - 1], 3) - karray[n - 1] * ux[n - 1]
        term1y = pow(emittance, 2) / pow(uy[n - 1], 3) + karray[n - 1] * uy[n - 1]

        # Update v_x and v_y first.
        vx[n] = (term + term1x) * ds + vx[n - 1]
        vy[n] = (term + term1y) * ds + uy[n - 1]

        # Use updated v to update u
        ux[n] = vx[n] * ds + ux[n - 1]
        uy[n] = vy[n] * ds + uy[n - 1]

    return True


# Read in parameters frome paramters script.
param_dict = parameters.main()

# Set system parameters/variables
Q = param_dict["Q"]
k = 5.7e4
emittance = param_dict["emittance"]
ux_initial = param_dict["inj_radius"]
uy_initial = param_dict["inj_radius"]
vx_initial = 5 * const.milli
vy_initial = -5 * const.milli

# Set up solver paramters
d = (2 / 3) * const.milli
g = 695 * const.micro
lq = 695 * const.micro
N = 501
L = d / 2 + lq + d + g + d + lq + d / 2
ds = L / N

# Construct arrays for differential equation. Columns ux, uy, vx, vy
soln_matrix = np.zeros(shape=(N, 4))
s = np.array([ds * i for i in range(N)])

# Enter initial condition into array
soln_matrix[0, :] = ux_initial, uy_initial, vx_initial, vy_initial

# Create kappa array
karray = hard_edge_kappa(k, s)
# Call solver
solve_KV()

# --Plotting outputs
# Grab x,x',y,and y'
x = soln_matrix[:, 0]
y = soln_matrix[:, 1]
xprime = soln_matrix[:, 2]
yprime = soln_matrix[:, 3]

# Create plots
fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(s / const.milli, x / const.milli, c="k")
ax[0].set_ylabel(r"$r_x(s)$ [mm]")

ax[1].plot(s / const.milli, xprime, c="k")
ax[1].set_xlabel(r"$s$ [mm]")
ax[1].set_ylabel(r"$r_x'(s)$ [mm]")
plt.show()

fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(s / const.milli, y / const.milli, c="k")
ax[0].set_ylabel(r"$r_y'(s)$")

ax[1].plot(s / const.milli, yprime, c="k")
ax[1].set_xlabel(r"$s$ [mm]")
ax[1].set_ylabel(r"$r_y'(s)$")
plt.show()
