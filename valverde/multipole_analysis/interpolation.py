"""This is the basis script for building the interpolation routine.
TODO:
- Make test case
- Check getgrid from Warp and assess applicability"""

import numpy as np
import matplotlib.pyplot as plt
import pdb

import warp as wp


# ------------------------------------------------------------------------------
# This section will build the test case.
# The test case will make and x-y grid of unit spacing, i.e. dx=dy=1, and will
# span from -50 to 50 in both directions. This will make it easier to verify
# the outputs are correct for the area-weighting interpolation.
# ------------------------------------------------------------------------------
x = np.linspace(-10, 10, 31)
y = np.linspace(-10, 10, 31)
dx, dy = x[1] - x[0], y[1] - y[0]

# Parametrize points based off a fixed radius and angle
R = 9.5
theta = np.linspace(0, 2 * np.pi, 30)
xm_array = R * np.cos(theta)
ym_array = R * np.sin(theta)

# Create xy-pairs from the meshgrid function in numpy. Create a function q(x,y)
# that holds a value at each grid point. This will be the function that is being
# interpolated for.
mesh = np.array(np.meshgrid(x, y))
xy_pairs = mesh.T.reshape(-1, 2)


def q(x, y):
    """Arbitrary quadratic function that will be interpolated."""
    return np.exp(-(x ** 2)) + np.sinh(y)


q_grid = q(mesh[0], mesh[1])

# Visualize the setup
show_schematic = True
if show_schematic:
    fig, ax = plt.subplots()
    for pair in xy_pairs:
        ax.scatter(pair[0], pair[1], c="k", s=5)
    ax.scatter(xm_array, ym_array, c="r", s=3)
    ax.axhline(y=0, lw=1, c="k")
    ax.axvline(x=0, lw=1, c="k")
    ax.plot(xm_array, ym_array, ls="--", c="r", lw=0.5, alpha=0.4)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    annotate = False
    if annotate:
        for i in range(len(x)):
            for j in range(len(y)):
                text = f"{q_grid[i,j]:.2f}"
                plt.annotate(text, (x[i] + 0.1, y[j] + 0.1), fontsize="xx-small")

    ax.set_aspect("equal", adjustable="box")
    plt.savefig("/Users/nickvalverde/Desktop/interpolate_setup.pdf", dpi=400)
    plt.show()


# ------------------------------------------------------------------------------
# This section will do the area weighting interpolation.
# To do this, the nearest grid points must be found for each xm,ym.
# For the moment, this is done using the first quadrant (all possitive grid
# values). Full geometry will to be treated later on
# ------------------------------------------------------------------------------
qm_array = np.zeros(len(xm_array))
dA = dx * dy
diff_tol = 1e-8
make_result_plot = False
if make_result_plot:
    fig, ax = plt.subplots()
    ax.set_xlim(-0.1, x.max())
    ax.set_ylim(-0.1, y.max())
    indices = []

for i, (xm, ym) in enumerate(zip(xm_array, ym_array)):
    random_fudge = np.random.random() * 1e-12
    xm += random_fudge
    ym += random_fudge

    # Find grid points that enclose the xm-ym coordinates in use
    lowx = xm - dx
    highx = xm + dx

    lowy = ym - dy
    highy = ym + dy

    mask_x = (x >= lowx) & (x <= highx)
    mask_y = (y >= lowy) & (x <= highy)

    left, right = x[mask_x][0], x[mask_x][-1]
    bottom, top = y[mask_y][0], y[mask_y][-1]

    # Record indices for the gridpoints in use.
    x_indices = np.where(mask_x)[0]
    ix_left = x_indices[0]
    ix_right = x_indices[-1]

    y_indices = np.where(mask_y)[0]
    iy_bottom = y_indices[0]
    iy_top = y_indices[-1]

    # Calculate Areas and weight grid data
    A1 = (xm - left) * (ym - bottom)
    A2 = (right - xm) * (ym - bottom)
    A3 = (xm - left) * (top - ym)
    A4 = (right - xm) * (top - ym)

    q1m = q_grid[ix_left, iy_bottom] * A4 / dA
    q2m = q_grid[ix_right, iy_bottom] * A3 / dA
    q3m = q_grid[ix_left, iy_top] * A2 / dA
    q4m = q_grid[ix_right, iy_top] * A1 / dA

    qm = q1m + q2m + q3m + q4m
    qm_array[i] = qm
    if make_result_plot:
        these_inds = np.array([ix_left, ix_right, iy_bottom, iy_top])
        indices.append(these_inds)

if make_result_plot:
    for this_ind in indices:
        ixl, ixr, iyb, iyt = this_ind
        ax.scatter(x[ixl], y[iyb], c="k", s=10)
        ax.scatter(x[ixl], y[iyt], c="k", s=10)
        ax.scatter(x[ixr], y[iyb], c="k", s=10)
        ax.scatter(x[ixr], y[iyt], c="k", s=10)
        text1 = f"{q_grid[ixl, iyb]:.2f}"
        text2 = f"{q_grid[ixr, iyt]:.2f}"
        ax.annotate(text1, (x[ixl] - 0.1, y[iyb] - 0.2), fontsize="xx-small")
        ax.annotate(text2, (x[ixr] + 0.1, y[iyt] + 0.1), fontsize="xx-small")

    ax.scatter(xm_array, ym_array, c="r", s=5)
    for i, qm in enumerate(qm_array):
        # pt1 = [xm_array[i], xm_array[i]]
        # pt2 = [ym_array[i], max(ax.get_ylim()) - i*(1 + .05)]
        # ax.plot(pt1, pt2, lw=.5, c='r')
        ax.annotate(f"{qm:.2f}", (xm_array[i], ym_array[i]), fontsize="xx-small")
    ax.set_aspect("equal", adjustable="box")
    plt.savefig("/Users/nickvalverde/Desktop/interpolated_results.pdf", dpi=400)

    plt.show()


# Use warp machinery to get interpolated points. Note, Fortran begins counting
# at 1 and so, to match with Python, the array lengths need to be deducted 1.
z_interpol = np.zeros(len(xm_array))
wp.top.getgrid2d(
    len(xm_array),
    xm_array,
    ym_array,
    z_interpol,
    len(x) - 1,
    len(y) - 1,
    q_grid,
    x.min(),
    x.max(),
    y.min(),
    y.max(),
)


#
