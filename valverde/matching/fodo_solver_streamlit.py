"""Script to investigate parameter settings on solving the KV equation. The
current MEMs lattice cell is asymmetrical and it is not obvious what causes the
difficulty in getting suitable focusing. The purpose of this script is to
incrementally tweak the FODO cell and solve the KV equations until the MEMS
MEMS cell is created."""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import pdb

import warp as wp

import parameters
from solver import hard_edge_kappa, solve_KV

# Define useful constants
mm = wp.mm
kV = wp.kV
mrad = 1e-3

# Grab parameter dictionary
param_dict = parameters.main()

# Set up streamlit
st.title("Parameter variation of the FODO Cell")
st.sidebar.markdown("## Design Parameter")

# Create slide bars for paramter variation
Q = st.sidebar.number_input(
    "Q perveance ", 1e-6, 1e-4, param_dict["Q"], step=4.95e-6, format="%.4e"
)
emittance = st.sidebar.number_input(
    "Emittance e [m-rad]",
    1e-6,
    1e-4,
    param_dict["emittance"],
    step=4.95e-6,
    format="%.4e",
)
V1 = st.sidebar.number_input(
    "Voltage on First ESQ V1 [V]", 0.0, 0.6 * kV, 0.1 * kV, step=0.1 * kV, format="%.2e"
)
V2 = st.sidebar.number_input(
    "Voltage on Second ESQ V2 [V]",
    -0.6 * kV,
    0.0,
    -0.1 * kV,
    step=0.1 * kV,
    format="%.2e",
)
ux_initial = st.sidebar.number_input(
    "x Injection Position [m]",
    0.2 * mm,
    0.5 * mm,
    0.5 * mm,
    step=0.05 * mm,
    format="%e",
)
uy_initial = st.sidebar.number_input(
    "y Injection Position [m]",
    0.2 * mm,
    0.5 * mm,
    0.5 * mm,
    step=0.05 * mm,
    format="%e",
)

vx_initial = st.sidebar.number_input(
    "x-angle Injection [rad]",
    2 * mrad,
    5 * mrad,
    5 * mrad,
    step=0.5 * mrad,
    format="%e",
)
vy_initial = st.sidebar.number_input(
    "y-angle Injection [rad]",
    -5 * mrad,
    -2 * mrad,
    -5 * mrad,
    step=0.5 * mrad,
    format="%e",
)

# I will now set up the simulation mesh. The whole mesh will be created. But,
# in order to see the effects of the beginning drift, the solver will start
# from the second gap initially. This gives the symmetric cell to start. Then,
# the solver can be tweaked to start further back essentially adding, in
# increments of ds, a long pre-drift.
d = 9.3 * mm
g = 2 * mm
lq = 0.695 * mm
N = 500
L = 2 * d
space = (d - g - 2 * lq) / 3  # Spacing between ESQs
ds = L / N
s = np.linspace(0, L, N)

# Create arrays for KV equation
karray, Varray = hard_edge_kappa([V1, V2], s)

# Add sidebar input to adjust amoun to pre-drift
maxshift = np.where(s <= d)[0][-1]
shift = st.sidebar.number_input(
    "Ammount of Drift to Add in units of ds", 0, maxshift, 0, format="%d"
)
start = d - shift * ds

# Find index for shifted start and reorient arrays
start_index = np.where(s >= start)[0][0]
s_solve = s.copy()[start_index:]
ksolve = karray.copy()[start_index:]
Vsolve = Varray.copy()[start_index:]
# Visualize shift. Show ESQ and Gap positions. Grey out portion of lattice not used.
fig, ax = plt.subplots()
ax.set_ylim(-0.6, 0.6)
ax.plot(s / mm, Varray / kV)
ax.fill_between(s[Varray > 0] / mm, max(Varray) / kV, y2=0, alpha=0.2, color="b")
ax.fill_between(s[Varray < 0] / mm, min(Varray) / kV, y2=0, alpha=0.2, color="r")
ax.fill_between(
    s[s <= s_solve[0]] / mm,
    ax.get_ylim()[0],
    ax.get_ylim()[1],
    alpha=0.4,
    color="lightgray",
)
plates = np.array([g / 2, d - g / 2, d + g / 2, 2 * d - g / 2])
for pos in plates:
    ax.axvline(x=pos / mm, c="k", ls="--", lw=2)
ax.set_xlabel("s [mm]")
ax.set_ylabel(r"$V$ [kV]")
ax.set_title("Schematic of Simulation Geometry")
st.pyplot(fig)

# Solve KV equation with reoritented arrays
soln_matrix = np.zeros(shape=(len(s_solve), 4))
soln_matrix[0, :] = ux_initial, uy_initial, vx_initial, vy_initial

# Grab position and angle arrays from matrix
ux = soln_matrix[:, 0]
uy = soln_matrix[:, 1]
vx = soln_matrix[:, 2]
vy = soln_matrix[:, 3]

# Main loop to update equation. Loop through matrix and update entries.
for n in range(1, len(soln_matrix)):
    # Evaluate term present in both equations
    term = 2 * Q / (ux[n - 1] + uy[n - 1])

    # Evaluate terms for x and y
    term1x = pow(emittance, 2) / pow(ux[n - 1], 3) - ksolve[n - 1] * ux[n - 1]
    term1y = pow(emittance, 2) / pow(uy[n - 1], 3) + ksolve[n - 1] * uy[n - 1]

    # Update v_x and v_y first.
    vx[n] = (term + term1x) * ds + vx[n - 1]
    vy[n] = (term + term1y) * ds + vy[n - 1]

    # Use updated v to update u
    ux[n] = vx[n] * ds + ux[n - 1]
    uy[n] = vy[n] * ds + uy[n - 1]


# Create plots for solution to KV equtions and overlay ESQ and gap positions
fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(s_solve / mm, ux / mm, c="k")
maxy0, miny0 = ax[0].get_ylim()[0], ax[0].get_ylim()[1]
ax[0].set_ylabel(r"$r_x(s)$ [mm]")
ax[0].fill_between(s_solve[Vsolve > 0] / mm, maxy0, y2=miny0, alpha=0.2, color="b")
ax[0].fill_between(s_solve[Vsolve < 0] / mm, maxy0, y2=miny0, alpha=0.2, color="r")
for pos in plates:
    ax[0].axvline(x=pos / mm, c="k", ls="--", lw=2)


ax[1].plot(s_solve / mm, vx, c="k")
maxy1, miny1 = ax[1].get_ylim()[0], ax[1].get_ylim()[1]
ax[1].set_ylabel(r"$r_x'(s)$")
ax[1].set_xlabel(r"$s$ [mm]")
ax[1].fill_between(s_solve[Vsolve > 0] / mm, maxy1, y2=miny1, alpha=0.2, color="b")
ax[1].fill_between(s_solve[Vsolve < 0] / mm, maxy1, y2=miny1, alpha=0.2, color="r")
for pos in plates:
    ax[1].axvline(x=pos / mm, c="k", ls="--", lw=2)
st.pyplot(fig)

fig, ax = plt.subplots(nrows=2, sharex=True)
ax[0].plot(s_solve / mm, uy / mm, c="k")
maxy0, miny0 = ax[0].get_ylim()[0], ax[0].get_ylim()[1]
ax[0].set_ylabel(r"$r_y(s)$")
ax[0].fill_between(s_solve[Vsolve > 0] / mm, maxy0, y2=miny0, alpha=0.2, color="b")
ax[0].fill_between(s_solve[Vsolve < 0] / mm, maxy0, y2=miny0, alpha=0.2, color="r")
for pos in plates:
    ax[0].axvline(x=pos / mm, c="k", ls="--", lw=2)

ax[1].plot(s_solve / mm, vy, c="k")
maxy1, miny1 = ax[1].get_ylim()[0], ax[1].get_ylim()[1]
ax[1].set_xlabel(r"$s$ [mm]")
ax[1].fill_between(s_solve[Vsolve > 0] / mm, maxy1, y2=miny1, alpha=0.2, color="b")
ax[1].fill_between(s_solve[Vsolve < 0] / mm, maxy1, y2=miny1, alpha=0.2, color="r")
for pos in plates:
    ax[1].axvline(x=pos / mm, c="k", ls="--", lw=2)

ax[1].set_ylabel(r"$r_y'(s)$")
st.pyplot(fig)
