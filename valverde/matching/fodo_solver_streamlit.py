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
st.sidebar.markdown("## Design Paramter")

# Create slide bars for paramter variation
Q = st.sidebar.number_input(
    "Q perveance", 1e-6, 1e-4, param_dict["Q"], step=4.95e-6, format="%.4e"
)
emittance = st.sidebar.number_input(
    "Emittance e", 1e-6, 1e-4, param_dict["emittance"], step=4.95e-6, format="%.4e"
)
V1 = st.sidebar.number_input(
    "Voltage on First ESQ V1", 0.0, 5.5 * kV, 3 * kV, step=0.5 * kV, format="%.2e"
)
V2 = st.sidebar.number_input(
    "Voltage on Second ESQ V2", -5.5 * kV, 0.0, -3 * kV, step=0.5 * kV, format="%.2e"
)
ux = st.sidebar.number_input(
    "x Injection Position", 0.2 * mm, 0.5 * mm, 0.5 * mm, step=0.05 * mm, format="%e"
)
uy = st.sidebar.number_input(
    "y Injection Position", 0.2 * mm, 0.5 * mm, 0.5 * mm, step=0.05 * mm, format="%e"
)

vx = st.sidebar.number_input(
    "x-angle Injection", 2 * mrad, 5 * mrad, 5 * mrad, step=0.5 * mrad, format="%e"
)
vx = st.sidebar.number_input(
    "x-angle Injection", -5 * mrad, -2 * mrad, -5 * mrad, step=0.5 * mrad, format="%e"
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

shift = 0  # Shift "injection" to left in units of ds
start = d - shift * ds

# Find index for shifted start and reorient arrays
start_index = np.where(s >= start)[0][0]
s_solve = s.copy()[start_index:]
k_solve = karray.copy()[start_index:]
V_solve = Varray.copy()[start_index:]

# Visualize shift. Show ESQ and Gap positions. Grey out portion of lattice not used.
fig, ax = plt.subplots()
ax.set_ylim(-6, 6)
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
