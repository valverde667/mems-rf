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
    "Voltage on First ESQ V1", 0.0, 5.5 * kV, step=0.5 * kV, format="%.2e"
)
V2 = st.sidebar.number_input(
    "Voltage on Second ESQ V2", -5.5 * kV, 0.0, step=0.5 * kV, format="%.2e"
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
ds = L / N
s = np.linspace(0, L, N)
