"""Script to investigate matching section before accelerating lattice. The
2018 paper found suitable solutions from six voltage settings and a FDDFFD
setup. The six voltages were in Volts: 551, -510, -512, 359, 352, -136.
Executing <streamlit run match_section_streamlit.py> in the terminal will open
a browser window where the voltages can be set and the solution to the KV
envelope equation will be plotted in this matching region."""

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
rad = np.pi

# Grab parameter dictionary
param_dict = parameters.main()

# Set up streamlit
st.title("Parameter Variation for Matching Section")
st.sidebar.markdown("## Design Parameter")

# Create slide bars for paramter variation
Q = st.sidebar.number_input(
    "Q perveance ",
    min_value=1e-6,
    max_value=1e-3,
    value=param_dict["Q"],
    step=4.95e-6,
    format="%.4e",
)
emittance = st.sidebar.number_input(
    "Emittance e [m-rad]",
    min_value=1e-6,
    max_value=1e-3,
    value=param_dict["emittance"],
    step=4.95e-6,
    format="%.4e",
)
V1 = st.sidebar.number_input(
    "Voltage on First ESQ V1 [V]",
    min_value=-0.6 * kV,
    max_value=0.6 * kV,
    value=0.552 * kV,
    step=0.1 * kV,
    format="%.2e",
)
V2 = st.sidebar.number_input(
    "Voltage on Second ESQ V1 [V]",
    min_value=-0.6 * kV,
    max_value=0.6 * kV,
    value=-0.5 * kV,
    step=0.1 * kV,
    format="%.2e",
)
V3 = st.sidebar.number_input(
    "Voltage on Third ESQ V1 [V]",
    min_value=-0.6 * kV,
    max_value=0.6 * kV,
    value=-0.512 * kV,
    step=0.1 * kV,
    format="%.2e",
)
V4 = st.sidebar.number_input(
    "Voltage on Fourth ESQ V1 [V]",
    min_value=-0.6 * kV,
    max_value=0.6 * kV,
    value=0.359 * kV,
    step=0.1 * kV,
    format="%.2e",
)
V5 = st.sidebar.number_input(
    "Voltage on Fifth ESQ V1 [V]",
    min_value=-0.6 * kV,
    max_value=0.6 * kV,
    value=0.352 * kV,
    step=0.1 * kV,
    format="%.2e",
)
V6 = st.sidebar.number_input(
    "Voltage on Sixth ESQ V1 [V]",
    min_value=-0.6 * kV,
    max_value=0.6 * kV,
    value=-0.136 * kV,
    step=0.1 * kV,
    format="%.2e",
)
ux_initial = st.sidebar.number_input(
    "x Injection Position [m]",
    min_value=0.2 * mm,
    max_value=0.5 * mm,
    value=param_dict["inj_radius"],
    step=0.05 * mm,
    format="%.3e",
)
uy_initial = st.sidebar.number_input(
    "y Injection Position [m]",
    min_value=0.2 * mm,
    max_value=0.5 * mm,
    value=param_dict["inj_radius"],
    step=0.05 * mm,
    format="%.3e",
)

vx_initial = st.sidebar.number_input(
    "x-angle Injection [rad]",
    min_value=0.01,
    max_value=0.05,
    value=param_dict["inj_xprime"],
    step=0.5 * mrad,
    format="%.3e",
)
vy_initial = st.sidebar.number_input(
    "y-angle Injection [rad]",
    min_value=-0.05,
    max_value=-0.01,
    value=param_dict["inj_yprime"],
    step=0.5 * mrad,
    format="%.3e",
)
