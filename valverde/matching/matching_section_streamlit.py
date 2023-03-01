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
from solver import hard_edge_kappa, solve_KV, matching_section

# Define useful constants
mm = wp.mm
kV = wp.kV
mrad = 1e-3
rad = np.pi

# Grab parameter dictionary
param_dict = parameters.param_dict

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
    min_value=0.0,
    max_value=0.05,
    value=param_dict["inj_xprime"],
    step=0.5 * mrad,
    format="%.3e",
)
vy_initial = st.sidebar.number_input(
    "y-angle Injection [rad]",
    min_value=-0.05,
    max_value=0.0,
    value=param_dict["inj_yprime"],
    step=0.5 * mrad,
    format="%.3e",
)

# Set up simulation mesh. The geometry is set from the matching section class
# itself. Thus, if there are changes to be made to the ESQ lengths, drift spaces,
# etc. then these must be set for the instantiated classes attributes.
voltage_list = [V1, V2, V3, V4, V5, V6]
nEsq = len(voltage_list)
s_solve, ksolve, Varray = matching_section(N_esq=nEsq).create_section(
    voltages=voltage_list
)
ds = s_solve[1] - s_solve[0]
# Solve KV equations
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
