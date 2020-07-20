import streamlit as st
import MEQALAC.simulation as meqsim
from matplotlib import pyplot as plt

st.title("STS 50 0d simulations of energy distribution")

st.sidebar.markdown("## Design parameter")

initial_energy = st.sidebar.number_input(
    "design beam energy", 0.1, 10e3, 7e3, step=100.0
)
design_voltage = st.sidebar.number_input(
    "design voltage/gap", 0.1, 15e3, 7e3, step=100.0
)
design_frequency = (
    st.sidebar.number_input("design frequency", 10.0, 20.0, 14.86, step=0.01) * 1e6
)

st.sidebar.markdown("## Beam parameter")


beam_length = st.sidebar.number_input("beam length [us]", 1, 20, 5, step=1) * 1e-6

real_energy = st.sidebar.number_input("real beam energy", 0.1, 10e3, 7e3, step=100.0)
real_voltage = st.sidebar.number_input("real voltage/gap", 0.1, 15e3, 7e3, step=100.0)
real_frequency = (
    st.sidebar.number_input("real frequency", 10.0, 20.0, 14.86, step=0.01) * 1e6
)

steps = st.sidebar.number_input("time steps in gaps", 1, 100, 1, step=1)
gaps = st.sidebar.number_input("number of  gaps", 1, 2, 1, step=1)

beam = meqsim.beam_setup(initial_energy, 10000, beam_length)
pos = meqsim.wafer_setup(
    E=initial_energy, V=design_voltage, f=design_frequency, N=gaps
)

t = meqsim.trace_particles(
    pos=pos,
    f=real_frequency,
    V=real_voltage,
    StartingCondition=beam,
    d=2e-3,
    steps=steps,
)

mask = t[:, 1] >= 0
out = t[mask, :]

plt.hist(out[:, 1], 100)
st.pyplot()
