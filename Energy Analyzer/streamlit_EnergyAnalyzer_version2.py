import streamlit as st
import time, glob, json
import numpy as np
import scipy.constants as c
import plotly.figure_factory as ff
import plotly.express as px
import pandas as pd


def readjson(fp):
    with open(fp, "r") as readfile:
        data = json.load(readfile)
    return data


def energy(v):
    vv = np.array(v)
    return list(1 / 2 * 40 * c.atomic_mass * vv * vv / c.elementary_charge)


####
st.title("Title")
# binwidth=st.slider(25,1000)

data = []
energies = []
voltages = []
voltagerange = range(500, 6500, 500)
df = pd.DataFrame()

# for vol in voltagerange:
#     for e in energy(readjson(f"{vol}.json")["uz"]):
#         energies.append(e)
#         voltages.append(vol)
#
# df=pd.DataFrame({"energy":energies,"voltage":voltages})

for vol in voltagerange:
    energies.append(energy(readjson(f"{vol}.json")["uz"]))
fig1 = ff.create_distplot(energies, voltagerange, bin_size=1000)

st.plotly_chart(fig1)
