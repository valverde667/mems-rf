"""
    Plotting HV Grid control data vs. 1D simulation
    2) matching section data (Argon, 11kV, 2017-01-28)
    Created by Grace Woods sometime around 4/1/2018
"""

import MEQALAC
import numpy as np
import matplotlib.pyplot as plt

amu = 1/6.022141e26 # convert kg to amu
Ar_mass = 40*amu # we used argon in this experiment


""" 2017-01-28 -- matching section data """

#matchingsection = MEQALAC.data.shots_from_scan("2017-01-24", 181133 , 182820)
matchingsection = MEQALAC.data.shots_from_scan("2017-01-24", 163703 , 170749)

matchingsection = sorted(matchingsection, key=lambda x: x.sn)
matchingcurrent = []
matchinggrid = []

for shot in matchingsection:
    matchingcurrent.append(shot.CH4.x[80:100].mean()/1e3*1e6) # in uA
    matchinggrid.append(shot.get_value_from_setting("HV Grid control")/1e3) # in kV

# getting rid of negative values - I think this only works in python3
matchingcurrent=np.array(matchingcurrent)
matchingcurrent[matchingcurrent<=0] = 0


# normalizing current for plotting
norm_matchingcurrent = matchingcurrent/matchingcurrent[0]

# 1D simulation

freq = matchingsection[0].get_value_from_setting("RF frequency")
V = 750 # this is typically a parameter for fitting to data
packages = 1000 # number of particles
pulse_length = 1e-3 # in seconds
E = 11e3 # injection energy
Ar_mass = 40*amu # ion species

StartingCondition = MEQALAC.simulation.beam_setup(E,packages,pulse_length,mass=Ar_mass,q=1)
pos = MEQALAC.simulation.wafer_setup(E,V,freq,N=4,Fcup_dist=5e-2,mass=Ar_mass,q=1) # N indicates number of acceleration gaps

# ***PLOTTING***
fig,ax= plt.subplots(1,1)
ax2 = None

MEQALAC.simulation.create_plot(pos,StartingCondition,ax,ax2,V,freq,zerr=0,current=1,runs=1,mass=Ar_mass,q=1,savefile='MatchingSection.txt',label='Simulation')
plt.plot(matchinggrid,norm_matchingcurrent,'ro')

plt.xlabel('Retarding Potential (kV)', fontsize = 16)
plt.ylabel('Fraction Transmitted', fontsize = 16)
plt.legend()
plt.show(block=False)
