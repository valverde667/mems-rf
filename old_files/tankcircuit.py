"""
    Plotting HV Grid control data vs. 1D simulation
    poster data (Argon, 8kV, 2018-03-08) to test maximum acceleration with LC tank circuit
    Created by Grace Woods sometime around 4/1/2018
"""
import MEQALAC
import numpy as np
import matplotlib.pyplot as plt

amu = 1/6.022141e26 # convert kg to amu

""" 2018-03-08 -- poster data """

posterdata = MEQALAC.data.shots_from_scan("2018-03-08", 121547 , 123640)
posterdata = sorted(posterdata, key=lambda x: x.sn)

postercurrent = []
postergrid = []
for shot in posterdata:
    postercurrent.append(shot.CH1.x[233:400].mean()/10e3*1e6) # in uA
    postergrid.append(shot.get_value_from_setting("HV Grid control")/1e3) #in kV
# getting rid of negative values - I think this only works in python3:
postercurrent = np.array(postercurrent)
postercurrent[postercurrent<=0] = 0

# if you are using python2:
#postercurrent = list(0 if i<=0 else i for i in postercurrent)

# normalizing current for plotting
norm_postercurrent = postercurrent/postercurrent[0]

# 1D simulation

freq = posterdata[0].get_value_from_setting("RF frequency") # fetch frequency used in experiment
V = 500 # this is typically a parameter for fitting to data
packages = 1000 # number of particles
pulse_length = 1e-3 # in seconds
E = 8e3 # injection Energy
Ar_mass = 40*amu # we used argon in this experiment

StartingCondition = MEQALAC.simulation.beam_setup(E,packages,pulse_length,mass=Ar_mass,q=1)
pos = MEQALAC.simulation.wafer_setup(E,V,freq, N=2,Fcup_dist=5e-2,mass=Ar_mass,q=1) # positions corresponding to wafer stack that I built - in mm

# ***PLOTTING***

fig,ax= plt.subplots(1,1)
ax2 = None
simcurrent = norm_postercurrent[0:9].mean()

MEQALAC.simulation.create_plot(pos,StartingCondition,ax,ax2,V,freq,zerr=0,current=simcurrent,runs=1,mass=Ar_mass,q=1,savefile='PosterData.txt',label='Simulation')

plt.plot(postergrid,norm_postercurrent,"ro")

plt.xlabel('Retarding Potential (kV)',fontsize = 16)
plt.ylabel('Fraction Transmitted',fontsize = 16)
plt.legend()
plt.show(block=False)
