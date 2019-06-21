"""
    Compare aspect ratio of Warp simulation (changing radius of aperature) from accelerator.py
    Created by Grace Woods on 4/2/2018
    Loads files from Warp simulation (quartwavelength.py)
    Output: 1) energy distribution vs grid voltage of 1D and warp
            2) maximum acceleration vs aspect ratio

    longitudinal gap = drift = 1.5 - 2*thickness of electrode
    aspect ratio = longitudinal gap / aperature = drift/diameter\
    originally = (1.5e-3 - 2*.005) / 2 (e.g. diameter of acceleration was 2mm)
"""
import matplotlib.pyplot as plt
import numpy as np

"""With dielectric"""

# r = 1*mm
r1= 1
ar1 = (1.5-2*.005)/(2*r1)

Her1 = np.loadtxt('H_E_er')
H2er1 = np.loadtxt('H2_E_er')

# r = .75*mm
r2 = .75
ar2 = (1.5-2*.005)/(2*r2)
Her75 = np.loadtxt('H_E_75')
H2er75 = np.loadtxt('H2_E_75')


# r = .5*mm
r3 = .5
ar3 = (1.5-2*.005)/(2*r3)
Her50 = np.loadtxt('H_E_50')
H2er50 = np.loadtxt('H2_E_50')

# r = .25
r4 = .25
ar4 = (1.5-2*.005)/(2*r4)
Her25 = np.loadtxt('H_E_25')
H2er25 = np.loadtxt('H2_E_25')

# r = .15
r5 = .15
ar5 = (1.5-2*.005)/(2*r5)
Her15 = np.loadtxt('H_E_15')
H2er15 = np.loadtxt('H2_E_15')


"""normalizing"""
Grid = np.arange(9200,10800,10)


numH_1=[]
numH2_1=[]
numH_75=[]
numH2_75=[]
numH_50=[]
numH2_50=[]
numH_25=[]
numH2_25=[]
numH_15=[]
numH2_15=[]

total1 = np.size(Her1)
total2 = np.size(H2er1)
total3 = np.size(Her75)
total4 = np.size(H2er75)
total5 = np.size(Her50)
total6 = np.size(H2er50)
total7 = np.size(Her25)
total8 = np.size(H2er25)
total9 = np.size(Her15)
total10 = np.size(H2er15)

for G in Grid:
    numH_1.append(sum(G<E for E in Her1)/total1)
    numH2_1.append(sum(G<E for E in H2er1)/total2)
    numH_75.append(sum(G<E for E in Her75)/total3)
    numH2_75.append(sum(G<E for E in H2er75)/total4)
    numH_50.append(sum(G<E for E in Her50)/total5)
    numH2_50.append(sum(G<E for E in H2er50)/total6)
    numH_25.append(sum(G<E for E in Her25)/total7)
    numH2_25.append(sum(G<E for E in H2er25)/total8)
    numH_15.append(sum(G<E for E in Her15)/total9)
    numH2_15.append(sum(G<E for E in H2er15)/total10)


"""1D simulation"""
# in order to generate data : run MEQALAC/atap-meqalac-code/python/MEQALAC simulation.py
# in create_plot function, Y/Y.max() = simY (normalized data) and V = simV (grid voltages)

H1dE = np.loadtxt('H1DsimY')
H1dV = np.loadtxt('H1DsimV')

H21dE = np.loadtxt('H21DsimY')
H21dV = np.loadtxt('H21DsimV')

""" plotting number of passing ions versus grid voltage for Warp + 1D simulations """
#fig, ax = plt.subplot(1,1)

plt.plot(Grid,numH_1, marker = 'o', label = 'H ar = {:.3f}'.format(ar1))
plt.plot(Grid,numH2_1, marker = 'o', label ='H2 ar = {:.3f}'.format(ar1))

#plt.plot(Grid,numH_75, marker = 'o', label = 'H ar = {:.2f}'.format(ar2))
#plt.plot(Grid,numH2_75, marker = 'o', label = 'H2 ar = {:.2f}'.format(ar2))

plt.plot(Grid,numH_50, marker = 'o', label = 'H ar = {:.3f}'.format(ar3))
plt.plot(Grid,numH2_50, marker = 'o', label = 'H2 ar = {:.3f}'.format(ar3))

plt.plot(Grid,numH_25, marker = 'o', label = 'H ar = {:.3f}'.format(ar4))
plt.plot(Grid,numH2_25, marker = 'o', label = 'H2 ar = {:.3f}'.format(ar4))

#plt.plot(Grid,numH_15, marker = 'o', label = 'H ar = {:.2f}'.format(ar5))
#plt.plot(Grid,numH2_15, marker = 'o', label = 'H2 ar = {:.2f}'.format(ar5))

plt.plot(H1dV,H1dE,marker='o',label='1D H')
plt.plot(H21dV,H21dE,marker='o',label='1D H2')


plt.xlabel('Retarding Voltage (V)')
plt.ylabel('Number of particles (normalized)')
plt.legend()
plt.show()


""" new plot to generate max acceleration versus aspect ratio """
# max acceleration occurs in first instance of num particles ~ 0
# data generated only for ar1 ar3 and ar4, but can be anything
# feel free to simplify in loops - this was just a brute force method

ar = np.array([ar1,ar3,ar4])

maxH_1 = Grid[min(q for q,num in enumerate(numH_1) if num<0.01)]
maxH_50 = Grid[min(q for q,num in enumerate(numH_50) if num<0.01)]
maxH_25 = Grid[min(q for q,num in enumerate(numH_25) if num<0.01)]
maxH = np.array([maxH_1,maxH_50,maxH_25])/1e3

maxH2_1 = Grid[min(q for q,num in enumerate(numH2_1) if num<0.01)]
maxH2_50 = Grid[min(q for q,num in enumerate(numH2_50) if num<0.01)]
maxH2_25 = Grid[min(q for q,num in enumerate(numH2_25) if num<0.01)]
maxH2 = np.array([maxH2_1,maxH2_50,maxH2_25])/1e3

plt.plot(ar,maxH,color='red',marker='o',label='H+')
plt.plot(ar,maxH2,color='blue',marker='X',label='H2+')
plt.legend()
plt.xlabel('Aspect Ratio (ar)')
plt.ylabel('Maximum Acceleration (keV)')
plt.show()
