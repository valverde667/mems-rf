import warpoptions

# 2020 Carlos latest commit 4-10
# 2020-03-23, Timo
# This file is based of maddy's version but has a lot of code
# re-written, espacially does it make use of the new code for
# the geometry of the RF-Wafers. The only current problem
# is that the wafer positions are not perfect yet.
# This can be solved by some try and error


#   only specify the parameters you want to change from their default values
#   the input will look like:
"""
python3 single-species-simulation.py --esq_voltage=500 --fraction=.8 --speciesMass=20 --ekininit=15e3
"""

#   bunch length
warpoptions.parser.add_argument('--bunch_length',
                                dest='Lbunch', type=float,
                                default='1e-9')

#   number of RF units
warpoptions.parser.add_argument('--units', dest='Units',
                                type=int,
                                default='2')  # number of RF gaps

#   voltage on the RF gaps at the peak of the sinusoid
warpoptions.parser.add_argument('--rf_voltage', dest='Vmax',
                                type=float,
                                default='10000')  # will be between 4000 and 10000

# voltage on the Vesq
warpoptions.parser.add_argument('--esq_voltage',
                                dest='Vesq', type=float,
                                default='.01')  # 850

#   fraction of the max voltage at which the selected ions cross
warpoptions.parser.add_argument('--fraction',
                                dest='V_arrival',
                                type=float,
                                default='1')  # .8

#   mass of the ions being accelerated
warpoptions.parser.add_argument('--species_mass',
                                dest='speciesMass',
                                type=int, default='40')

#   injection energy in eV
warpoptions.parser.add_argument('--ekininit',
                                dest='ekininit', type=float,
                                default='10e3')

#   frequency of the RF
warpoptions.parser.add_argument('--freq', dest='freq',
                                type=float,
                                default='13.56e6')  # 27e6

#   emitting radius
warpoptions.parser.add_argument('--emit',
                                dest='emittingRadius',
                                type=float,
                                default='.25e-3')  # .37e-3 #.25

#   divergence angle
warpoptions.parser.add_argument('--diva',
                                dest='divergenceAngle',
                                type=float, default='5e-3')

#   special cgm name - for mass output / scripting
warpoptions.parser.add_argument('--name', dest='name',
                                type=str,
                                default='unnamedRun')

#   special cgm path - for mass output / scripting
warpoptions.parser.add_argument('--path', dest='path',
                                type=str,
                                default='')

#   divergence angle
warpoptions.parser.add_argument('--tstep',
                                dest='timestep',
                                type=float, default='1e-9')  # 1e-11

#   enables some additional code if True
warpoptions.parser.add_argument('--autorun',
                                dest='autorun',
                                type=bool, default=False)

import warp as wp
import numpy as np
import geometry
from geometry import RF_stack
from helper import gitversion
import matplotlib.pyplot as plt
import datetime
import time
import math
import os
from pathlib import Path
import json
import sys
from warp.particles.extpart import ZCrossingParticles
from warp.particles import particlescraper

start = time.time()

wp.w3d.solvergeom = wp.w3d.XYZgeom
# Timesteps
wp.top.dt = warpoptions.options.timestep
# wp.top.dt = 1e-10# updated from original script, check first before pushing 4/15
# 10-9 for short; 10e-11 for a nice one

# --- keep track of when the particles are born
wp.top.ssnpid = wp.nextpid()
wp.top.tbirthpid = wp.nextpid()

# set input parameters
L_bunch = warpoptions.options.Lbunch
Units = warpoptions.options.Units
Vmax = warpoptions.options.Vmax  # RF voltage
Vesq = warpoptions.options.Vesq
V_arrival = warpoptions.options.V_arrival  # fraction of the total voltage gained across each gap
speciesMass = warpoptions.options.speciesMass * wp.amu
selectedIons = wp.Species(charge_state=1, name='Ar',
                          mass=warpoptions.options.speciesMass * wp.amu,
                          color=wp.green)
ekininit = warpoptions.options.ekininit
freq = warpoptions.options.freq  # RF freq
emittingRadius = warpoptions.options.emittingRadius
divergenceAngle = warpoptions.options.divergenceAngle
name = warpoptions.options.name
# Invoke setup routine for the plotting (name the cgm output file)
# add a date & timestamp to the cgm file
now = datetime.datetime.now()
datetimestamp = datetime.datetime.now().strftime(
    '%m-%d-%y_%H:%M:%S')

# --- where to store the outputfiles
cgm_name = name
step1path = "/home/timo/Documents/LBL/Warp/CGM"
# step1path = '/home/cverdoza/Documents/LBL/WARP/berkeleylab-atap-meqalac-simulations/RF-Wafers-Simulations/test'

# overwrite if path is given by command
if warpoptions.options.path != '':
    step1path = warpoptions.options.path

wp.setup(prefix=f"{step1path}/{cgm_name}")  # , cgmlog= 0)

# --- Set basic beam parameters

ibeaminit = 10e-6  # initial beam current May vary up to 25e-6
n = 0  # to be used to switch the voltages on the RF wafers

wp.top.a0 = emittingRadius
wp.top.b0 = emittingRadius
wp.top.ap0 = divergenceAngle
wp.top.bp0 = divergenceAngle
wp.top.vbeam = .0e0
wp.top.emit = 9.45E-7
wp.top.ibeam = ibeaminit
wp.top.ekin = ekininit
wp.top.zion = selectedIons.charge_state
wp.top.lrelativ = False
wp.top.linj_efromgrid = True
wp.derivqty()

# --- Set input parameters describing the 3d simulation

wp.w3d.l4symtry = True  # True
wp.w3d.l2symtry = False

# ---   Set boundary conditions

# ---   for field solve
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.neumann

# ---   for particles
wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb
wp.top.prwall = 1 * wp.mm  # prwall slightly bigger than aperture radius so ions can get absorbed by conductors

# --- Set field grid size, this is the width of the window
wp.w3d.xmmin = 3 / 2 * wp.mm
wp.w3d.xmmax = 3 / 2 * wp.mm
wp.w3d.ymmin = 3 / 2 * wp.mm  #
wp.w3d.ymmax = 3 / 2 * wp.mm
wp.w3d.zmmin = 0.0
# changes the length of the gist output window.
wp.w3d.zmmax = 23 * wp.mm # 10

# set grid spacing, this is the number of mesh elements in one window
wp.w3d.nx = 30  # 60.
wp.w3d.ny = 30  # 60.
wp.w3d.nz = 180. # 180 for 23 # 6-85 for 10
# ToDo what and why the following
if wp.w3d.l4symtry:
    wp.w3d.xmmin = 0.
    wp.w3d.nx /= 2
if wp.w3d.l2symtry or wp.w3d.l4symtry:
    wp.w3d.ymmin = 0.
    wp.w3d.ny /= 2

# --- Select plot intervals, etc.
# ToDo what is this?
wp.top.npmax = 300
wp.top.inject = 1  # 2 means space-charge limited injection
wp.top.rinject = 5000
wp.top.npinject = 10  # 300  # needed!! macro particles per time step or cell#modified smaller step from 30 on 4/15 to monitor particles correctly. refer to original script
wp.top.linj_eperp = False  # Turn on transverse E-fields near emitting surface
wp.top.zinject = wp.w3d.zmmin
wp.top.vinject = 1.0
print("--- Ions start at: ", wp.top.zinject)

wp.top.nhist = 5  # Save history data every N time step
wp.top.itmomnts[0:4] = [0, 1000000, wp.top.nhist,
                        0]  # Calculate moments every N steps
# --- Save time histories of various quantities versus z.
wp.top.lhpnumz = True
wp.top.lhcurrz = True
wp.top.lhrrmsz = True
wp.top.lhxrmsz = True
wp.top.lhyrmsz = True
wp.top.lhepsnxz = True
wp.top.lhepsnyz = True
wp.top.lhvzrmsz = True

# --- Set up fieldsolver - 7 means the multigrid solver
solver = wp.MRBlock3D()
wp.registersolver(solver)
solver.mgtol = 1.0  # Poisson solver tolerance, in volts
solver.mgparam = 1.5
solver.downpasses = 2
solver.uppasses = 2

# --- Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.)
wp.package("w3d")
wp.generate()

ESQs = []
RFs = []
ID_ESQ = 100
ID_RF = 201
ID_target = 1


# --- generates voltage for the RFs
def gen_volt(toffset=0):  # 0
    """ A cos voltage function with variable offset"""

    def RFvoltage(time):
        return -Vmax * np.cos(
            2 * np.pi * freq * (time + toffset))

    return RFvoltage


# --- generates voltage for the ESQs
def gen_volt_esq(Vesq, inverse=False, toffset=0):
    def ESQvoltage(time):
        if inverse:
            return -Vesq  # *np.sin(2*np.pi*freq*(time+toffset))
        else:
            return Vesq  # *np.sin(2*np.pi*freq*(time+toffset))

    return ESQvoltage


# --- calculate the distances and time offset for the RFs
centerOfFirstRFGap = 5 * wp.mm
tParticlesAtCenterFirstGap = \
    centerOfFirstRFGap / wp.sqrt(2 * ekininit *
                                 selectedIons.charge / speciesMass)
print(f"Particles need {tParticlesAtCenterFirstGap * 1e6}us"
      f" to reach the center of the first gap")
# RF should be maximal when particles arrive
# time for the cos to travel there minus the time the
# particles take # Todo check this -> seems to work
RF_offset = centerOfFirstRFGap / wp.clight - \
            tParticlesAtCenterFirstGap
print(f"RF_offset {RF_offset}")
ESQ_toffset = 0

Vpos = []


# calculating the ideal positions
def calculateRFwaferpositions():
    positionArray = []
    # Calculating first position
    # this is not actually C but the very first wafer a
    c = centerOfFirstRFGap - geometry.gapGNDRF / 2 - \
        geometry.copper_thickness - \
        geometry.wafer_thickness / 2
    betalambda0 = 0
    betalambda1 = 0
    for i in np.arange(0, Units):
        # a, b, c & d are the positions of the center of RFs/GND wafers
        # GND RF RF GND
        a = c + betalambda0
        b = a + geometry.gapGNDRF + \
            geometry.copper_thickness * 2 + \
            geometry.wafer_thickness
        betalambda1 = wp.sqrt(
            (ekininit + V_arrival * Vmax * (2 * i + 1)) * 2 * selectedIons.charge / speciesMass) * 1 / freq / 2
        c = a + betalambda1
        d = b + betalambda1
        betalambda0 = wp.sqrt(
            (ekininit + V_arrival * Vmax * (2 * i + 2))
            * 2 * selectedIons.charge /
            speciesMass) * 1 / freq / 2

        if Units == 1:
            c = c - 10 * wp.mm
        elif Units == 2:
            a = a - 50 * wp.mm
            b = b + 40 * wp.mm

        positionArray.append([a, b, c, d])
    return positionArray


# Here it is optional to overwrite the position Array, to
# simulate the ACTUAL setup:
calculatedPositionArray = calculateRFwaferpositions()
# print(calculatedPositionArray)
positionArray = [[.0036525, .0056525, 0.01323279, 0.01523279],
                 # [0.0233854,0.0253854,0.03420207,0.03620207],
                 # [0.0485042,0.0505042,0.06300143,0.06500143] #timo testrun
                 ]
# for 9kV
# positionArray = [[.0036525,.0056525,0.01243279,0.01463279],
#                  [0.0211854,0.0233854,0.03130207,0.03330207],
#                  [0.04460842,0.0460842,0.06009143,0.062009143]
#                        ]
# for 7kV


# catching it at the plates with peak voltage #april 15

### Functions for automated wafer position by batch running
basepath = warpoptions.options.path
thisrunID = warpoptions.options.name
markedpositions = []
markedpositionsenergies = []


#
def readjson():
    fp = f'{basepath}{thisrunID}.json'
    with open(fp, 'r') as readfile:
        data = json.load(readfile)
    return data


#
def writejson(key, value):
    print(' IN SIM')
    print(thisrunID)
    print(type(thisrunID))
    writedata = readjson()
    print(writedata.keys())
    writedata[key] = value
    with open(f'{basepath}{thisrunID}.json', 'w') as writefile:
        json.dump(writedata, writefile, sort_keys=True, indent=1)


#
def autoinit():
    # assert thisrunID.__len__() == 4
    #
    rj = readjson()
    global positionArray
    waferposloaded = rj["rf_gaps"]
    positionArray = waferposloaded
    global markedpositions
    markedpositions = rj["markedpositions"]
    #
    print(f'marked positions {markedpositions}')
    writejson("rf_voltage", Vmax)
    writejson("bunch_length", L_bunch)
    writejson("ekininit", ekininit)
    writejson("freq", freq)
    writejson("tstep", warpoptions.options.timestep)
    writejson("rfgaps_ideal", calculateRFwaferpositions())
    #


#
def autosave(se):
    print(f'marked positions {markedpositions}')
    '''se : selected Ions'''
    if warpoptions.options.autorun:
        if se.getz().max() > markedpositions[0]:
            print('YES')
            ekinmax = se.getke().max()
            ekinav = se.getke().mean()
            markedpositionsenergies.append({"ekinmax": ekinmax, "ekinav": ekinav})
            writejson("markedpositionsenergies", markedpositionsenergies)
            del markedpositions[0]
            if len(markedpositions) == 0:
                print('ENDING SIM')
                return True  # cancels the entire simulation loop
    return False


#
if warpoptions.options.autorun:
    autoinit()
###
### Placing the Wafers
for i, pa in enumerate(positionArray):
    print(f"Unit {i} placed at {pa}")

# add actual stack
conductors = RF_stack(positionArray,
                      gen_volt(RF_offset))
###
# ToDo Timo
# calculate ESQ positions each plotting step is 10 timesteps
esqPositions = []
# for i in range(len(positionArray) - 1):
#    esqPositions.append(
#        (positionArray[i][-1] + positionArray[i + 1][0]) / 2)
# print(f'Placing ESQs at {esqPositions}')

# re-added after simulation on 4/14
# firstesq placed incorrectly
# consider adding manually!!!!

velo = np.sqrt(
    2 * ekininit * selectedIons.charge / selectedIons.mass)  # used to calculate tmax
length = positionArray[-1][-1] + 25 * wp.mm
tmax = length / velo  # this is used for the maximum time for timesteps
zrunmax = length  # this is used for the maximum distance for timesteps

# define the electrodes
wp.installconductors(conductors)
# print(f"Installed conductors : {wp.listofallconductors}")

# --- Recalculate the fields
wp.fieldsol(-1)

solver.gridmode = 0  # makes the fields oscillate properly at the beginning

#############################
## ToDo Timo ZC this could need some improvement -> change to saving the entire beam properly
# maybe a complete tracking of all particles might be useful for some applications
### track particles after crossing a Z location -
# in this case after the final rf amp
lastWafer = positionArray[-1][-1]
zc_pos = lastWafer + 2 * wp.mm
print(f"recording particles crossing at z = {zc_pos}")
zc = ZCrossingParticles(zz=zc_pos, laccumulate=1)
z_snapshots = [lastWafer + 0 * wp.mm,
               lastWafer + 1 * wp.mm,
               lastWafer + 2 * wp.mm,
               lastWafer + 3 * wp.mm]


def saveBeamSnapshot(z):
    if z_snapshots:  # checks for remaining snapshots
        nextZ = min(z_snapshots)
        if z > nextZ:
            # this is an approximation and in keV
            avEkin = np.square(
                selectedIons.getvz()).mean() * \
                     .5 * warpoptions.options.speciesMass * wp \
                         .amu / wp.echarge / 1000
            json_Zsnap = {
                # "ekin" : avEkin,
                "z_snap_pos": nextZ - lastWafer,
                "x": selectedIons.getx().tolist(),
                "y": selectedIons.gety().tolist(),
                "z": selectedIons.getz().tolist(),
                "vx": selectedIons.getvx().tolist(),
                "vy": selectedIons.getvy().tolist(),
                "vz": selectedIons.getvz().tolist(),
            }
            with open(f"{step1path}/{cgm_name}_snap_"
                      f"{(nextZ - lastWafer) / wp.mm:.2f}"
                      f"mm.json",
                      "w") as write_file:
                json.dump(json_Zsnap, write_file, indent=2)
            print(f"Particle snapshot created at "
                  f"{nextZ} with mean Ekin"
                  f" {avEkin}keV")
            z_snapshots.remove(nextZ)


#############################

# I want contour plots for levels between 0 and 1kV
# contours = range(0, int(Vesq), int(Vesq/10))

wp.winon(xon=0)

# Plots conductors and contours of electrostatic potential in the Z-X plane

wp.pfzx(fill=1, filled=1,
        plotphi=0)  # does not plot contours of potential
wp.fma()  # first frame in cgm file

wp.pfzx(fill=1, filled=1,
        plotphi=1)  # plots contours of potential
wp.fma()  # second frame in cgm file

zmin = wp.w3d.zmmin
zmax = wp.w3d.zmmax
zmid = 0.5 * (zmax + zmin)

# make a circle to show the beam pipe
R = .5 * wp.mm  # beam radius
t = np.linspace(0, 2 * np.pi, 100)
X = R * np.sin(t)
Y = R * np.cos(t)
deltaKE = 10e3
time_time = []
numsel = []
KE_select = []
KE_select_Max = []  # modified 4/16

Particle_Counts_Above_E = []  # modified 4/17,
# will list how much Particles have higher E than avg KE at that moment
beamwidth = []
energy_time = []
starting_particles = []
Z = [0]
scraper = wp.ParticleScraper(conductors,
                             lcollectlpdata=True)  # to use until target is fixed to output data properly

# -- name the target particles and count them
# targetz_particles = ZCrossingParticles(zz=targetz, laccumulate=1)
# this is where the actual sim runs
# TODO: wp.top.zbeam is always zero

# attempt at graphing
zEnd = 10 * wp.mm + lastWafer
print(f'Simulation runs until Z = {zEnd}')


def plotf(axes, component, new_page=True):
    if axes not in ["xy", "zx", "zy"]:
        print("error!!!! wrong axes input!!")
        return

    if component not in ["x", "y", "z", "E"]:
        print("Error! Wrong component declared!!!")
        return

    if axes == 'xy':
        plotfunc = wp.pfxy
    elif axes == 'zy':
        plotfunc = wp.pfzy
    elif axes == 'zx':
        plotfunc = wp.pfzx

    plotfunc(fill=1, filled=1, plotselfe=True, comp=component,
             titles=0, cmin=-1.2 * Vmax / geometry.gapGNDRF,
             cmax=1.2 * Vmax / geometry.gapGNDRF)  # Vmax/geometry.RF_gap

    if component == 'E':
        wp.ptitles(axes, "plot of magnitude of field")
    elif component == 'x':
        wp.ptitles(axes, "plot of E_x component of field")
    elif component == 'y':
        wp.ptitles(axes, " plot of E_y component of field")
    elif component == 'z':
        wp.ptitles(axes, "plot of E_z component of field")

    if new_page:
        wp.fma()


axes1 = 'xy'
axes2 = 'zx'
# axes3 = 'zy'
component1 = 'x'
component2 = 'z'
# component3 = 'y'
# magnitude = 'E'
# plotf(axes1,component1)
# plotf(axes2,component2)
# plotf(axes3,component3)

while (wp.top.time < tmax and max(Z) < zEnd):
    ### Running the sim
    wp.step(10)
    ### Informations
    print(f'first Particle at {max(Z)};'
          f' simulations stops at {zEnd}')

    ###### Collecting data
    ### collecting data for Particle count vs Time Plot
    time_time.append(wp.top.time)
    numsel.append(len(selectedIons.getke()))
    ### collecting for kinetic energy plots
    KE_select.append(np.mean(selectedIons.getke()))
    KE_select_Max.append(np.max(selectedIons.getke()))
    # Particle_Count = len(selectedIons.getke())  #Particle Count at this time interval
    KE = selectedIons.getke()
    # print(f"KE in this loop is = {KE}")
    Particle_Count_Over_Avg_KE = 0  # Set to zero at each moment as time elapses
    for i in range(len(selectedIons.getke())):  # goes through each particle
        Avg_KE = np.mean(selectedIons.getke())
        # print(f"Mean KE in this for loop is = {Avg_KE}")
        KE_i = selectedIons.getke()[i]
        # print(f"KE in this for loop is = {KE_i}")     # obtains KE of said particle
        if (KE_i > Avg_KE):  # checks to see if KE of particle is greater than avg KE
            Particle_Count_Over_Avg_KE += 1  # adds to count of particles above avg KE
    Particle_Counts_Above_E.append(Particle_Count_Over_Avg_KE)
    # particles in beam plot
    # accounts for all particles at that moment in time

    wp.top.pline1 = "V_RF: {:.0f}".format(
        gen_volt(RF_offset)(wp.top.time))  # Move this where it belongs
    ###### Injection
    if 0 * wp.ns < wp.top.time < L_bunch:  # changes the beam
        wp.top.finject[0, selectedIons.jslist[0]] = 1
    else:
        wp.top.inject = 0

    ###### Moving the frame dependent on particle position
    Z = selectedIons.getz()
    if Z.mean() > zmid:  # if the mean distance the particles have travelled is greater than the middle of the frame do this: MODIFIED 4/15
        # the velocity of the frame is equal to the mean velocity of the ions
        wp.top.vbeamfrm = selectedIons.getvz().mean()
        # wp.top.vbeamfrm = selectedIons.getvz().max()
        solver.gridmode = 0  # ToDo Test if this is needed
    # Todo is this needed? wp.top.zbeam is always zero
    zmin = wp.top.zbeam + wp.w3d.zmmin
    zmax = wp.top.zbeam + wp.w3d.zmmax  # trying to get rid of extra length at the end of the simulation, this is wasting computing power
    # wp.top.zbeam+wp.w3d.zmmax #scales the window length #redefines the end of the simulation tacks on the 53mm

    ###### Plotting
    ### Frame, showing
    wp.pfxy(fill=1, filled=1, plotselfe=True, comp='x',
            # added on 4/2 by Carlos
            titles=0)
    wp.ptitles(f"xy plot of E_x", f"z mean: {Z.mean()}", "x",
               "y")
    wp.fma()
    ### Frame, showing
    wp.pfxy(fill=1, filled=1, plotselfe=True, comp='z',
            titles=0)
    wp.ptitles(f"xy plot of E_z", f"z mean: {Z.mean()}", "x",
               "y")
    wp.fma()
    #
    # plotf(axes1,component1, z mean: {Z.mean()}) # Todo Carlos, can this be removed?

    ### Frame, the instantaneous kinetic energy plot
    KE = selectedIons.getke()
    print(np.mean(KE))
    if len(KE) > 0:
        selectedIons.ppzke(color=wp.blue)
        KEmin, KEmax = KE.min(), KE.max()
        while KEmax - KEmin > deltaKE:
            deltaKE += 10e3
    wp.ylimits(0.95 * KEmin, 0.95 * KEmin + deltaKE)  # is this fraction supposed to match with V_arrival?
    wp.fma()
    ### Frame the side view field plot
    # plotf(axes2,component2, z mean: {Z.mean()})
    # wp.fma()
    ### Frame, showing
    plotf('xy', 'E', 1)
    ### Frame, showing
    plotf('xy', 'z', 1)
    ### Frame, showing the side view plot of Ez and the electrical components
    wp.pfzx(fill=1, filled=1, plotselfe=True, comp='z',
            titles=0, cmin=-1.2 * Vmax / geometry.gapGNDRF,
            cmax=1.2 * Vmax / geometry.gapGNDRF)
    # sort by birthtime
    t_birth_min = selectedIons.tbirth.min()
    t_birth_max = selectedIons.tbirth.max()
    tarray = np.linspace(t_birth_min, t_birth_max, 6)
    # mask to sort particles by birthtime
    mask = []
    for i in range(len(
            tarray) - 1):  # the length of tarray must be changing overtime which changes the mask which recolors the particles
        m = (selectedIons.tbirth > tarray[i]) * (
                selectedIons.tbirth < tarray[i + 1])
        mask.append(m)
        # plot particles on top of fild plot, sort by birthtime and color them accordingly
    colors = [wp.red, wp.yellow, wp.green, wp.blue,
              wp.magenta]
    for m, c in zip(mask, colors):
        wp.plp(selectedIons.getx()[m], selectedIons.getz()[m], msize=1.0,
               color=c)  # the selected ions are changing through time
    wp.limits(zmin, zmax)
    wp.ptitles("Particles and Fields", "Z [m]", "X [m]")
    wp.fma()
    ### Frame, shwoing
    selectedIons.ppxy(color=wp.red, titles=0)  # ppxy (particle plot x horizontal axis, y on vertical axis
    wp.limits(-R, R)
    wp.ylimits(-R, R)
    wp.plg(Y, X, type="dash")
    wp.fma()
    ###
    # autosave(selectedIons)
    if autosave(selectedIons):
        print(f'Postion {selectedIons.getz().max()}')
        break
    ### check if a snapshot should be taken for export for the energy analyzer
    # saveBeamSnapshot(Z.mean())
### END of Simulation

###### Final Plots
### Frame, Particle count vs Time Plot
wp.plg(numsel, time_time, color=wp.blue)
wp.ptitles("Particle Count vs Time", "Time (s)", "Number of Particles")
wp.fma()  # fourth to last frame in cgm file
# plot lost particles with respect to Z
wp.plg(selectedIons.lostpars, wp.top.zplmesh + wp.top.zbeam)
wp.ptitles("Particles Lost vs Z", "Z", "Number of Particles Lost")
wp.fma()
### Frame, surviving particles plot:
for i in range(len(numsel)):
    p = max(numsel)
    starting_particles.append(p)
# fraction of surviving particles
f_survive = [i / j for i, j in
             zip(numsel, starting_particles)]
# want the particles that just make it through the last RF, need position of RF. This way we can see how many particles made it through the last important component of the accelerator

wp.plg(f_survive, time_time, color=wp.green)
wp.ptitles("Fraction of Surviving Particles vs Time",
           "Time (s)", "Fraction of Surviving Particles")
wp.fma()
### Frame, rms envelope plot
wp.hpxrms(color=wp.red, titles=0)
wp.hpyrms(color=wp.blue, titles=0)
wp.hprrms(color=wp.green, titles=0)
wp.ptitles("X(red), Y(blue), R(green)", "Z [m]",
           "X/Y/R [m]", "")
wp.fma()
### Frame, Kinetic Energy at certain Z value
wp.plg(KE_select, time_time, color=wp.blue)
wp.limits(0, 70e-9)  # limits(xmin,xmax,ymin,ymax)
wp.ptitles("Kinetic Energy vs Time")
wp.fma()
### Frame, maximal kinetic energy at certain Z value
wp.plg(KE_select_Max, time_time, color=wp.blue)
wp.limits(0, 70e-9)  # limits(xmin,xmax,ymin,ymax)
wp.ptitles(" Maximal Kinetic Energy vs Time")
wp.fma()
# kinetic energy plot
wp.plg(KE_select, time_time, color=wp.blue)
wp.ptitles("kinetic energy vs time")
wp.fma()
### TODO Here are duplicates, if I see that correctly @Carlos
wp.plg(KE_select_Max, time_time, color=wp.red)
wp.ptitles("Max kinetic energy vs time")
wp.fma()

wp.plg(Particle_Counts_Above_E, KE_select, color=wp.blue)
wp.ptitles("Particle Count(t) vs Energy(t)")  # modified 4/16
wp.fma()
### Frame, showing ---
KE = selectedIons.getke()
plotmin = np.min(KE) - 1
plotmax = np.max(KE) + 1
plotE = np.linspace(plotmin, plotmax, 20)
listcount = []
for e in plotE:
    elementcount = 0
    for k in KE:
        if k > e:
            elementcount += 1

    listcount.append(elementcount)
wp.plg(listcount, plotE, color=wp.red)
wp.ptitles("Number of Particles above E vs E after last gap ")
C, edges = np.histogram(KE, bins=len(plotE), range=(plotmin, plotmax))
wp.plg(C, plotE)
wp.fma()
#####

# ToDo: who wrote this? @Carlos
# You had KE_mean vs. time already, Add another plot KE_max vs. time
# Modify your script to get plot
# On the target plane (when ions exit all the gaps), For E range from 0 to 50 keV, plot number of ions at range of (E, E+0.1keV) as a function of E.


# Zcrossing Particles Plot
# x = targetz_particles.getx() #this is the x coordinate of the particles that made it through target
# t = targetz_particles.getvz()
# print(x)
# print(t)

### Data storage
# save history information, so that we can plot all cells in one plot

t = np.trim_zeros(wp.top.thist, 'b')
hepsny = selectedIons.hepsny[0]
hepsnz = selectedIons.hepsnz[0]
hep6d = selectedIons.hepsx[0] * selectedIons.hepsy[0] * \
        selectedIons.hepsz[0]
hekinz = 1e-6 * 0.5 * wp.top.aion * wp.amu * \
         selectedIons.hvzbar[0] ** 2 / wp.jperev
u = selectedIons.hvxbar[0] ** 2 + selectedIons.hvybar[0] ** 2 + selectedIons.hvzbar[0] ** 2
hekin = 1e-6 * 0.5 * wp.top.aion * wp.amu * u / wp.jperev
hxrms = selectedIons.hxrms[0]
hyrms = selectedIons.hyrms[0]
hrrms = selectedIons.hrrms[0]

hpnum = selectedIons.hpnum[0]
datetimestamp2 = datetime.datetime.now().strftime(
    '%m-%d-%y')

print('debug', t.shape, hepsny.shape)
out = np.stack((t, hepsny, hepsnz, hep6d, hekinz, hekin,
                hxrms, hyrms, hrrms, hpnum))

### END BELOW HERE ARE ONLY COMMENTS

# store files in certain folder related to filename - not used here
# atap_path = Path(r'/home/timo/Documents/Warp/Sims/') #insert your path here

# Convert data into JSON serializable..............................................#nsp = len(x)#"number_surviving_particles" : nsp,fs = len(x)/m, "fraction_particles" : fs,Ve = str(Vesq), se = list(geometry.start_ESQ_gaps), ee = list(geometry.end_ESQ_gaps), "ESQ_start" : se,"ESQ_end" : ee,"Vesq" : Ve,
# t = list(time_time)
# ke = list(KE_select)
# z = list(Z)
# m = max(numsel)
# L = str(L_bunch)
# n = Units * 2
# fA = str(V_arrival)
# sM = int(speciesMass)
# eK = int(ekininit)
# fq = int(freq)
# em = str(emittingRadius)
# dA = str(divergenceAngle)
# pt = list(numsel)
# sa = list(geometry.start_accel_gaps)
# ea = list(geometry.end_accel_gaps)
# json_data = {
#     "data": {
#         "max_particles": m,
#         "time": t,
#         "kinetic_energy": ke,
#         "z_values": z,
#         "particles_overtime": pt,
#         "RF_start": sa,
#         "RF_end": ea
#     },
#     "parameter_dict": {
#
#         "Vmax": Vmax,
#         "L_bunch": L,
#         "numRF": n,
#         'Units': Units,
#         "V_arrival": fA,
#         "speciesMass": sM,
#         "ekininit": eK,
#         "freq": fq,
#         "emittingRadius": em,
#         "divergenceAngle": dA
#
#     }
# }

# Timo
# ZCrossing store closed for autorunner
# json_ZC = {
#     "x": zc.getx().tolist(),
#     "y": zc.gety().tolist(),
#     "vx": zc.getvx().tolist(),
#     "vy": zc.getvy().tolist(),
#     "vz": zc.getvz().tolist(),
#     "t": zc.gett().tolist(),
# }
# with open(f"{step1path}/{cgm_name}_zc.json",
#           "w") as write_file:
#     json.dump(json_ZC, write_file, indent=2)
# now_end = time.time()
# print(f"Runtime in seconds is {now_end - start}")

# Optional plots:
"""#plot history of scraped particles plot for conductors
wp.plg(conductors.get_energy_histogram)
wp.fma()

wp.plg(conductors.plot_energy_histogram)
wp.fma() 

wp.plg(conductors.get_current_history)
wp.fma()

wp.plg(conductors.plot_current_history)
wp.fma()"""

"""wp.plg(t, x, color=wp.green)
wp.ptitles("Spread of survived particles in the x direction")
wp.fma() #last frame -1 in file
"""
