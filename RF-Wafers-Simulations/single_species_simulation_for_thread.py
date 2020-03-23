import warpoptions
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
                                default='2e-9')

#   number of acceleration gaps (must be a multiple of 2)
warpoptions.parser.add_argument('--numRF', dest='numRF',
                                type=int,
                                default='4')  # number of RF gaps

#   voltage on the RF gaps at the peak of the sinusoid
warpoptions.parser.add_argument('--rf_voltage', dest='Vmax',
                                type=float,
                                default='10000')  # will be between 5000 and 10000 most likely 8000

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

#   injeciton energy in eV
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
                                type=str, default='')

#   divergence angle
warpoptions.parser.add_argument('--tstep',
                                dest='timestep',
                                type=float, default='1e-11')

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
# 10-9 for short; 10e-11 for a nice one

# --- keep track of when the particles are born
wp.top.ssnpid = wp.nextpid()
wp.top.tbirthpid = wp.nextpid()

# set input parameters
L_bunch = warpoptions.options.Lbunch
numRF = warpoptions.options.numRF
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
step1path = "/home/timo/Documents/Warp/atap-meqalac" \
            "-simulations/Spectrometer-Sim/step1/"
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
# ToDo why wired numbers?
wp.w3d.xmmin = -23 / 14 * wp.mm
wp.w3d.xmmax = 23 / 14 * wp.mm
wp.w3d.ymmin = -0.02 / 8. * 1.2  #
wp.w3d.ymmax = +0.02 / 8. * 1.2
wp.w3d.zmmin = 0.0
# changes the length of the gist output window.
wp.w3d.zmmax = 23 * wp.mm

# set grid spacing, this is the number of mesh elements in one window
wp.w3d.nx = 50.
wp.w3d.ny = 50.
wp.w3d.nz = 180.
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
wp.top.npinject = 30  # 300  # needed!! macro particles per time step or cell
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
# Todo explain calculation
positionArray = []
a = centerOfFirstRFGap
betalambda0 = 0
betalambda1 = 0
for i in np.arange(0, numRF / 2, 2):
    # a, b, c & d are the positions of the RFs/GND
    a = a - geometry.gapGNDRF / 2 - \
        geometry.copper_thickness - \
        geometry.wafer_thickness / 2 + betalambda1
    b = a + geometry.gapGNDRF + \
        geometry.copper_thickness * 2 + \
        geometry.wafer_thickness
    betalambda0 = wp.sqrt((
                                  ekininit + V_arrival * Vmax * (
                                      2 * i + 1)) * 2 *
                          selectedIons.charge / speciesMass) * 1 / freq / 2
    c = a + betalambda0
    d = b + betalambda0
    betalambda1 = wp.sqrt(
        (ekininit + V_arrival * Vmax * (2 * i + 2))
        * 2 * selectedIons.charge /
        speciesMass) * 1 / freq / 2
    positionArray.append([a, b, c, d])
print(
    f"The {numRF * 2} wafers will be placed at {positionArray}")

# add actual stack
conductors = RF_stack(positionArray,
                          gen_volt(RF_offset))

# ToDo Timo - work here
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
##ToDo Timo ZC
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
beamwidth = []
energy_time = []
starting_particles = []

scraper = wp.ParticleScraper(conductors,
                             lcollectlpdata=True)  # to use until target is fixed to output data properly

# -- name the target particles and count them
# targetz_particles = ZCrossingParticles(zz=targetz, laccumulate=1)
# this is where the actual sim runs
while (wp.top.time < tmax or zmax < zrunmax):
    wp.step(10)  # each plotting step is 10 timesteps
    time_time.append(wp.top.time)

    numsel.append(len(selectedIons.getke()))
    KE_select.append(np.mean(selectedIons.getke()))

    wp.top.pline1 = "V_RF: {:.0f}".format(
        gen_volt(RF_offset)(wp.top.time))  # ToDo check if
    # this is correct
    # wp.top.pline1 = "V_RF: {:.0f}   V_esq: {:.0f}".format(gen_volt(RF_toffset)(wp.top.time), gen_volt_esq(Vesq, False, ESQ_toffset)(wp.top.time))

    # Todo this can be adapted to run an endless stream
    #  of ions, and see what happens
    # inject only for 1 ns, so that we can get onto the rising edge of the RF
    if 0 * wp.ns < wp.top.time < L_bunch:  # changes the beam
        wp.top.finject[0, selectedIons.jslist[0]] = 1
    else:
        wp.top.inject = 0

    Z = selectedIons.getz()
    # ToDo make a version that follows the fastet particle
    if Z.mean() > zmid:  # if the mean distance the particles have travelled is greater than the middle of the frame do this:
        # the velocity of the frame is equal to the mean velocity of the ions
        wp.top.vbeamfrm = selectedIons.getvz().mean()
        # "" for maximal ions
        # wp.top.vbeamfrm = selectedIons.getvz().max()
        solver.gridmode = 0  # oscillates the fields, not sure if this is needed since we already called this at the beginning of the simulation

    # Todo is this needed
    zmin = wp.top.zbeam + wp.w3d.zmmin
    zmax = wp.top.zbeam + wp.w3d.zmmax  # trying to get rid of extra length at the end of the simulation, this is wasting computing power
    # wp.top.zbeam+wp.w3d.zmmax #scales the window length #redefines the end of the simulation tacks on the 53mm

    # create some plots

    # the instantaneous kinetic energy plot
    KE = selectedIons.getke()
    print(np.mean(KE))
    if len(KE) > 0:
        selectedIons.ppzke(color=wp.blue)
        KEmin, KEmax = KE.min(), KE.max()
        while KEmax - KEmin > deltaKE:
            deltaKE += 10e3
    wp.ylimits(0.95 * KEmin,
               0.95 * KEmin + deltaKE)  # is this fraction supposed to match with V_arrival?
    wp.fma()  # third frame in cgm file, repeating

    # the side view field plot
    wp.pfzx(fill=1, filled=1, plotselfe=True, comp='z',
            titles=0, cmin=-1.2 * Vmax / geometry.gapGNDRF,
            cmax=1.2 * Vmax / geometry.gapGNDRF)  # Vmax/geometry.RF_gap #1.2*Vmax/geometry.RF_gap (if want to see 20% increase in electric field) #comp='z': the component of the electric field to plot, 'x', 'y', 'z' or 'E',use 'E' for the magnitude.
    # look at different components of Ez, to confirm the direction, summarize this
    # cmax adjusts the righthand scale of the voltage, not sure how

    # keep track of minimum and maximum birth times of particles
    '''
    #only call this after the window starts moving? When does the window start moving?
    if selectedIons.getz() > drifti: #this is not working
        t_birth_min = selectedIons.tbirth.min()
        t_birth_max = selectedIons.tbirth.max()
        tarray = np.linspace(t_birth_min, t_birth_max, 6)
    '''  # this is not currently working, try without first

    # sort by birthtime (using this while we figure out why above code is not working)
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

    # plot particles on top of feild plot, sort by birthtime and color them accordingly
    colors = [wp.red, wp.yellow, wp.green, wp.blue,
              wp.magenta]
    for m, c in zip(mask, colors):
        wp.plp(selectedIons.getx()[m],
               selectedIons.getz()[m], msize=1.0,
               color=c)  # the selected ions are changing through time
    wp.limits(zmin, zmax)
    wp.ptitles("Particles and Fields", "Z [m]", "X [m]")
    wp.fma()  # fourth frame in cgm file, repeating

    # TODO
    # keep track of when the beam crosses the gaps (for the phase plot at the end)
    # if gap_num_select < len(distances) and len(selectedIons.getz() > 0):
    #     if np.max(selectedIons.getz()) > np.cumsum(distances)[gap_num_select] -0.5*distances[0]:
    #         sct.append(wp.top.time)
    #         gap_num_select += 1

    # the head on particle plot
    selectedIons.ppxy(color=wp.red,
                      titles=0)  # ppxy (particle plot x horizontal axis, y on vertical axis)
    wp.limits(-R, R)
    wp.ylimits(-R, R)
    wp.plg(Y, X, type="dash")
    wp.fma()  # fifth frame in the cgm file, repeating
    # check if a snapshot should be taken (timo)
    saveBeamSnapshot(Z.mean())

# particles in beam plot
wp.plg(numsel, time_time, color=wp.blue)
wp.ptitles("Particle Count vs Time", "Time (s)",
           "Number of Particles")
wp.fma()  # fourth to last frame in cgm file

# plot lost particles with respect to Z
wp.plg(selectedIons.lostpars, wp.top.zplmesh + wp.top.zbeam)
wp.ptitles("Particles Lost vs Z", "Z",
           "Number of Particles Lost")
wp.fma()

"""#plot history of scraped particles plot for conductors
wp.plg(conductors.get_energy_histogram)
wp.fma()

wp.plg(conductors.plot_energy_histogram)
wp.fma()

wp.plg(conductors.get_current_history)
wp.fma()

wp.plg(conductors.plot_current_history)
wp.fma()"""

# above should work for target Z as well however, it has not worked yet

# make an array of starting_particles the same length as numsel
for i in range(len(numsel)):
    p = max(numsel)
    starting_particles.append(p)

# fraction of surviving particles
f_survive = [i / j for i, j in
             zip(numsel, starting_particles)]

# want the particles that just make it through the last RF, need position of RF. This way we can see how many particles made it through the last important component of the accelerator
# last_f_survive =

wp.plg(f_survive, time_time, color=wp.green)
wp.ptitles("Fraction of Surviving Particles vs Time",
           "Time (s)", "Fraction of Surviving Particles")
wp.fma()  # third to last frame in cgm file

# rms envelope plot
wp.hpxrms(color=wp.red, titles=0)
wp.hpyrms(color=wp.blue, titles=0)
wp.hprrms(color=wp.green, titles=0)
wp.ptitles("X(red), Y(blue), R(green)", "Z [m]",
           "X/Y/R [m]", "")
wp.fma()  # second to last frame in cgm file

# Kinetic Energy at certain Z value
wp.plg(KE_select, time_time, color=wp.blue)
wp.limits(0, 70e-9)  # limits(xmin,xmax,ymin,ymax)
wp.ptitles("Kinetic Energy vs Time")
wp.fma()

# kinetic energy plot
wp.plg(KE_select, time_time, color=wp.blue)
wp.ptitles("kinetic energy vs time")
wp.fma()  # last frame in cgm file

# Zcrossing Particles Plot
# x = targetz_particles.getx() #this is the x coordinate of the particles that made it through target
# t = targetz_particles.getvz()
# print(x)
# print(t)
"""wp.plg(t, x, color=wp.green)
wp.ptitles("Spread of survived particles in the x direction")
wp.fma() #last frame -1 in file
"""
# save history information, so that we can plot all cells in one plot
t = np.trim_zeros(wp.top.thist, 'b')
hepsny = selectedIons.hepsny[0]
hepsnz = selectedIons.hepsnz[0]
hep6d = selectedIons.hepsx[0] * selectedIons.hepsy[0] * \
        selectedIons.hepsz[0]
hekinz = 1e-6 * 0.5 * wp.top.aion * wp.amu * \
         selectedIons.hvzbar[0] ** 2 / wp.jperev

u = selectedIons.hvxbar[0] ** 2 + selectedIons.hvybar[
    0] ** 2 + selectedIons.hvzbar[0] ** 2
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

# store files in certain folder related to filename - not used here
# atap_path = Path(r'/home/timo/Documents/Warp/Sims/') #insert your path here

# Convert data into JSON serializable..............................................#nsp = len(x)#"number_surviving_particles" : nsp,fs = len(x)/m, "fraction_particles" : fs,Ve = str(Vesq), se = list(geometry.start_ESQ_gaps), ee = list(geometry.end_ESQ_gaps), "ESQ_start" : se,"ESQ_end" : ee,"Vesq" : Ve,
t = list(time_time)
ke = list(KE_select)
z = list(Z)
m = max(numsel)
L = str(L_bunch)
n = numRF
fA = str(V_arrival)
sM = int(speciesMass)
eK = int(ekininit)
fq = int(freq)
em = str(emittingRadius)
dA = str(divergenceAngle)
pt = list(numsel)
sa = list(geometry.start_accel_gaps)
ea = list(geometry.end_accel_gaps)
json_data = {
    "data": {
        "max_particles": m,
        "time": t,
        "kinetic_energy": ke,
        "z_values": z,
        "particles_overtime": pt,
        "RF_start": sa,
        "RF_end": ea
    },
    "parameter_dict": {

        "Vmax": Vmax,
        "L_bunch": L,
        "numRF": n,
        "V_arrival": fA,
        "speciesMass": sM,
        "ekininit": eK,
        "freq": fq,
        "emittingRadius": em,
        "divergenceAngle": dA

    }
}

# Timo
# ZCrossing store
json_ZC = {
    "x": zc.getx().tolist(),
    "y": zc.gety().tolist(),
    "vx": zc.getvx().tolist(),
    "vy": zc.getvy().tolist(),
    "vz": zc.getvz().tolist(),
    "t": zc.gett().tolist(),
}
with open(f"{step1path}/{cgm_name}_zc.json",
          "w") as write_file:
    json.dump(json_ZC, write_file, indent=2)

now_end = time.time()
print(f"Runtime in seconds is {now_end - start}")
