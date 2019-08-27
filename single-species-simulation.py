import warpoptions

#   get input parameters
#   you don't have to put all parameters into the command line,
#   only the ones you want to change from their default vaues
#   you can get a good idea of what these values should be from the ExcelSimulationOfMassSeperation.xlsx
#   or from interactivePlot.ipynb
#   use mathmatica to find out esq_voltage

#   only specify the parameters you want to change from their default values
#   the input will look like:
"""
python single-species-simulation.py --esq_voltage=500 --fraction=.8 --speciesMass=20 --ekininit=15e3
"""
#   bunch length
warpoptions.parser.add_argument('--bunch_length', dest='Lbunch', type=float, default='1e-9')
#   voltage on the focusing quads
warpoptions.parser.add_argument('--numRF', dest='numRF', type=int, default='4')
#   voltage on the RF gaps at the peak of the sinusoid
warpoptions.parser.add_argument('--rf_voltage', dest='Vmax', type=float, default='5000') #we can play with this
#   fraction of the max voltage at which the selected ions cross the gap
warpoptions.parser.add_argument('--esq_voltage', dest='Vesq', type=float, default='.01') #850
#   total number of RF acceleration gaps (must be a multiple of 2)
warpoptions.parser.add_argument('--fraction', dest='V_arrival', type=float, default='1') #.8
#   mass of the ions being accelerated
warpoptions.parser.add_argument('--species_mass', dest='speciesMass', type=int, default='40')
#   injeciton energy in eV
warpoptions.parser.add_argument('--ekininit', dest='ekininit', type=float, default='10e3')
#   frequency of the RF
warpoptions.parser.add_argument('--freq', dest='freq', type=float, default='13.56e6') #27e6
#   emitting radius
warpoptions.parser.add_argument('--emit', dest='emittingRadius', type=float, default='.25e-3') #.37e-3 #.25
#   divergence angle
warpoptions.parser.add_argument('--diva', dest='divergenceAngle', type=float, default='5e-3')

import warp as wp
import numpy as np
import geometry
from geometry import ESQ, RF_stack3, Gap, mid_gap
from helper import gitversion
import matplotlib.pyplot as plt
import datetime
import time
import math
import os
from pathlib import Path
import json
import sys
#import particlescraper

start = time.time()

wp.w3d.solvergeom = wp.w3d.XYZgeom
wp.top.dt = 5e-9#5e-11

# --- keep track of when the particles are born
wp.top.ssnpid = wp.nextpid()
wp.top.tbirthpid = wp.nextpid()

#set input parameters
L_bunch = warpoptions.options.Lbunch
numRF = warpoptions.options.numRF
Vmax = warpoptions.options.Vmax #RF voltage
Vesq = warpoptions.options.Vesq
V_arrival = warpoptions.options.V_arrival #fraction of the total voltage gained across each gap
speciesMass = warpoptions.options.speciesMass*wp.amu
selectedIons = wp.Species(charge_state=1, name='Ar', mass=warpoptions.options.speciesMass*wp.amu, color=wp.green)
ekininit = warpoptions.options.ekininit
freq = warpoptions.options.freq #RF freq
emittingRadius = warpoptions.options.emittingRadius
divergenceAngle = warpoptions.options.divergenceAngle

# --- Invoke setup routine for the plotting (name the cgm output file)

#add a date & timestamp to the cgm file
now = datetime.datetime.now()
datetimestamp = datetime.datetime.now().strftime('%m-%d-%y_%H:%M:%S')

a = b = c = d = e = f = g = 0 #placeholder values to keep track of how many parameters were changed

#find which change was made to record in the file name
if L_bunch != 1e-9:
    parameter_name = "L_bunch"
    change = L_bunch
    a = 1
elif numRF != 4:
    parameter_name = "numRF"
    change = numRF
    b = 1
elif Vmax != 5000:
    parameter_name = "RF_Voltage"
    change = Vmax
    c = 1
elif Vesq != .01:
    parameter_name = "ESQ_Voltage"
    change = Vesq
    d = 1
elif V_arrival != 1:
    parameter_name = "Fraction_of_Arrival_Voltage"
    change = V_arrival
    e = 1
elif emittingRadius != .25e-3:
    parameter_name = "Emitting_Radius"
    change = emittingRadius
    f = 1
elif divergenceAngle != 5e-3:
    parameter_name = "Divergence_Angle"
    change = divergenceAngle
    g = 1
else:
    answer = input("You are using the basic parameters, \n y : continue , x : exit ")
    type(answer)
    if answer == 'x':
        answer = False
        sys.exit()
    elif answer == 'y':
        answer = True
        parameter_name = "All_Origional_Parameters"
        change = "null"

total = a + b + c + d + e + f + g

print(f"{total} parameters were changed......{parameter_name} changed to a value of {change}........................................................")

#make cgm file name depend on what was changed

#name files based on date and above parameters
cgm_name = f"{parameter_name}__{change}__{datetimestamp}"

wp.setup(prefix=f"{cgm_name}", cgmlog= 0)

# --- Set basic beam parameters, these should be calculated in mathmatica first

ibeaminit = 10e-6 # inital beam current May vary up to 25e-6

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
wp.w3d.l4symtry = True
wp.w3d.l2symtry = False

# --- Set boundary conditions

# ---   for field solve
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.neumann

# ---   for particles
wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb
wp.top.prwall = .5*wp.mm #should use particles scraper for this. Keep a prwall however make it slightly bigger so that the ions can get absorbed by conductors

# --- Set field grid size, this is the width of the window
wp.w3d.xmmin = -23/14*wp.mm
wp.w3d.xmmax = 23/14*wp.mm
wp.w3d.ymmin = -0.02/8.*1.2
wp.w3d.ymmax = +0.02/8.*1.2
wp.w3d.zmmin = 0.0
wp.w3d.zmmax = 53*wp.mm #changes the length of the gist output window

# set grid spacing, this is the number of mesh elements in one window
wp.w3d.nx = 50.
wp.w3d.ny = 50.
wp.w3d.nz = 180.

if wp.w3d.l4symtry:
    wp.w3d.xmmin = 0.
    wp.w3d.nx /= 2
if wp.w3d.l2symtry or wp.w3d.l4symtry:
    wp.w3d.ymmin = 0.
    wp.w3d.ny /= 2

# --- Select plot intervals, etc.
wp.top.npmax = 300
wp.top.inject = 1  # 2 means space-charge limited injection
wp.top.rinject = 5000 #what is this?
wp.top.npinject = 30  # 300  # needed!! macro particles per time step or cell
wp.top.linj_eperp = False  # Turn on transverse E-fields near emitting surface
wp.top.zinject = wp.w3d.zmmin
wp.top.vinject = 1.0 #what does this do?
print("--- Ions start at: ", wp.top.zinject)

wp.top.nhist = 5  # Save history data every N time step
wp.top.itmomnts[0:4] = [0, 1000000, wp.top.nhist, 0]  # Calculate moments every N steps
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

# --- define voltages
Vground = 0.0e3 #do we actually need to do this? It is not being defined anywhere
VRF = 1000.0 #" "

ESQs = []
RFs = []
ID_ESQ = 100
ID_RF = 201

# --- generates voltage for the RFs
def gen_volt(toffset=0): #0
    """ A sin voltage function with variable offset"""
    def RFvoltage(time):
        return Vmax*np.sin(2*np.pi*freq*(time+toffset))
    return RFvoltage

# --- generates voltage for the ESQs
def gen_volt_esq(Vesq, inverse=False, toffset=0):
    def ESQvoltage(time):
        if inverse:
            return -Vesq#*np.sin(2*np.pi*freq*(time+toffset))
        else:
            return Vesq#*np.sin(2*np.pi*freq*(time+toffset))
    return ESQvoltage

# --- calculate the distances and time offset for the RFs

energies = [ekininit + V_arrival*Vmax*(i+1) for i in range(numRF)] #(i+1) to start at proper 15,000 energy gain
distances = [wp.sqrt(energy*selectedIons.charge/(2*speciesMass))*1/freq for energy in energies] #beta lambda/2
drifti = 10.05*wp.mm #added to start the particles well before the first RF wafer
geometry.pos = -0.5*distances[0] - .5*geometry.RF_gap - geometry.RF_thickness + drifti #starting position of particles
d_mid = geometry.pos + geometry.RF_thickness + .5*geometry.RF_gap #middle of the first gap
veloinit = np.sqrt(2*ekininit*selectedIons.charge/speciesMass) #initial velocity of particles
t_offset = (d_mid - geometry.RF_thickness - .5*geometry.RF_gap)/veloinit #phase offset to account for moving the injection beam farther to the left
print("TIME OFFSET = {}".format(t_offset)) #not the same as 8e-9
RF_toffset = np.arcsin(V_arrival)/(2*np.pi*freq) - (d_mid + drifti)/veloinit + 8e-9#.5*L_bunch - 1e-9# L_bunch - 8e-9
ESQ_toffset = 0

Vpos = []

def pairwise(it):
    """Return two items from a list per iteration"""
    it = iter(it)
    while True:
        try:
            yield next(it), next(it)
        except StopIteration:
            return

# this loop generates the geometry
for i, bl2s in enumerate(pairwise(zip(distances, energies))):
    (rf_bl2, E1), (esq_bl2, E2) = bl2s #calling distances rf_bl2s
    # use first betalamba_half for the RF unit
    RF = RF_stack3(condid=[ID_RF, ID_RF+1, ID_RF+2],
            betalambda_half=rf_bl2 ,voltage=gen_volt(RF_toffset))
    Vpos.append(geometry.pos)

    # scale esq voltages
    voltage = Vesq * E2/ekininit #why do we need to do this? -MWG

    # and second betalamba_half for ESQ unit
    gaplength = esq_bl2-geometry.RF_gap-2*geometry.RF_thickness
    gaplength = gaplength/2-geometry.ESQ_gap/2-geometry.ESQ_wafer_length
    assert gaplength > 0
    Gap(gaplength)
    E1 = ESQ(voltage=gen_volt_esq(Vesq, False, ESQ_toffset), condid=[ID_ESQ, ID_ESQ+1])
    Gap(geometry.ESQ_gap)
    E2 = ESQ(voltage=gen_volt_esq(Vesq, True, ESQ_toffset), condid=[ID_ESQ+2, ID_ESQ+3])
    Gap(gaplength)
    #add more ESQs to see the effect
    
    
    ESQs.append(E1)
    ESQs.append(E2)
    RFs.append(RF)
    ID_ESQ += 4
    ID_RF += 3

conductors = wp.sum(ESQs) + wp.sum(RFs) #names all ESQs and RFs conductors in order to feed into warp

velo = np.sqrt(2*ekininit*selectedIons.charge/selectedIons.mass) #used to caluclate tmax
length = geometry.pos + 2.5*wp.cm #2.5mm added to allow particles to completely pass through last RF gap
tmax = length/velo #this is used for the maximum time for timesteps
zrunmax = length #this is used for the maximum distance for timesteps

# define the electrodes
wp.installconductors(conductors)

# --- Recalculate the fields
wp.fieldsol(-1)

solver.gridmode = 0 #makes the fields oscillate properly at the beginning

# I want contour plots for levels between 0 and 1kV
#contours = range(0, int(Vesq), int(Vesq/10))

wp.winon(xon=0)

# some plots of the geometry
wp.pfzx(fill=1, filled=1, plotphi=0)
wp.fma()
wp.pfzx(fill=1, filled=1, plotphi=1)
wp.fma()

zmin = wp.w3d.zmmin
zmax = wp.w3d.zmmax
zmid = 0.5*(zmax+zmin)

# make a circle to show the beam pipe
R = .5*wp.mm #beam radius
t = np.linspace(0, 2*np.pi, 100)
X = R*np.sin(t)
Y = R*np.cos(t)
deltaKE = 10e3
time_time = []
numsel = []
KE_select = []
beamwidth=[]
energy_time = []
starting_particles = []

dist = 2.5*wp.mm #why is this here?
distN = 0

sct = [] #when the particles cross the acceleration gaps
gap_num_select = 0

geometry.mid_gap


while (wp.top.time < tmax and zmax < zrunmax):
    wp.step(10) # each plotting step is 10 timesteps
    time_time.append(wp.top.time)

    numsel.append(len(selectedIons.getke()))
    KE_select.append(np.mean(selectedIons.getke()))

    wp.top.pline1 = "V_RF: {:.0f}   V_esq: {:.0f}".format(
        gen_volt(RF_toffset)(wp.top.time), gen_volt_esq(Vesq, False, ESQ_toffset)(wp.top.time))
        
    # inject only for 1 ns, so that we can get onto the rising edge of the RF
    if 0*wp.ns < wp.top.time < L_bunch: #changes the beam length
        wp.top.finject[0,selectedIons.jslist[0]] = 1
    else:
        wp.top.inject = 0

    Z = selectedIons.getz()
    if Z.mean() > zmid:
        wp.top.vbeamfrm = selectedIons.getvz().mean()
        solver.gridmode = 0

    # record time when we cross certain points
    if Z.mean() > dist + np.cumsum(distances)[distN]:
        energy_time.append(wp.top.time)
        distN += 1

    zmin = wp.top.zbeam+wp.w3d.zmmin
    zmax = wp.top.zbeam+wp.w3d.zmmax #scales the window length

    # create some plots

    # the instantaneous kinetic energy plot
    KE = selectedIons.getke()
    print(np.mean(KE))
    if len(KE) > 0:
        selectedIons.ppzke(color=wp.blue)
        KEmin, KEmax = KE.min(), KE.max()
        while KEmax-KEmin > deltaKE:
            deltaKE += 10e3
    wp.ylimits(0.95*KEmin, 0.95*KEmin+deltaKE) #is this fraction supposed to match with V_arrival?
    wp.fma()

    # the side view field plot
    wp.pfzx(fill=1, filled=1, plotselfe=True, comp='E', titles=0, cmin=0, cmax=1.2*Vmax/geometry.RF_gap)
    #cmax adjusts the righthand scale of the voltage, not sure how

    #keep track of minimum and maximum birth times of particles
    '''#only call this after the window starts moving? When does the window start moving?
    if selectedIons.getz() > drifti: #this is not working
        t_birth_min = selectedIons.tbirth.min()
        t_birth_max = selectedIons.tbirth.max()
        tarray = np.linspace(t_birth_min, t_birth_max, 6)''' #this is not currently working, try without first

    #sort by birthtime (using this while we figure out why above code is not working)
    t_birth_min = selectedIons.tbirth.min()
    t_birth_max = selectedIons.tbirth.max()
    tarray = np.linspace(t_birth_min, t_birth_max, 6)

    #mask to sort particles by birthtime
    mask = []
    for i in range(len(tarray)-1): #the length of tarray must be changing overtime which changes the mask which recolors the particles
        m = (selectedIons.tbirth > tarray[i])*(selectedIons.tbirth < tarray[i+1])
        mask.append(m)

    #plot particles on top of feild plot, sort by birthtime and color them accordingly
    colors = [wp.red, wp.yellow, wp.green, wp.blue, wp.magenta]
    for m, c in zip(mask, colors):
        wp.plp(selectedIons.getx()[m], selectedIons.getz()[m], msize=1.0, color = c) #the selected ions are changing through time
    wp.limits(zmin, zmax)
    wp.ptitles("Particles and Fields", "Z [m]", "X [m]")
    wp.fma()

    # keep track of when the beam crosses the gaps (for the phase plot at the end)
    if gap_num_select < len(distances) and len(selectedIons.getz() > 0):
        if np.max(selectedIons.getz()) > np.cumsum(distances)[gap_num_select] -0.5*distances[0]:
            sct.append(wp.top.time)
            gap_num_select += 1

    # the head on particle plot
    selectedIons.ppxy(color=wp.red, titles=0) #ppxy (particle plot x horizontal axis, y on vertical axis)
    wp.limits(-R, R)
    wp.ylimits(-R, R)
    wp.plg(Y, X, type="dash")
    wp.fma()

# particles in beam plot
wp.plg(numsel, time_time, color=wp.blue)
wp.ptitles("Particle Count vs Time", "Time (s)", "Number of Particles")
wp.fma()

#make an array of starting_particles the same length as numsel
for i in range(len(numsel)):
    p = max(numsel)
    starting_particles.append(p)

#fraction of surviving particles
f_survive = [i / j for i, j in zip(numsel, starting_particles)]

wp.plg(f_survive, time_time, color = wp.green)
wp.ptitles("Fraction of Surviving Particles vs Time", "Time (s)", "Fraction of Surviving Particles")
wp.fma()

# rms envelope plot
wp.hpxrms(color=wp.red, titles=0)
wp.hpyrms(color=wp.blue, titles=0)
wp.hprrms(color=wp.green, titles=0)
wp.ptitles("X(red), Y(blue), R(green)", "Z [m]", "X/Y/R [m]", "")
wp.fma()

# kinetic energy plot
wp.plg(KE_select, time_time, color=wp.blue)
wp.ptitles("kinetic energy vs time")
wp.fma()

# save history information, so that we can plot all cells in one plot
t = np.trim_zeros(wp.top.thist, 'b')
hepsny = selectedIons.hepsny[0]
hepsnz = selectedIons.hepsnz[0]
hep6d = selectedIons.hepsx[0] * selectedIons.hepsy[0] * selectedIons.hepsz[0]
hekinz = 1e-6*0.5*wp.top.aion*wp.amu*selectedIons.hvzbar[0]**2/wp.jperev

u = selectedIons.hvxbar[0]**2 + selectedIons.hvybar[0]**2 + selectedIons.hvzbar[0]**2
hekin = 1e-6 * 0.5*wp.top.aion*wp.amu*u/wp.jperev

hxrms = selectedIons.hxrms[0]
hyrms = selectedIons.hyrms[0]
hrrms = selectedIons.hrrms[0]
hpnum = selectedIons.hpnum[0]

datetimestamp2 = datetime.datetime.now().strftime('%m-%d-%y')

print('debug', t.shape, hepsny.shape)
out = np.stack((t, hepsny, hepsnz, hep6d, hekinz, hekin, hxrms, hyrms, hrrms, hpnum))

#store files in certian folder related to filename
atap_path = Path(r'/Users/mwgarske/atap-meqalac-simulations')


#JSON

#change arrays into lists to feed into JSON

fs = list(f_survive)
sp = list(numsel)
t = list(time_time)
ke = list(KE)
z = list(Z)

#change all values into integers/strings to feed into JSON
m = max(numsel)
L = str(L_bunch)
n = numRF
Ve = str(Vesq)
fA = str(V_arrival)
sM = int(speciesMass)
eK = int(ekininit)
fq = int(freq)
em = str(emittingRadius)
dA = str(divergenceAngle)


json_data = {
    "max_particles" : m,
    "fraction_particles" : fs,
    "surviving_particles" : sp,
    "time" : t,
    "kinetic_energy" : ke,
    "z_values" : z,
    "parameter_dict": {
        
        "Vmax" : Vmax,
        "L_bunch" : L,
        "numRF" : n,
        "Vesq" : Ve,
        "V_arrival" : fA,
        "speciesMass" : sM,
        "ekininit" : eK,
        "freq" : fq,
        "emittingRadius" : em,
        "divergenceAngle" : dA

        }
    }

#print(json.dumps(json_data, indent=4))

with open(f"{parameter_name}__{change}__{datetimestamp}__surviving_particles.json", "w") as write_file:
    json.dump(json_data, write_file)

"""print(json.dumps(json_data, indent=4, sort_keys=True))""" #pretty print

"""with open(f"{parameter_name}__{change}__{datetimestamp}__surviving_particles.txt", "a+") as f:
    f.write(f"{parameter_name} = {change} \n # timestamp: {datetimestamp} parameters: Length:{L_bunch}_gap:{numRF}_VRF:{Vmax/1000}e{3}_Vesq:{Vesq/1000}_frac:{V_arrival}_emitR:{emittingRadius/.001}e-{3}_divA:{divergenceAngle/(.001)}e-{3} \n")
    f.write(str(numsel))""" #not needed now that we are formatting in JSON

now_end = time.time()
print(f'Runtime in seconds is {now_end-start}')

#change into the correct directory based off of the parameter change
if not os.path.isdir(f"{parameter_name}"):
    #make a new directory
    os.system(f"mkdir {atap_path}/{parameter_name}")
    print("The path did not exist, but I have made it")

np.save(f"{atap_path}/{parameter_name}/esqhist_{datetimestamp2}.npy", out)
#np.save(f"esqhist_{datetimestamp2}.npy", out)

#move files to their respective folders
os.system(f"mv {parameter_name}__{change}__{datetimestamp}__surviving_particles.txt {atap_path}/{parameter_name}")
os.system(f"mv {cgm_name}* {atap_path}/{parameter_name}")

print("<<<<<<<<<<..........All files have been successfully moved..........>>>>>>>>>>")
