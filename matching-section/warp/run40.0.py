import warpoptions
#   only specify the parameters you want to change from their default values
#   the input will look like:
"""
python3 single-species-simulation.py --esq_voltage=500 --fraction=.8 --speciesMass=20 --ekininit=15e3
"""

dt_string = "40.0"




#   bunch length
warpoptions.parser.add_argument(
    "--bunch_length", dest="Lbunch", type=float, default="1e-9"
)

#   number of RF units
warpoptions.parser.add_argument(
    "--units", dest="Units", type=int, default="2"
)  # number of RF gaps

#   voltage on the RF gaps at the peak of the sinusoid
warpoptions.parser.add_argument(
    "--rf_voltage", dest="Vmax", type=float, default="8000"
)  # will be between 4000 and 10000

# voltage on the Vesq
warpoptions.parser.add_argument(
    "--esq_voltage", dest="Vesq", type=float, default=".01"
)  # 850

#   fraction of the max voltage at which the selected ions cross
warpoptions.parser.add_argument(
    "--fraction", dest="V_arrival", type=float, default="1"
)  # .8

#   mass of the ions being accelerated
warpoptions.parser.add_argument(
    "--species_mass", dest="speciesMass", type=int, default="40"
)

#   injection energy in eV
warpoptions.parser.add_argument(
    "--ekininit", dest="ekininit", type=float, default="8e3"
)

#   frequency of the RF
warpoptions.parser.add_argument(
    "--freq", dest="freq", type=float, default="14.8e6"
)  # 27e6

#   emitting radius
warpoptions.parser.add_argument(
    "--emit", dest="emittingRadius", type=float, default="0.3e-3"
)  # .37e-3 #.25

#   divergence angle
warpoptions.parser.add_argument(
    "--diva", dest="divergenceAngle", type=float, default="9e-3"
)
from datetime import datetime

# datetime object containing current date and time
now = datetime.now()
# dd/mm/YY H:M:S


#   special cgm name - for mass output / scripting
warpoptions.parser.add_argument("--name", dest="name", type=str, default=dt_string)

#   special cgm path - for mass output / scripting
warpoptions.parser.add_argument("--path", dest="path", type=str, default="")

#   timestep
warpoptions.parser.add_argument(
    "--tstep", dest="timestep", type=float, default="2.5e-9"
)  # 1e-11

#   Volt ratio for ESQs @ToDo Zhihao : is this correct?
warpoptions.parser.add_argument(
    "--volt_ratio", dest="volt_ratio", type=float, default="1.04"
)

#   enables some additional code if True
warpoptions.parser.add_argument("--autorun", dest="autorun", type=bool, default=False)

# sets wp.steps(#)
warpoptions.parser.add_argument("--plotsteps", dest="plotsteps", type=int, default=1)

# changes simulation to a "cb-beam" simulation
warpoptions.parser.add_argument("--cb", dest="cb_framewidth", type=float, default=0)

# enables a Z-Crossing location, saving particles that are crossing the given z-value
warpoptions.parser.add_argument("--zcrossing", dest="zcrossing", type=float, default=0)

# set maximal running time, this disables other means of ending the simulation
warpoptions.parser.add_argument("--runtime", dest="runtime", type=float, default=0)

# beam current initial beam current may vary up to 25e-6
warpoptions.parser.add_argument(
    "--ibeaminit", dest="ibeaminit", type=float, default=0.09e-6
)

#   stores the beam at once the average particle position passed the given positions (list)
# --storebeam "[0.01, 0.024, 0.045]" uses the --name given for the simulation. Stored beams are ordered.
# Needs to be an array with stings
warpoptions.parser.add_argument("--storebeam", dest="storebeam", default="[]")
# --loadbeam "path/to/file.json"
warpoptions.parser.add_argument("--loadbeam", dest="loadbeam", type=str, default="")
# --beamnumber 3  or negative number to count from the back. Stored beams are ordered.
warpoptions.parser.add_argument("--beamnumber", dest="beamnumber", type=int, default=-1)

#--Python packages
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
import pandas as pd 


#--Import third-party packages
import warp as wp
from warp.particles.extpart import ZCrossingParticles

#--Import custom packages
from geometry_test7 import RF_stack, ESQ_doublet, ESQ_singlet
import geometry


# # There is no Matplotlib installed on Lawrencium
# from matplotlib import pyplot as plt

start = time.time()

wp.w3d.solvergeom = wp.w3d.XYZgeom
# Timesteps
wp.top.dt = warpoptions.options.timestep
# 10-9 for short; 10e-11 for a nice one;

# --- keep track of when the particles are born
wp.top.ssnpid = wp.nextpid()
wp.top.tbirthpid = wp.nextpid()

# set input parameters
L_bunch = warpoptions.options.Lbunch
Units = warpoptions.options.Units
Vmax = warpoptions.options.Vmax  # RF voltage
Vesq = warpoptions.options.Vesq
V_arrival = (
    warpoptions.options.V_arrival
)  # fraction of the total voltage gained across each gap
speciesMass = warpoptions.options.speciesMass * wp.amu
selectedIons = wp.Species(
    charge_state=1,
    name="Ar",
    mass=warpoptions.options.speciesMass * wp.amu,
    color=wp.green,
)
ekininit = warpoptions.options.ekininit
freq = warpoptions.options.freq  # RF freq
emittingRadius = warpoptions.options.emittingRadius
divergenceAngle = warpoptions.options.divergenceAngle
name = warpoptions.options.name
storebeam = warpoptions.options.storebeam
loadbeam = warpoptions.options.loadbeam
beamnumber = warpoptions.options.beamnumber
ibeaminit = warpoptions.options.ibeaminit

# --- where to store the outputfiles
cgm_name = name
step1path = "."
step1path = os.getcwd()

# overwrite if path is given by command
if warpoptions.options.path != "":
    step1path = warpoptions.options.path

wp.setup(prefix=f"{step1path}/{cgm_name}")  # , cgmlog= 0)

### read / write functionality #ToDo: move into helper file
basepath = warpoptions.options.path
if basepath == "":
    basepath = f"{step1path}/"
thisrunID = warpoptions.options.name


def initjson(fp=f"{basepath}{thisrunID}.json"):
    if not os.path.isfile(fp):
        print(f"Saving new Json")
        with open(fp, "w") as writefile:
            json.dump({}, writefile, sort_keys=True, indent=1)


def readjson(fp=f"{basepath}{thisrunID}.json"):
    initjson(fp)
    with open(fp, "r") as readfile:
        data = json.load(readfile)
    return data


def writejson(key, value, fp=f"{basepath}{thisrunID}.json"):
    print(f"Writing Data to json {fp}")
    # print("WRITING DATA")
    # print(f" KEY {key} \n VALUE {value}")
    writedata = readjson(fp)
    writedata[key] = value
    with open(fp, "w") as writefile:
        json.dump(writedata, writefile, sort_keys=True, indent=1)


### loading beam functionalitly
def restorebeam(nb_beam=beamnumber):
    if loadbeam != "":
        rj = readjson(loadbeam)
        print(rj["storedbeams"])
        beamdata = rj["storedbeams"][nb_beam]
        # those things need to be overwritten
        selectedIons.addparticles(
            x=beamdata["x"],
            y=beamdata["y"],
            z=beamdata["z"],
            vx=beamdata["vx"],
            vy=beamdata["vy"],
            vz=beamdata["vz"],
            lallindomain=True,
        )
        wp.top.time = beamdata["t"]
        wp.w3d.zmmin = beamdata["framecenter"] - framewidth / 2
        wp.w3d.zmmax = beamdata["framecenter"] + framewidth / 2
        wp.top.vbeamfrm = selectedIons.getvz().mean()
        wp.top.inject = 0
        wp.top.npmax = len(beamdata["z"])


# --- Set basic beam parameters

wp.top.vbeam = 0.0e0
wp.top.emit = 1.42E-6
wp.top.ibeam = ibeaminit
wp.top.ekin = ekininit
wp.top.zion = selectedIons.charge_state
wp.top.lrelativ = False
wp.top.linj_efromgrid = True
wp.derivqty()

# --- Set input parameters describing the 3d simulation

wp.w3d.l4symtry = False  # True
wp.w3d.l2symtry = False

# ---   Set boundary conditions

# ---   for field solve
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.neumann

# ---   for particles
wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb
wp.top.prwall = (
    1 * wp.mm
)  # prwall slightly bigger than aperture radius so ions can get absorbed by conductors

# --- Set field grid size, this is the width of the window
wp.w3d.xmmin = -2 * wp.mm
wp.w3d.xmmax = 2 * wp.mm
wp.w3d.ymmin = -2 * wp.mm  #
wp.w3d.ymmax = 2 * wp.mm
framewidth = 60 * wp.mm

restorebeam()

if loadbeam == "":
    # changes the length of the gist output window.
    wp.w3d.zmmin = 0.0
    wp.w3d.zmmax = wp.w3d.zmmin + framewidth

    wp.top.npmax = 8000  # maximal number of particles (for injection per timestep???)
    wp.top.inject = 1  # 2 means space-charge limited injection
    wp.top.npinject = 200 # approximate number of particles injected per step

    wp.top.ainject = emittingRadius
    wp.top.binject = emittingRadius
    wp.top.apinject = divergenceAngle
    wp.top.bpinject = divergenceAngle
    #wp.top.rinject = 5000  # emitting surface curvature
    wp.top.linj_eperp = False  # Turn on transverse E-fields near emitting surface
    wp.top.vinject = 1  # source voltage

    wp.top.xinject = 0
    wp.top.yinject = 0
    wp.top.zinject = wp.w3d.zmmin

# set grid spacing, this is the number of mesh elements in one window
wp.w3d.nx = 30 #60. < for higher resolution
wp.w3d.ny = 30 #60.
### There are two 35um copper layers on ESQ, so we need high resolution.
# 180 / 23mm < 8mm⁻¹
wp.w3d.nz = 150.0  # 180 for 23 # 6-85 for 10

# overwrite if the beam is cb:
if warpoptions.options.cb_framewidth:
    wp.w3d.zmmax = warpoptions.options.cb_framewidth
    wp.w3d.nz = 8e3 * warpoptions.options.cb_framewidth

if wp.w3d.l4symtry:
    wp.w3d.xmmin = 0.0
    wp.w3d.nx /= 2
if wp.w3d.l2symtry or wp.w3d.l4symtry:
    wp.w3d.ymmin = 0.0
    wp.w3d.ny /= 2

# --- Select plot intervals, etc.
# print("--- Ions start at: ", wp.top.zinject)

wp.top.nhist = 1  # Save history data every N time step
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

ESQs = []
RFs = []
ID_ESQ = 100
ID_RF = 201
ID_target = 1


# --- generates voltage for the RFs
def gen_volt(toffset=0, frequency=freq):  # 0
    """ A cos voltage function with variable offset"""

    def RFvoltage(time):
        return -Vmax * np.cos(2 * np.pi * frequency * (time + toffset))

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
tParticlesAtCenterFirstGap = centerOfFirstRFGap / wp.sqrt(
    2 * ekininit * selectedIons.charge / speciesMass
)
print(
    f"Particles need {tParticlesAtCenterFirstGap * 1e6}us"
    f" to reach the center of the first gap"
)
# RF should be maximal when particles arrive
# time for the cos to travel there minus the time the
# particles take
RF_offset = centerOfFirstRFGap / wp.clight - tParticlesAtCenterFirstGap
print(f"RF_offset {RF_offset}")
ESQ_toffset = 0

Vpos = []


# calculating the ideal positions
def calculateRFwaferpositions():
    positionArray = []
    # Calculating first position
    # this is not actually C but the very first wafer a
    c = (
        centerOfFirstRFGap
        - geometry.gapGNDRF / 2
        - geometry.copper_thickness
        - geometry.wafer_thickness / 2
    )
    betalambda0 = 0
    betalambda1 = 0
    for i in np.arange(0, Units):
        # a, b, c & d are the positions of the center of RFs/GND wafers
        # GND RF RF GND
        a = c + betalambda0
        b = (
            a
            + geometry.gapGNDRF
            + geometry.copper_thickness * 2
            + geometry.wafer_thickness
        )
        betalambda1 = (
            wp.sqrt(
                (ekininit + V_arrival * Vmax * (2 * i + 1))
                * 2
                * selectedIons.charge
                / speciesMass
            )
            * 1
            / freq
            / 2
        )
        c = a + betalambda1
        d = b + betalambda1
        betalambda0 = (
            wp.sqrt(
                (ekininit + V_arrival * Vmax * (2 * i + 2))
                * 2
                * selectedIons.charge
                / speciesMass
            )
            * 1
            / freq
            / 2
        )

        if Units == 1:
            c = c - 10 * wp.mm
        elif Units == 2:
            a = a - 50 * wp.mm
            b = b + 40 * wp.mm

        positionArray.append([a, b, c, d])
    return positionArray


def rrms():
    x_dis = selectedIons.getx()
    y_dis = selectedIons.gety()

    xrms = np.sqrt(np.mean(x_dis ** 2))
    yrms = np.sqrt(np.mean(y_dis ** 2))
    rrms = np.sqrt(np.mean(x_dis ** 2 + y_dis ** 2))

    print(f" XRMS: {xrms} \n YRMS: {yrms} \n RRMS: {rrms}")

    return rrms


# Here it is optional to overwrite the position Array, to
# simulate the ACTUAL setup:
calculatedPositionArray = calculateRFwaferpositions()
# print(calculatedPositionArray)
positionArray = [
]
# for 9kV
# positionArray = [[.0036525,.0056525,0.01243279,0.01463279],
#                  [0.0211854,0.0233854,0.03130207,0.03330207],
#                  [0.04460842,0.0460842,0.06009143,0.062009143]
#                        ]
# for 7kV


# catching it at the plates with peak voltage #april 15

### Functions for automated wafer position by batch running
markedpositions = []
markedpositionsenergies = []


def autoinit():  # AUTORUN METHOD
    rj = readjson()
    global positionArray
    waferposloaded = rj["rf_gaps"]
    positionArray = waferposloaded
    global markedpositions
    markedpositions = rj["markedpositions"]
    #
    print(f"marked positions {markedpositions}")
    writejson("rf_voltage", Vmax)
    writejson("bunch_length", L_bunch)
    writejson("ekininit", ekininit)
    writejson("freq", freq)
    writejson("tstep", warpoptions.options.timestep)
    writejson("rfgaps_ideal", calculateRFwaferpositions())
    if "beamsavepositions" in rj.keys():
        global storebeam
        storebeam = rj["beamsavepositions"]


def autosave(se):  # AUTORUN METHOD
    # print(f'marked positions {markedpositions}')
    """se : selected Ions"""
    if warpoptions.options.autorun:
        if se.getz().max() > markedpositions[0]:
            print(f"STORING BEAM at {se.getz().max()} ")
            ekinmax = se.getke().max()
            ekinav = se.getke().mean()
            markedpositionsenergies.append({"ekinmax": ekinmax, "ekinav": ekinav})
            writejson("markedpositionsenergies", markedpositionsenergies)
            print("markedpositionsenergies stored")
            del markedpositions[0]
            if len(markedpositions) == 0:
                print("ENDING SIM")
                return True  # cancels the entire simulation loop
    return False


if warpoptions.options.autorun:
    autoinit()

# this needs to be called after autoinit
if storebeam != []:
    if type(storebeam) == str:
        import ast

        res = ast.literal_eval(storebeam)
        storebeam = res
    print(f"STOREBEAM {storebeam}")
    storebeam.sort()
    storebeam.reverse()


def beamsave():
    if storebeam != []:
        if selectedIons.getz().mean() >= storebeam[-1]:
            sbpos = storebeam.pop()
            print(f"STORING BEAM AT POSTION {sbpos}")
            # [{'x':..,'y'...,'vz'...}, {}] -> array of dictionaries
            sb = {
                "x": selectedIons.getx().tolist(),
                "y": selectedIons.gety().tolist(),
                "z": selectedIons.getz().tolist(),
                "vx": selectedIons.getvx().tolist(),
                "vy": selectedIons.getvy().tolist(),
                "vz": selectedIons.getvz().tolist(),
                "t": wp.top.time,
                "framecenter": selectedIons.getz().mean(),
                "storemarker": sbpos,
            }
            rj = readjson()
            if "storedbeams" not in rj.keys():
                writejson("storedbeams", [])
                rj = readjson()
            storedbeams = rj["storedbeams"]
            storedbeams.append(sb)
            writejson("storedbeams", storedbeams)

for i, pa in enumerate(positionArray):
    print(f"Unit {i} placed at {pa}")

# Voltages for each RF UNIT
# setting frequency overwrites the default/waroptions
# frequency setting;

# add actual stack
conductors = [] #RF_stack(positionArray, voltages)

print("CONDUCT DONE")
rfv = gen_volt(toffset=RF_offset, frequency=14.8e6)
###add ctual stack
conductors = [] #RF_stack(positionArray, voltages)
print("CONDUCT DONE")

#add esqs
d_wafers = 1.795 * wp.mm
t_wafer = 1.59 * wp.mm

V_esq =[  90.9, -227.3,  227.3, -140.9,   90.9,  -72.7,   63.6,  -54.5,
         54.5,  -54.5,   50. ,  -54.5,   59.1,  -54.5,   50. ,  -50. ,
         54.5,  -59.1,   59.1]
d_drift = 1e-3
d_esq =1.59e-3 
d_beginning=0
V_total = np.zeros(len(V_esq)*2+1)
d_total=np.zeros(len(V_esq)*2+1)
esq_positions=np.zeros(len(V_esq))
k=0
for i in range(len(V_esq)*2+1):
	if i==0:
    		d_total[i]=d_drift+d_beginning
    		V_total[i]=0
	elif i%2 == 0:
    		d_total[i]=d_total[i-1]+d_drift
    		V_total[i]=0
	else:
    		d_total[i]=d_total[i-1]+d_esq
    		esq_positions[k]=d_total[i]
    		k=k+1
    		V_total[i]=V_esq[int((i-1)/2)]
voltages=np.zeros(len(V_esq))
invertPolarity=np.zeros(len(V_esq))
for i in range(len(V_esq)):
    voltages[i]=abs(V_esq[i])
    if (V_esq[i]<0):
        invertPolarity[i]=-1
    else: 
        invertPolarity[i]=1
        
        



esq0 = ESQ_singlet(esq_positions[0], invertPolarity[0], voltages[0])
esq1 = ESQ_singlet(esq_positions[1], invertPolarity[1], voltages[1])
esq2 = ESQ_singlet(esq_positions[2], invertPolarity[2], voltages[2])
esq3 = ESQ_singlet(esq_positions[3], invertPolarity[3], voltages[3])
esq4 = ESQ_singlet(esq_positions[4], invertPolarity[4], voltages[4])
esq5 = ESQ_singlet(esq_positions[5], invertPolarity[5], voltages[5])
esq6 = ESQ_singlet(esq_positions[6], invertPolarity[6], voltages[6])
esq7 = ESQ_singlet(esq_positions[7], invertPolarity[7], voltages[7])
esq8 = ESQ_singlet(esq_positions[8], invertPolarity[8], voltages[8])
esq9 = ESQ_singlet(esq_positions[9], invertPolarity[9], voltages[9])
esq10 = ESQ_singlet(esq_positions[10], invertPolarity[10], voltages[10])
esq11 = ESQ_singlet(esq_positions[11], invertPolarity[11], voltages[11])
esq12 = ESQ_singlet(esq_positions[12], invertPolarity[12], voltages[12])
esq13 = ESQ_singlet(esq_positions[13], invertPolarity[13], voltages[13])
esq14 = ESQ_singlet(esq_positions[14], invertPolarity[14], voltages[14])
esq15 = ESQ_singlet(esq_positions[15], invertPolarity[15], voltages[15])
esq16 = ESQ_singlet(esq_positions[16], invertPolarity[16], voltages[16])
esq17 = ESQ_singlet(esq_positions[17], invertPolarity[17], voltages[17])
esq18 = ESQ_singlet(esq_positions[18], invertPolarity[18], voltages[18])
"""
esq19 = ESQ_singlet(esq_positions[19], invertPolarity[19], voltages[19])
esq20 = ESQ_singlet(esq_positions[20], invertPolarity[20], voltages[20])
esq21 = ESQ_singlet(esq_positions[21], invertPolarity[21], voltages[21])
esq22 = ESQ_singlet(esq_positions[22], invertPolarity[22], voltages[22])
esq23 = ESQ_singlet(esq_positions[23], invertPolarity[23], voltages[23])
esq24 = ESQ_singlet(esq_positions[24], invertPolarity[24], voltages[24])
esq25 = ESQ_singlet(esq_positions[25], invertPolarity[25], voltages[25])
esq26 = ESQ_singlet(esq_positions[26], invertPolarity[26], voltages[26])
esq27 = ESQ_singlet(esq_positions[27], invertPolarity[27], voltages[27])
esq28 = ESQ_singlet(esq_positions[28], invertPolarity[28], voltages[28])
"""
#conductors=[]
#for i in range(len(V_esq)):
#	esq=ESQ_singlet(esq_positions[i], invertPolarity[i], voltages[i])
#	conductors=conductors +[esq]

conductors=esq0+esq1+esq2+esq3+esq4+esq5+esq6+esq7+esq8+esq9+esq10+esq11+esq12+esq13+esq14+esq15+esq16+esq17+esq18#+esq19+esq20+esq21+esq22+esq23+esq24+esq25+esq26+esq27+esq28

wp.installconductors(conductors)
wp.fieldsol(-1)



# creat submesh for ESQ
meshes = []
for esq_pos in esq_positions:

    solver.root.finalized = 0
    child_1 = solver.addchild(
        mins=[wp.w3d.xmmin, wp.w3d.ymmin, esq_pos - d_wafers / 2 - t_wafer * 4 / 7,],
        maxs=[wp.w3d.xmmax, wp.w3d.ymmax, esq_pos - d_wafers / 2 + t_wafer * 4 / 7,],
        refinement=[1, 1, 5],
    )
    child_2 = solver.addchild(
        mins=[wp.w3d.xmmin, wp.w3d.ymmin, esq_pos + d_wafers / 2 - t_wafer * 4 / 7,],
        maxs=[wp.w3d.xmmax, wp.w3d.ymmax, esq_pos + d_wafers / 2 + t_wafer * 4 / 7,],
        refinement=[1, 1, 5],
    )
    meshes.append(child_1)
    meshes.append(child_2)
 
velo = np.sqrt(
    2 * ekininit * selectedIons.charge / selectedIons.mass
)  # used to calculate tmax
length = esq_positions[-1] + 2 * wp.mm
tmax = length / velo  # this is used for the maximum time for timesteps
zrunmax = length  # this is used for the maximum distance for timesteps
if warpoptions.options.runtime:
    tmax = warpoptions.options.runtime

# Install conductors
wp.installconductors(conductors)

# --- Recalculate the fields
wp.fieldsol(-1)

solver.gridmode = 0  # makes the fields oscillate properly at the beginning

#############################
# def saveBeamSnapshot(z):
#     if z_snapshots:  # checks for remaining snapshots
#         nextZ = min(z_snapshots)
#         if z > nextZ:
#             # this is an approximation and in keV
#             avEkin = (
#                 np.square(selectedIons.getvz()).mean()
#                 * 0.5
#                 * warpoptions.options.speciesMass
#                 * wp.amu
#                 / wp.echarge
#                 / 1000
#             )
#             json_Zsnap = {
#                 # "ekin" : avEkin,
#                 "z_snap_pos": nextZ - lastWafer,
#                 "x": selectedIons.getx().tolist(),
#                 "y": selectedIons.gety().tolist(),
#                 "z": selectedIons.getz().tolist(),
#                 "vx": selectedIons.getvx().tolist(),
#                 "vy": selectedIons.getvy().tolist(),
#                 "vz": selectedIons.getvz().tolist(),
#             }
#             with open(
#                 f"{step1path}/{cgm_name}_snap_"
#                 f"{(nextZ - lastWafer) / wp.mm:.2f}"
#                 f"mm.json",
#                 "w",
#             ) as write_file:
#                 json.dump(json_Zsnap, write_file, indent=2)
#             print(
#                 f"Particle snapshot created at "
#                 f"{nextZ} with mean Ekin"
#                 f" {avEkin}keV"
#             )
#             z_snapshots.remove(nextZ)


### track particles after crossing a Z location -
zc_pos = warpoptions.options.zcrossing
if zc_pos:
    print(f"recording particles crossing at z = {zc_pos}")
    zc = ZCrossingParticles(zz=zc_pos, laccumulate=1)
    zc_start_position = 0.1e-3
    zc_start = ZCrossingParticles(zz=zc_start_position, laccumulate=1)


def savezcrossing():
    if zc_pos:
        zc_data = {
            "x": zc.getx().tolist(),
            "y": zc.gety().tolist(),
            "z": zc_pos,
            "vx": zc.getvx().tolist(),
            "vy": zc.getvy().tolist(),
            "vz": zc.getvz().tolist(),
            "t": zc.gett().tolist(),
            "tbirth": zc.tbirth.tolist(),
        }
        writejson("zcrossing", zc_data)
        zc_start_data = {
            "x": zc_start.getx().tolist(),
            "y": zc_start.gety().tolist(),
            "z": zc_start_position,
            "vx": zc_start.getvx().tolist(),
            "vy": zc_start.getvy().tolist(),
            "vz": zc_start.getvz().tolist(),
            "t": zc_start.gett().tolist(),
            "tbirth": zc_start.tbirth.tolist(),
        }
        writejson("zcrossing_start", zc_data)
        print("STORED Z CROSSING")


#############################

# I want contour plots for levels between 0 and 1kV
# contours = range(0, int(Vesq), int(Vesq/10))

wp.winon(xon=0)

# Plots conductors and contours of electrostatic potential in the Z-X plane

wp.pfzx(fill=1, filled=1, plotphi=0)  # does not plot contours of potential
wp.fma()  # first frame in cgm file

wp.pfzx(fill=1, filled=1, plotphi=1)  # plots contours of potential
wp.fma()  # second frame in cgm file

zmin = wp.w3d.zmmin
zmax = wp.w3d.zmmax
zmid = 0.5 * (zmax + zmin)

# make a circle to show the beam pipe
R = 0.5 * wp.mm  # beam radius
t = np.linspace(0, 2 * np.pi, 100)
X = R * np.sin(t)
Y = R * np.cos(t)
deltaKE = 10e3
time_time = []
numsel = []
KE_select = []
KE_select_Max = []  # modified 4/16
RMS = []

Particle_Counts_Above_E = []  # modified 4/17,
# will list how much Particles have higher E than avg KE at that moment
beamwidth = []
energy_time = []
starting_particles = []
Z = [0]
scraper = wp.ParticleScraper(
    conductors, lcollectlpdata=True
)  # to use until target is fixed to output data properly

# attempt at graphing
lastWafer = esq_positions[-1]
zEnd = 10 * wp.mm + lastWafer
if warpoptions.options.runtime:
    zEnd = 1e3
    print(f"Simulation runs until Z = {zEnd}")


def plotf(axes, component, new_page=True):
    if axes not in ["xy", "zx", "zy"]:
        print("error!!!! wrong axes input!!")
        return

    if component not in ["x", "y", "z", "E"]:
        print("Error! Wrong component declared!!!")
        return

    if axes == "xy":
        plotfunc = wp.pfxy
    elif axes == "zy":
        plotfunc = wp.pfzy
    elif axes == "zx":
        plotfunc = wp.pfzx

    plotfunc(
        fill=1,
        filled=1,
        plotselfe=True,
        comp=component,
        titles=0,
        cmin=-1.2 * Vmax / geometry.gapGNDRF,
        cmax=1.2 * Vmax / geometry.gapGNDRF,
    )  # Vmax/geometry.RF_gap

    if component == "E":
        wp.ptitles(axes, "plot of magnitude of field")
    elif component == "x":
        wp.ptitles(axes, "plot of E_x component of field")
    elif component == "y":
        wp.ptitles(axes, " plot of E_y component of field")
    elif component == "z":
        wp.ptitles(axes, "plot of E_z component of field")

    if new_page:
        wp.fma()


axes1 = "xy"
axes2 = "zx"
# axes3 = 'zy'
component1 = "x"
component2 = "z"
# component3 = 'y'
# magnitude = 'E'
# plotf(axes1,component1)
# plotf(axes2,component2)
# plotf(axes3,component3)
if warpoptions.options.loadbeam == "":  # workaround
    wp.step(1)  # This is needed, so that selectedIons exists

while wp.top.time < tmax and max(Z) < zEnd:
    beamsave()

    ### Informations
    print(
        f"first Particle at {max(Z)*1e3}mm; simulations stops at {zEnd*1e3}mm; == {max(Z)/zEnd*100:.2f}%"
    )
    print(
        f"simulation runs for {wp.top.time*1e9:.2f}ns; stops at {tmax*1e9:.2f}ns == {wp.top.time/tmax*100:.3f}%"
    )
    print(f"Number of particles : {len(selectedIons.getz())}")
    ###### Collecting data
    ### collecting data for Particle count vs Time Plot
    time_time.append(wp.top.time)
    numsel.append(len(selectedIons.getke()))
    RMS.append(rrms())
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
        if KE_i > Avg_KE:  # checks to see if KE of particle is greater than avg KE
            Particle_Count_Over_Avg_KE += 1  # adds to count of particles above avg KE
    Particle_Counts_Above_E.append(Particle_Count_Over_Avg_KE)
    # particles in beam plot
    # accounts for all particles at that moment in time

    wp.top.pline1 = "V_RF: {:.0f}".format(
        gen_volt(RF_offset)(wp.top.time)
    )  # Move this where it belongs
    ###### Injection
    if 0 * wp.ns < wp.top.time < L_bunch and loadbeam == "":  # changes the beam
        wp.top.finject[0, selectedIons.jslist[0]] = 1
    elif (
        not warpoptions.options.cb_framewidth
    ):  # only disable injection if not continuously
        wp.top.inject = 0

    ###### Moving the frame dependent on particle position
    Z = selectedIons.getz()
    if (
        Z.mean() > zmid
    ):  # if the mean distance the particles have travelled is greater than the middle of the frame do this: MODIFIED 4/15
        # the velocity of the frame is equal to the mean velocity of the ions
        wp.top.vbeamfrm = selectedIons.getvz().mean()
        # wp.top.vbeamfrm = selectedIons.getvz().max()
        solver.gridmode = 0  # ToDo Test if this is needed
    # Todo is this needed? wp.top.zbeam is always zero
    zmin = wp.top.zbeam + wp.w3d.zmmin
    zmax = (
        wp.top.zbeam + wp.w3d.zmmax
    )  # trying to get rid of extra length at the end of the simulation, this is wasting computing power
    # wp.top.zbeam+wp.w3d.zmmax #scales the window length #redefines the end of the simulation tacks on the 53mm

    ###### Plotting
    ### Frame, showing
    wp.pfxy(
        fill=1,
        filled=1,
        plotselfe=True,
        comp="x",
        # added on 4/2 by Carlos
        titles=0,
    )
    wp.ptitles(f"xy plot of E_x", f"z mean: {Z.mean()}", "x", "y")
    wp.fma()
    ### Frame, showing
    wp.pfxy(fill=1, filled=1, plotselfe=True, comp="z", titles=0)
    wp.ptitles(f"xy plot of E_z", f"z mean: {Z.mean()}", "x", "y")
    wp.fma()
    #
    # plotf(axes1,component1, z mean: {Z.mean()}) # Todo Carlos, can this be removed?

    ### Frame, the instantaneous kinetic energy plot
    KE = selectedIons.getke()
    print(f"Mean kinetic Energy : {np.mean(KE)}")
    if len(KE) > 0:
        selectedIons.ppzke(color=wp.blue)
        KEmin, KEmax = KE.min(), KE.max()
        while KEmax - KEmin > deltaKE:
            deltaKE += 10e3
    wp.ylimits(
        0.95 * KEmin, 0.95 * KEmin + deltaKE
    )  # is this fraction supposed to match with V_arrival?
    wp.fma()
    ### Frame the side view field plot
    # plotf(axes2,component2, z mean: {Z.mean()})
    # wp.fma()
    ### Frame, showing
    plotf("xy", "E", 1)
    ### Frame, showing
    plotf("xy", "z", 1)
    ### Frame, showing the side view plot of Ez and the electrical components
    wp.pfzx(
        fill=1,
        filled=1,
        plotselfe=True,
        comp="z",
        titles=0,
        cmin=-1.2 * Vmax / geometry.gapGNDRF,
        cmax=1.2 * Vmax / geometry.gapGNDRF,
    )
    # sort by birthtime
    t_birth_min = selectedIons.tbirth.min()
    t_birth_max = selectedIons.tbirth.max()
    tarray = np.linspace(t_birth_min, t_birth_max, 6)
    # mask to sort particles by birthtime
    mask = []
    for i in range(
        len(tarray) - 1
    ):  # the length of tarray must be changing overtime which changes the mask which recolors the particles
        m = (selectedIons.tbirth > tarray[i]) * (selectedIons.tbirth < tarray[i + 1])
        mask.append(m)
        # plot particles on top of fild plot, sort by birthtime and color them accordingly
    colors = [wp.red, wp.yellow, wp.green, wp.blue, wp.magenta]
    for m, c in zip(mask, colors):
        if loadbeam == "":
            wp.plp(
                selectedIons.getx()[m], selectedIons.getz()[m], msize=1.0, color=c
            )  # the selected ions are changing through time
        else:
            wp.plp(selectedIons.getx(), selectedIons.getz(), msize=1.0)
    wp.limits(zmin, zmax)
    wp.ptitles(
        f"Particles and Fields \ntime {wp.top.time}\n voltage : {rfv(wp.top.time)}",
        "Z [m]",
        "X [m]",
    )
    wp.fma()
    
        ## adding Y:
    
    
    
    mask = []
    for i in range(
        len(tarray) - 1
    ):  # the length of tarray must be changing overtime which changes the mask which recolors the particles
        m = (selectedIons.tbirth > tarray[i]) * (selectedIons.tbirth < tarray[i + 1])
        mask.append(m)
        # plot particles on top of fild plot, sort by birthtime and color them accordingly
    colors = [wp.red, wp.yellow, wp.green, wp.blue, wp.magenta]
    for m, c in zip(mask, colors):
        if loadbeam == "":
            wp.plp(
                selectedIons.gety()[m], selectedIons.getz()[m], msize=1.0, color=c
            )  # the selected ions are changing through time
        else:
            wp.plp(selectedIons.gety(), selectedIons.getz(), msize=1.0)
    wp.limits(zmin, zmax)
    wp.ptitles(
        f"Particles and Fields \ntime {wp.top.time}\n voltage : {rfv(wp.top.time)}",
        "Z [m]",
        "Y [m]",
    )
    wp.fma()
    
    
    ### Frame, shwoing
    selectedIons.ppxy(
        color=wp.red
    )  # ppxy (particle plot x horizontal axis, y on vertical axis
    wp.limits(-R*2, R*2)
    wp.ylimits(-R*2, R*2)
    wp.plg(Y, X, type="dash")
    wp.fma()
    ###
    
    
    
    wp.pzepsx()
    wp.fma()

    wp.pzepsnx()
    wp.fma()

    wp.pzepsny()
    wp.fma()
    wp.pfzx(fill=1, filled=1, plotphi=1)  # plots contours of potential
    wp.fma()  # second frame in cgm file

    # autosave(selectedIons)
    if autosave(selectedIons):
        print(f"Postion {selectedIons.getz().max()}")
        break
    ### check if a snapshot should be taken for export for the energy analyzer
    # saveBeamSnapshot(Z.mean())
    if warpoptions.options.cb_framewidth:  # if continous beam, do steps normally
        wp.step(warpoptions.options.plotsteps)
    elif wp.top.inject == 0:  # if there is no injection, do steps normally
        wp.step(warpoptions.options.plotsteps)
    else:  # if there is injection going on, make timesteps as fine as possible so that the bunch has the right length
        wp.step(1)

### END of Simulation
savezcrossing()

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
f_survive = [i / j for i, j in zip(numsel, starting_particles)]
# want the particles that just make it through the last RF, need position of RF. This way we can see how many particles made it through the last important component of the accelerator

wp.plg(f_survive, time_time, color=wp.green)
wp.ptitles(
    "Fraction of Surviving Particles vs Time",
    "Time (s)",
    "Fraction of Surviving Particles",
)
wp.fma()


wp.pzepsx()
wp.fma()

wp.pzepsnx()
wp.fma()

wp.pzepsny()
wp.fma()
wp.pzepsnz()
wp.fma()

wp.hpenvx(color=wp.red, titles=0)
#wp.fma()

wp.hpenvy(color=wp.blue, titles=0)
wp.ptitles("X(red), Y(blue)", "Time [s]", "X, Y [m]", "")
wp.fma()


### Frame, rms envelope plot
wp.hpxrms(color=wp.red, titles=0)
wp.hpyrms(color=wp.blue, titles=0)
wp.hprrms(color=wp.green, titles=0)
wp.ptitles("X(red), Y(blue), R(green)", "Time [s]", "X/Y/R [m]", "")
wp.fma()
### Frame, rms envelope plot
wp.pzenvx(color=wp.red, titles=0)
wp.pzenvy(color=wp.blue, titles=0)
wp.ptitles("X(red), Y(blue)", "Z [m]", "X/Y [m]", "")
wp.fma()
### Frame, vx and vy plot
wp.hpvxbar(color=wp.red)
wp.hpvybar(color=wp.blue)
#wp.ptitles("X(red), Y(blue), R(green)", "Time [s]", "X/Y/R [m]", "")
wp.fma()

#vz plot
wp.hpvzbar(color=wp.blue)
wp.fma()


### Frame, Kinetic Energy at certain Z value
wp.plg(KE_select, time_time, color=wp.blue)
wp.limits(0, 70e-9, 0, 30e3)  # limits(xmin,xmax,ymin,ymax)
wp.ptitles("Kinetic Energy vs Time")
wp.fma()
# ### Frame, rms and kinetic energy vs time
# fig, ax1 = plt.subplots()

# color = "tab:red"
# ax1.set_xlabel("time [s]")
# ax1.set_ylabel("RMS", color=color)
# ax1.plot(time_time, RMS, color=color)
# ax1.tick_params(axis="y", labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = "tab:blue"
# ax2.set_ylabel("Kinetic Energy [eV]", color=color)
# ax2.plot(time_time, KE_select, color=color)
# ax2.tick_params(axis="y", labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
# plt.savefig("RMS & E vs Time.png")
### Frame, maximal kinetic energy at certain Z value
wp.plg(KE_select_Max, time_time, color=wp.blue)
wp.limits(0, 70e-9, 0, 30e3)  # limits(xmin,xmax,ymin,ymax)
wp.ptitles(" Maximal Kinetic Energy vs Time")
wp.fma()
# kinetic energy plot
wp.plg(KE_select, time_time, color=wp.blue)
wp.ptitles("kinetic energy vs time")
wp.fma()
### kinetic energy plot
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

### Data storage
# save history information, so that we can plot all cells in one plot

t = np.trim_zeros(wp.top.thist, "b")
hepsny = selectedIons.hepsny[0]
hepsnz = selectedIons.hepsnz[0]
hep6d = selectedIons.hepsx[0] * selectedIons.hepsy[0] * selectedIons.hepsz[0]
hekinz = 1e-6 * 0.5 * wp.top.aion * wp.amu * selectedIons.hvzbar[0] ** 2 / wp.jperev
u = (
    selectedIons.hvxbar[0] ** 2
    + selectedIons.hvybar[0] ** 2
    + selectedIons.hvzbar[0] ** 2
)
hekin = 1e-6 * 0.5 * wp.top.aion * wp.amu * u / wp.jperev
hxrms = selectedIons.hxrms[0]
hyrms = selectedIons.hyrms[0]
hrrms = selectedIons.hrrms[0]

hpnum = selectedIons.hpnum[0]

print("debug", t.shape, hepsny.shape)
out = np.stack((t, hepsny, hepsnz, hep6d, hekinz, hekin, hxrms, hyrms, hrrms, hpnum))
outhxrms=hxrms
print(outhxrms)

outtime=t
print(outtime)
print(velo)

df=pd.DataFrame()
df['t']=t
df['xrms']=hxrms
df['yrms']=hyrms
df['hepsny']=hepsny
df['hepsnz']=hepsnz
df['hep6d']=hep6d
df['hekinz']=hekinz
df['hekin']=hekin
df['hrrms']=hrrms
df['hpnum']=hpnum


csv_name=dt_string+".csv"
df.to_csv(csv_name)
print("V_esq: ")
print(V_esq)
print("ESQ positions: ")
print(esq_positions)
print("Z velocity: ")
print(velo)

rt = (time.time() - start) / 60
print(f"RUNTIME OF THIS SIMULATION: {rt:.0f} minutes")
if warpoptions.options.autorun:
    writejson("runtimeminutes", rt)

### END BELOW HERE IS CODE THAT MIGHT BE USEFUL LATER

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
