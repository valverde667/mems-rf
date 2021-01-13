import warpoptions

#   only specify the parameters you want to change from their default values
#   the input will look like:
"""
python3 single-species-simulation.py --esq_voltage=500 --fraction=.8 --speciesMass=20 --ekininit=15e3
"""

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
    "--rf_voltage", dest="Vmax", type=float, default="7000"
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
    "--ekininit", dest="ekininit", type=float, default="10e3"
)

#   frequency of the RF
warpoptions.parser.add_argument(
    "--freq", dest="freq", type=float, default="14.8e6"
)  # 27e6

#   emitting radius
warpoptions.parser.add_argument(
    "--emit", dest="emittingRadius", type=float, default=".25e-3"
)  # .37e-3 #.25

#   divergence angle
warpoptions.parser.add_argument(
    "--diva", dest="divergenceAngle", type=float, default="5e-3"
)

#   special cgm name - for mass output / scripting
warpoptions.parser.add_argument("--name", dest="name", type=str, default="multions")

#   special cgm path - for mass output / scripting
warpoptions.parser.add_argument("--path", dest="path", type=str, default="")

#   divergence angle
warpoptions.parser.add_argument(
    "--tstep", dest="timestep", type=float, default="1e-10"
)  # 1e-11

#   Volt ratio for ESQs @ToDo Zhihao : is this correct?
warpoptions.parser.add_argument(
    "--volt_ratio", dest="volt_ratio", type=float, default="1.04"
)

#   enables some additional code if True
warpoptions.parser.add_argument("--autorun", dest="autorun", type=bool, default=False)

# sets wp.steps(#)
warpoptions.parser.add_argument("--plotsteps", dest="plotsteps", type=int, default=20)

# changes simulation to a "cb-beam" simulation
warpoptions.parser.add_argument("--cb", dest="cb_framewidth", type=float, default=0)

# enables a Z-Crossing location, saving particles that are crossing the given z-value
warpoptions.parser.add_argument("--zcrossing", dest="zcrossing", type=float, default=0)

# set maximal running time, this disables other means of ending the simulation
warpoptions.parser.add_argument("--runtime", dest="runtime", type=float, default=0)

# beam current initial beam current may vary up to 25e-6
warpoptions.parser.add_argument(
    "--ibeaminit", dest="ibeaminit", type=float, default=10e-6
)

# offset to make beam bunch asynchronous
warpoptions.parser.add_argument(
    "--beamdelay", dest="beamdelay", type=float, default=0.0
)

#   stores the beam at once the average particle position passed the given positions (list)
# --storebeam "[0.01, 0.024, 0.045]" uses the --name given for the simulation. Stored beams are ordered.
# Needs to be an array with stings
warpoptions.parser.add_argument("--storebeam", dest="storebeam", default="[]")
# --loadbeam "path/to/file.json"
warpoptions.parser.add_argument("--loadbeam", dest="loadbeam", type=str, default="")
# --beamnumber 3  or negative number to count from the back. Stored beams are ordered.
warpoptions.parser.add_argument("--beamnumber", dest="beamnumber", type=int, default=-1)

# --Python packages
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
import pdb

# --Import third-party packages
import warp as wp
from warp.particles.extpart import ZCrossingParticles

# --Import custom packages
import geometry
from geometry import RF_stack, ESQ_doublet

start = time.time()

# Utility definitions
name = warpoptions.options.name
beamnumber = warpoptions.options.beamnumber

# --- where to store the outputfiles
cgm_name = name
step1path = "."
# step1path = os.getcwd()

# overwrite if path is given by command
if warpoptions.options.path != "":
    step1path = warpoptions.options.path

wp.setup(prefix=f"{step1path}/{cgm_name}")  # , cgmlog= 0)

### read / write functionality #ToDo: move into helper file
basepath = warpoptions.options.path
if basepath == "":
    basepath = f"{step1path}/"
thisrunID = warpoptions.options.name

# Utility Functions
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


# Initialize input variables
L_bunch = warpoptions.options.Lbunch
Units = warpoptions.options.Units
Vmax = warpoptions.options.Vmax  # RF voltage
Vesq = warpoptions.options.Vesq
V_arrival = warpoptions.options.V_arrival  # fraction total voltage gained each gap
ekininit = warpoptions.options.ekininit
freq = warpoptions.options.freq  # RF freq
emittingRadius = warpoptions.options.emittingRadius
divergenceAngle = warpoptions.options.divergenceAngle
storebeam = warpoptions.options.storebeam
loadbeam = warpoptions.options.loadbeam
ibeaminit = warpoptions.options.ibeaminit
beamdelay = warpoptions.options.beamdelay


# Specify  simulation mesh
wp.w3d.solvergeom = wp.w3d.XYZgeom
wp.w3d.xmmin = -3 / 2 * wp.mm
wp.w3d.xmmax = 3 / 2 * wp.mm
wp.w3d.ymmin = -3 / 2 * wp.mm
wp.w3d.ymmax = 3 / 2 * wp.mm
framewidth = 23 * wp.mm
wp.w3d.zmmin = 0.0
wp.w3d.zmmax = wp.w3d.zmmin + framewidth
wp.top.dt = warpoptions.options.timestep
wp.w3d.nx = 30  # 60.
wp.w3d.ny = 30  # 60.
wp.w3d.nz = 180.0

# Set boundary conditions
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.neumann
wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb
wp.top.prwall = 1 * wp.mm

# Create Species
selectedIons = wp.Species(type=wp.Nitrogen, charge_state=1, name="N+", color=wp.blue,)
ions = wp.Species(type=wp.Dinitrogen, charge_state=1, name="N2+", color=wp.red,)

# keep track of when the particles are born
wp.top.ssnpid = wp.nextpid()
wp.top.tbirthpid = wp.nextpid()

writejson("bunchlength", L_bunch)
writejson("Vmax", Vmax)
writejson("Vesq", Vesq)
writejson("ekininit", ekininit)
writejson("frequency", freq)
writejson("emitting_Radus", emittingRadius)
writejson("divergance_Angle", divergenceAngle)
writejson("name", name)
writejson("beamnumber", beamnumber)
writejson("ibeaminit", ibeaminit)
writejson("beamdelay", beamdelay)
writejson("tstep", warpoptions.options.timestep)

# Set Injection Parameters for injector and beam
wp.top.ns = 2
wp.top.npmax = 5  # maximal number of particles (for injection per timestep???)
wp.top.ns = 2  # numper of species
# wp.top.np_s = [5, 2]
wp.top.inject = 1  # Constant current injection
wp.top.npinje_s = [1, 1]  # Number of particles injected per step by species
wp.top.ainject = emittingRadius
wp.top.binject = emittingRadius
wp.top.apinject = divergenceAngle
wp.top.bpinject = divergenceAngle
wp.top.vinject = 1.0  # source voltage

wp.top.ibeam_s = [ibeaminit, ibeaminit]
wp.top.ekin_s = [ekininit, ekininit]
wp.derivqty()

# Setup Histories and moment calculations
wp.top.lspeciesmoments = True
wp.top.nhist = 1  # Save history data every N time step
wp.top.lhpnumz = True
wp.top.lhcurrz = True
wp.top.lhrrmsz = True
wp.top.lhxrmsz = True
wp.top.lhyrmsz = True
wp.top.lhepsnxz = True
wp.top.lhepsnyz = True
wp.top.lhvzrmsz = True

# Set up fieldsolver
solver = wp.MRBlock3D()
wp.registersolver(solver)
solver.mgtol = 1.0  # Poisson solver tolerance, in volts
solver.mgparam = 1.5
solver.downpasses = 2
solver.uppasses = 2

# Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.)
wp.package("w3d")
wp.generate()
x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh
ESQs = []
RFs = []
ID_ESQ = 100
ID_RF = 201
ID_target = 1

# Conductor utility functions
def gen_volt(toffset=0, frequency=freq):  # 0
    """ A cos voltage function with variable offset"""

    def RFvoltage(time):
        return -Vmax * np.cos(2 * np.pi * frequency * (time + toffset))

    return RFvoltage


def gen_volt_esq(Vesq, inverse=False, toffset=0):
    def ESQvoltage(time):
        if inverse:
            return -Vesq  # *np.sin(2*np.pi*freq*(time+toffset))
        else:
            return Vesq  # *np.sin(2*np.pi*freq*(time+toffset))

    return ESQvoltage


# Calculate the distances and time offset for the RFs
centerOfFirstRFGap = 5 * wp.mm
tParticlesAtCenterFirstGap = centerOfFirstRFGap / wp.sqrt(
    2 * ekininit * selectedIons.charge / selectedIons.mass
)
print(
    f"Particles need {tParticlesAtCenterFirstGap * 1e6}us"
    f" to reach the center of the first gap"
)
# RF should be maximal when particles arrive
# time for the cos to travel there minus the time the
# particles take
RF_offset = centerOfFirstRFGap / wp.clight - tParticlesAtCenterFirstGap - beamdelay
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
                / selectedIons.mass
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
                / selectedIons.mass
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
positionArray = [[0.001, 0.003, 0.007, 0.009], [0.015, 0.017, 0.021, 0.023]]
writejson("waferpositions", positionArray)

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
voltages = [gen_volt(toffset=RF_offset, frequency=14.8e6)]
# add actual stack
conductors = RF_stack(positionArray, voltages)
print("CONDUCT DONE")

velo = np.sqrt(
    2 * ekininit * selectedIons.charge / selectedIons.mass
)  # used to calculate tmax
length = positionArray[-1][-1] + 25 * wp.mm
tmax = length / velo  # this is used for the maximum time for timesteps
zrunmax = length  # this is used for the maximum distance for timesteps
period = 1 / freq
tcontrol = period / 2
if warpoptions.options.runtime:
    tmax = warpoptions.options.runtime

# Install conductors
wp.installconductors(conductors)

# Recalculate the fields
wp.fieldsol(-1)
solver.gridmode = 0  # Temporary fix for fields to oscillate in time.
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
        }
        writejson("zcrossing_start", zc_data)
        print("STORED Z CROSSING")


#############################
# zcrossing after every gap
p0 = [x[3] + 1e-3 for x in positionArray]
p1 = [x[1] + 1e-3 for x in positionArray]
zcs_pos = [positionArray[0][0] - 1e-3] + p0 + p1
print(f"Saving positions: {zcs_pos}")
zcs_pos.sort()
zcs_staple = zcs_pos.copy()
zcs_staple.reverse()
zcs = [ZCrossingParticles(zz=sp, laccumulate=1) for sp in zcs_pos]


@wp.callfromafterstep
def allzcrossing():
    if len(zcs_staple):
        if min(selectedIons.getz()) > zcs_staple[-1]:
            zcs_data = []
            zcs_staple.pop()
            for zcc, pos in zip(zcs, zcs_pos):
                zcs_data.append(
                    {
                        "x": zcc.getx().tolist(),
                        "y": zcc.gety().tolist(),
                        "z": pos,
                        "vx": zcc.getvx().tolist(),
                        "vy": zcc.getvy().tolist(),
                        "vz": zcc.getvz().tolist(),
                        "t": zcc.gett().tolist(),
                    }
                )
            writejson("allzcrossing", zcs_data)
            print("Json Saved")


zmid = 0.5 * (z.max() + z.min())
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


if warpoptions.options.loadbeam == "":  # workaround
    wp.step(1)  # This is needed, so that selectedIons exists

# Create cgm windows for plotting
wp.winon(winnum=2, suffix="pzx", xon=False)
wp.winon(winnum=3, suffix="pxy", xon=False)
wp.winon(winnum=4, suffix="stats", xon=False)

# Calculate various control values to dictate when the simulation ends
velo = np.sqrt(2 * ekininit * selectedIons.charge / selectedIons.mass)
length = positionArray[-1][-1] + 25 * wp.mm
tmax = length / velo  # this is used for the maximum time for timesteps
zrunmax = length  # this is used for the maximum distance for timesteps
period = 1 / freq
tcontrol = period / 2
if warpoptions.options.runtime:
    tmax = warpoptions.options.runtime


# Main loop. Advance particles and output graphics.
while wp.top.time < tcontrol:
    # Plot particle trajectory in zx
    wp.window(2)
    wp.pfzx(fill=1, filled=1, plotphi=1, cmin=-Vmax, cmax=Vmax, contours=50)
    selectedIons.ppzx(color=wp.blue, msize=10)
    ions.ppzx(color=wp.red, msize=10)
    wp.limits(z.min(), z.max(), x.min(), x.max())
    wp.fma()

    # Plot particle trajectory in xy
    wp.window(3)
    selectedIons.ppxy(color=wp.blue, msize=10)
    ions.ppxy(color=wp.red, msize=10)
    wp.limits(x.min(), x.max(), y.min(), y.max())
    wp.plg(Y, X, type="dash")
    wp.fma()

    wp.step(1)

### END of Simulation
raise Exception()
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
wp.hpvxbar(color=wp.red, titles=0)
wp.hpvybar(color=wp.blue, titles=0)
wp.ptitles("X(red), Y(blue), R(green)", "Time [s]", "X/Y/R [m]", "")
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

rt = (time.time() - start) / 60
print(f"RUNTIME OF THIS SIMULATION: {rt:.0f} minutes")
if warpoptions.options.autorun:
    writejson("runtimeminutes", rt)

### END BELOW HERE IS CODE THAT MIGHT BE USEFUL LATER
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
