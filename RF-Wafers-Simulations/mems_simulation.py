import warpoptions

"""
python3 single-species-simulation.py --esq_voltage=500 --fraction=.8 --speciesMass=20 --ekininit=15e3
"""

#   mass of the ions being accelerated
warpoptions.parser.add_argument(
    "--species_mass", dest="speciesMass", type=int, default="40"
)

#   special cgm name - for mass output / scripting
warpoptions.parser.add_argument("--name", dest="name", type=str, default="multions")

#   special cgm path - for mass output / scripting
warpoptions.parser.add_argument("--path", dest="path", type=str, default="")


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

warpoptions.parser.add_argument(
    "--beamdelay", dest="beamdelay", type=float, default=0.0
)
warpoptions.parser.add_argument("--storebeam", dest="storebeam", default="[]")
warpoptions.parser.add_argument("--loadbeam", dest="loadbeam", type=str, default="")
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

# Define useful constants
mm = 1e-3
um = 1e-6
nm = 1e-9
kV = 1e3
keV = 1e3
ms = 1e-3
us = 1e-6
ns = 1e-9
MHz = 1e6
uA = 1e-6

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


def create_wafer(
    cent,
    width=2.0 * mm,
    cell_width=3.0 * mm,
    length=0.7 * mm,
    rin=0.55 * mm,
    rout=0.75 * mm,
    xcent=0.0 * mm,
    ycent=0.0 * mm,
    voltage=0.0,
):
    """ Create a single wafer

    An acceleration gap will be comprised of two wafers, one grounded and one
    with an RF varying voltage. Creating a single wafer without combining them
    (create_gap function) will allow to place a time variation using Warp that
    one mess up the potential fields."""

    prong_width = rout - rin
    ravg = (rout + rin) / 2

    # Left wafer first.

    # Create box surrounding wafer. The extent is slightly larger than 5mm unit
    # cell. The simulation cell will chop this to be correct so long as the
    # inner box separation is correct (approximately 0.2mm thickness)
    box_out = wp.Box(
        xsize=cell_width,
        ysize=cell_width,
        zsize=length,
        zcent=cent,
        xcent=xcent,
        ycent=ycent,
        voltage=voltage,
    )
    box_in = wp.Box(
        xsize=cell_width - 0.0002,
        ysize=cell_width - 0.0002,
        zsize=length,
        zcent=cent,
        xcent=xcent,
        ycent=ycent,
        voltage=voltage,
        condid=box_out.condid,
    )
    box = box_out - box_in

    annulus = wp.Annulus(
        rmin=rin,
        rmax=rout,
        length=length,
        zcent=cent,
        xcent=xcent,
        ycent=ycent,
        voltage=voltage,
        condid=box.condid,
    )

    # Create prongs. This is done using four box conductors and shifting
    # respective x/y centers to create the prong.
    top_prong = wp.Box(
        xsize=prong_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        zcent=cent,
        xcent=xcent,
        ycent=ycent + (cell_width / 2 + ravg) / 2,
        voltage=voltage,
        condid=box.condid,
    )
    bot_prong = wp.Box(
        xsize=prong_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        zcent=cent,
        xcent=xcent,
        ycent=ycent - (cell_width / 2 + ravg) / 2,
        voltage=voltage,
        condid=box.condid,
    )
    rside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=prong_width,
        zsize=length,
        zcent=cent,
        xcent=xcent + (cell_width / 2 + ravg) / 2,
        ycent=ycent,
        voltage=voltage,
        condid=box.condid,
    )
    lside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=prong_width,
        zsize=length,
        zcent=cent,
        xcent=xcent - (cell_width / 2 + ravg) / 2,
        ycent=ycent,
        voltage=voltage,
        condid=box.condid,
    )

    # Add together
    cond = annulus + box + top_prong + bot_prong + rside_prong + lside_prong

    return cond


def create_gap(
    cent,
    left_volt,
    right_volt,
    width=2.0 * mm,
    cell_width=3.0 * mm,
    length=0.7 * mm,
    rin=0.55 * mm,
    rout=0.75 * mm,
    xcent=0.0 * mm,
    ycent=0.0 * mm,
):
    """Create an acceleration gap consisting of two wafers.

    The wafer consists of a thin annulus with four rods attaching to the conducting
    cell wall. The cell is 5mm where the edge is a conducting square. The annulus
    is approximately 0.2mm in thickness with an inner radius of 0.55mm and outer
    radius of 0.75mm. The top, bottom, and two sides of the annulus are connected
    to the outer conducting box by 4 prongs that are of approximately equal
    thickness to the ring.

    Here, the annuli are created easy enough. The box and prongs are created
    individually for each left/right wafer and then added to give the overall
    conductor.

    Note, this assumes l4 symmetry is turned on. Thus, only one set of prongs needs
    to be created for top/bottom left/right symmetry."""

    prong_width = rout - rin
    ravg = (rout + rin) / 2

    # Left wafer first.
    left_wafer = wp.Annulus(
        rmin=rin,
        rmax=rout,
        length=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent,
        ycent=ycent,
    )

    # Create box surrounding wafer. The extent is slightly larger than 5mm unit
    # cell. The simulation cell will chop this to be correct so long as the
    # inner box separation is correct (approximately 0.2mm thickness)
    l_box_out = wp.Box(
        xsize=cell_width * (1 + 0.02),
        ysize=cell_width * (1 + 0.02),
        zsize=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent,
        ycent=ycent,
    )
    l_box_in = wp.Box(
        xsize=cell_width * (1 - 0.02),
        ysize=cell_width * (1 - 0.02),
        zsize=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent,
        ycent=ycent,
    )
    l_box = l_box_out - l_box_in

    # Create prongs. This is done using four box conductors and shifting
    # respective x/y centers to create the prong.
    l_top_prong = wp.Box(
        xsize=prong_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent,
        ycent=ycent + (cell_width / 2 + ravg) / 2,
    )
    l_bot_prong = wp.Box(
        xsize=prong_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent,
        ycent=ycent - (cell_width / 2 + ravg) / 2,
    )
    l_rside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=prong_width,
        zsize=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent + (cell_width / 2 + ravg) / 2,
        ycent=ycent,
    )
    l_lside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=prong_width,
        zsize=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent - (cell_width / 2 + ravg) / 2,
        ycent=ycent,
    )

    # Add together
    left = (
        left_wafer + l_box + l_top_prong + l_bot_prong + l_rside_prong + l_lside_prong
    )

    right_wafer = wp.Annulus(
        rmin=rin,
        rmax=rout,
        length=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent,
        ycent=ycent,
    )

    r_box_out = wp.Box(
        xsize=cell_width * (1 + 0.02),
        ysize=cell_width * (1 + 0.02),
        zsize=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent,
        ycent=ycent,
    )
    r_box_in = wp.Box(
        xsize=cell_width * (1 - 0.02),
        ysize=cell_width * (1 - 0.02),
        zsize=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent,
        ycent=ycent,
    )
    r_box = r_box_out - r_box_in

    r_top_prong = wp.Box(
        xsize=prong_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent,
        ycent=ycent + (cell_width / 2 + ravg) / 2,
    )
    r_bot_prong = wp.Box(
        xsize=prong_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent,
        ycent=ycent - (cell_width / 2 + ravg) / 2,
    )
    r_rside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=prong_width,
        zsize=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent + (cell_width / 2 + ravg) / 2,
        ycent=ycent,
    )
    r_lside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=prong_width,
        zsize=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent - (cell_width / 2 + ravg) / 2,
        ycent=ycent,
    )
    right = (
        right_wafer + r_box + r_top_prong + r_bot_prong + r_rside_prong + r_lside_prong
    )

    gap = left + right
    return gap


# -------------------------------------------------------------------------------
#    Script inputs
# Parameter inputs for running the script. Initially were set as command line
# arguments. Setting to a designated section for better organization and
# overview. Eventually this will be stripped and turn into an input file.
# -------------------------------------------------------------------------------
L_bunch = 1 * ns
Units = 2
Vmax = 7 * kV
Vesq = 0.1 * kV
V_arrival = 1.0
ekininit = 7 * keV
freq = 13.6 * MHz
emittingRadius = 0.25 * mm
divergenceAngle = 5e-3
ibeaminit = 10 * uA
beamdelay = 0.0

storebeam = warpoptions.options.storebeam
loadbeam = warpoptions.options.loadbeam

first_gapzc = 5 * mm  # First gap center

rf_volt = lambda time: Vmax * np.cos(2.0 * np.pi * freq * time)
# -------------------------------------------------------------------------------
#    Mesh setup
# Specify mesh sizing and time stepping for simulation.
# -------------------------------------------------------------------------------
# Specify  simulation mesh
wp.w3d.xmmax = 3 / 2 * mm
wp.w3d.xmmin = -wp.w3d.xmmax
wp.w3d.ymmax = wp.w3d.xmmax
wp.w3d.ymmin = -wp.w3d.ymmax

framewidth = 10 * mm
wp.w3d.zmmin = 0 * mm
wp.w3d.zmmax = 14 * mm
wp.w3d.nx = 40
wp.w3d.ny = 40
wp.w3d.nz = 200
dz = (wp.w3d.zmmax - wp.w3d.zmmin) / wp.w3d.nz
dt = 0.2 * ns
wp.top.dt = dt

# Set boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.periodic

wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb
wp.top.prwall = 1 * mm

# Create Species
selectedIons = wp.Species(type=wp.Argon, charge_state=1, name="Ar+", color=wp.blue)

# keep track of when the particles are born
wp.top.ssnpid = wp.nextpid()
wp.top.tbirthpid = wp.nextpid()


# Set Injection Parameters for injector and beam
wp.top.ns = 2
wp.top.ns = 2  # numper of species
# wp.top.np_s = [5, 2]
wp.top.inject = 1  # Constant current injection
wp.top.rnpinje_s = [1, 1]  # Number of particles injected per step by species
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
wp.top.itmomnts = wp.top.nhist
wp.top.lhpnumz = True
wp.top.lhcurrz = True
wp.top.lhrrmsz = True
wp.top.lhxrmsz = True
wp.top.lhyrmsz = True
wp.top.lhepsnxz = True
wp.top.lhepsnyz = True
wp.top.lhvzrmsz = True
wp.top.lsavelostpart = True

# Set up fieldsolver
wp.w3d.l4symtry = False
solver = wp.MRBlock3D()
wp.registersolver(solver)
solver.mgtol = 1.0  # Poisson solver tolerance, in volts
solver.mgparam = 1.5
solver.downpasses = 2
solver.uppasses = 2

# Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.)
wp.package("w3d")
wp.generate()
solver.gridmode = 0  # Temporary fix for fields to oscillate in time.
x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh
ESQs = []
RFs = []
ID_ESQ = 100
ID_RF = 201
ID_target = 1


def rrms():
    x_dis = selectedIons.getx()
    y_dis = selectedIons.gety()

    xrms = np.sqrt(np.mean(x_dis ** 2))
    yrms = np.sqrt(np.mean(y_dis ** 2))
    rrms = np.sqrt(np.mean(x_dis ** 2 + y_dis ** 2))

    print(f" XRMS: {xrms} \n YRMS: {yrms} \n RRMS: {rrms}")

    return rrms


positionArray = np.array([3, 8]) * mm

### Functions for automated wafer position by batch running
markedpositions = []
markedpositionsenergies = []

for i, pa in enumerate(positionArray):
    print(f"Unit {i} placed at {pa}")

conductors = []
for i, pos in enumerate(positionArray):
    zl = pos - 1 * mm
    zr = pos + 1 * mm
    if i % 2 == 0:
        this_lcond = create_wafer(zl, voltage=0.0)
        this_rcond = create_wafer(zr, voltage=rf_volt)
    else:
        this_lcond = create_wafer(zl, voltage=rf_volt)
        this_rcond = create_wafer(zr, voltage=0.0)

    conductors.append(this_lcond)
    conductors.append(this_rcond)

for cond in conductors:
    wp.installconductors(cond)

# Recalculate the fields
wp.fieldsol(-1)

zc_pos = True


# def savezcrossing():
#     if zc_pos:
#         zc_data = {
#             "x": zc.getx().tolist(),
#             "y": zc.gety().tolist(),
#             "z": zc_pos,
#             "vx": zc.getvx().tolist(),
#             "vy": zc.getvy().tolist(),
#             "vz": zc.getvz().tolist(),
#             "t": zc.gett().tolist(),
#         }
#         writejson("zcrossing", zc_data)
#         zc_start_data = {
#             "x": zc_start.getx().tolist(),
#             "y": zc_start.gety().tolist(),
#             "z": zc_start_position,
#             "vx": zc_start.getvx().tolist(),
#             "vy": zc_start.getvy().tolist(),
#             "vz": zc_start.getvz().tolist(),
#             "t": zc_start.gett().tolist(),
#         }
#         writejson("zcrossing_start", zc_data)
#         print("STORED Z CROSSING")


#############################


# @wp.callfromafterstep
# def allzcrossing():
#     if len(zcs_staple):
#         if min(selectedIons.getz()) > zcs_staple[-1]:
#             zcs_data = []
#             zcs_staple.pop()
#             for zcc, pos in zip(zcs, zcs_pos):
#                 zcs_data.append(
#                     {
#                         "x": zcc.getx().tolist(),
#                         "y": zcc.gety().tolist(),
#                         "z": pos,
#                         "vx": zcc.getvx().tolist(),
#                         "vy": zcc.getvy().tolist(),
#                         "vz": zcc.getvz().tolist(),
#                         "t": zcc.gett().tolist(),
#                     }
#                 )
#             writejson("allzcrossing", zcs_data)
#             print("Json Saved")


zmid = 0.5 * (z.max() + z.min())

# Make a circle to show the beam pipe on warp plots in xy.
R = 0.5 * mm  # beam radius
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
# wp.winon(winnum=4, suffix="stats", xon=False)

# Calculate various control values to dictate when the simulation ends
velo = np.sqrt(2 * ekininit * selectedIons.charge / selectedIons.mass)
length = positionArray[-1] + 25 * mm
tmax = length / velo  # this is used for the maximum time for timesteps
zrunmax = length  # this is used for the maximum distance for timesteps
period = 1 / freq
tcontrol = period / 2
scale_maxEz = 1.25
app_maxEz = scale_maxEz * Vmax / geometry.gapGNDRF
if warpoptions.options.runtime:
    tmax = warpoptions.options.runtime

# Create a lab window for the collecting diagnostic data at the end of the run.
# Create zparticle diagnostic. The function gchange is needed to allocate
# arrays for the windo moments. Lastly, create variables for the species index.
zdiagn = ZCrossingParticles(zz=max(z) - 10 * solver.dz, laccumulate=0)
selectind = 0
otherind = 1
# First loop. Inject particles for 1.5 RF cycles then cut in injection.
# while wp.top.time <= period:
#     # Create pseudo random injection
#     Nselect = np.random.randint(low=1, high=20)
#     Nother = np.random.randint(low=1, high=20)
#     wp.top.rnpinje_s = [Nselect, Nother]
#
#     # Plot particle trajectory in zx
#     wp.window(2)
#     wp.pfzx(
#         fill=1,
#         filled=1,
#         plotselfe=1,
#         comp="z",
#         contours=50,
#         cmin=-app_maxEz,
#         cmax=app_maxEz,
#         titlet="Ez, N+(Blue) and N2+(Red)",
#     )
#     selectedIons.ppzx(color=wp.blue, msize=2, titles=0)
#     ions.ppzx(color=wp.red, msize=2, titles=0)
#     wp.limits(z.min(), z.max(), x.min(), x.max())
#     wp.fma()
#
#     # Plot particle trajectory in xy
#     wp.window(3)
#     selectedIons.ppxy(
#         color=wp.blue, msize=2, titlet="Particles N+(Blue) and N2+(Red) in XY"
#     )
#     ions.ppxy(color=wp.red, msize=2, titles=0)
#     wp.limits(x.min(), x.max(), y.min(), y.max())
#     wp.plg(Y, X, type="dash")
#     wp.titlet = "Particles N+(Blue) and N2+(Red) in XY"
#     wp.fma()
#
#     wp.step(1)
#
#
# # Turn injection off
# wp.top.inject = 0

# Grab number of particles injected.
hnpinj = wp.top.hnpinject[: wp.top.jhist + 1, :]
hnpselected = sum(hnpinj[:, 0])
hnpother = sum(hnpinj[:, 1])

# Creat array for holding number of particles that cross diagnostics
npdiagn_select = []
npdiagn_other = []
vz_select = []
vz_other = []
tdiagn_select = []
tdiagn_other = []

# Create vertical line for diagnostic visual
pltdiagn_x = np.ones(3) * zdiagn.zz
pltdiagn_y = np.linspace(-wp.largepos, wp.largepos, 3)

# Main loop. Advance particles until N+ reaches end of frame and output graphics.
while wp.top.time < 5 * period:
    # Create pseudo random injection
    Nselect = np.random.randint(low=1, high=20)
    Nother = np.random.randint(low=1, high=20)
    wp.top.rnpinje_s = [Nselect, Nother]

    # while max(ions.getz()) < z.max() - 3 * solver.dz:
    # Check whether diagnostic arrays are empty
    if zdiagn.getn(selectind) != 0:
        npdiagn_select.append(zdiagn.getn(selectind))
        vz_select.append(zdiagn.getvz(selectind).mean())
        tdiagn_select.append(zdiagn.gett(selectind).mean())

    if zdiagn.getn(otherind) != 0:
        npdiagn_other.append(zdiagn.getn(otherind))
        vz_other.append(zdiagn.getvz(otherind).mean())
        tdiagn_other.append(zdiagn.gett(selectind).mean())

    wp.window(2)
    wp.pfzx(
        fill=1,
        filled=1,
        plotselfe=1,
        comp="z",
        contours=50,
        cmin=-app_maxEz,
        cmax=app_maxEz,
        titlet="Ez, Ar+(Blue) and N2+(Red)",
    )
    selectedIons.ppzx(color=wp.blue, msize=2, titles=0)
    wp.plg(pltdiagn_y, pltdiagn_x, width=3, color=wp.magenta)
    wp.limits(z.min(), z.max(), x.min(), x.max())
    wp.fma()

    wp.window(3)
    selectedIons.ppxy(
        color=wp.blue, msize=2, titlet="Particles Ar+(Blue) and N2+(Red) in XY"
    )
    wp.limits(x.min(), x.max(), y.min(), y.max())
    wp.plg(Y, X, type="dash")
    wp.titlet = "Particles Ar+(Blue) and N2+(Red) in XY"
    wp.fma()

    wp.step(1)

### END of Simulation
# Grab number of particles injected.
hnpinj = wp.top.hnpinject[: wp.top.jhist + 1, :]
hnpselected = sum(hnpinj[:, 0])
hnpother = sum(hnpinj[:, 1])
print("Number {} injected: {}".format(selectedIons.name, hnpselected))
npdiagn_select, npdiagn_other = np.array(npdiagn_select), np.array(npdiagn_other)
vz_select, vz_other = np.array(vz_select), np.array(vz_other)
tdiagn_select, tdiagn_other = np.array(tdiagn_select), np.array(tdiagn_other)

# Calculate KE and current statistics
keselect = selectedIons.mass * pow(vz_select, 2) / 2 / wp.jperev

currselect = selectedIons.charge * vz_select * npdiagn_select

# Calculate end of simulation KE for all particles. This will entail grabbing
# values from the lost particle histories.
inslost = wp.top.inslost  # Starting index for each species in the lost arrays
uzlost = wp.top.uzplost  # Vz array for lost particle velocities
Nuz = np.hstack((selectedIons.getvz(), uzlost[inslost[0] : inslost[-1]]))
Nke = selectedIons.mass * pow(Nuz, 2) / 2 / wp.jperev

# Plot statistics. Find limits for axes.
KEmax_limit = max(max(keselect), max(keother))
tmin_limit = min(min(tdiagn_select), min(tdiagn_other))
tmax_limit = max(max(tdiagn_select), max(tdiagn_other))
currmax_limit = max(max(currselect), max(currother))

# Create plots for kinetic energy, current, and particle counts.
fig, ax = plt.subplots(nrows=3, ncols=1)
keplt = ax[0]
currplt = ax[1]
currplt.sharex(keplt)
kehist = ax[2]

# Make KE plots
keplt.plot(tdiagn_select / ns, keselect / wp.kV, c="b")
keplt.plot(tdiagn_other / ns, keother / wp.kV, c="r")
keplt.set_xlim(tmin_limit / ns, tmax_limit / ns)
keplt.set_ylim(ekininit / wp.kV, KEmax_limit / wp.kV)
keplt.set_ylabel("Avg KE in z [KeV]")

# Make Current Plots
currplt.plot(tdiagn_select / ns, currselect / 1e-6, c="b")
currplt.plot(tdiagn_other / ns, currother / 1e-6, c="r")
currplt.set_xlim(tmin_limit / ns, tmax_limit / ns)
currplt.set_ylim(0, currmax_limit / 1e-6)
currplt.set_ylabel(r" Avg Current [$\mu$A]")
currplt.set_xlabel("Time [ns]")

# Make histogram of particle energies for each species
kehist.hist(
    Nke / wp.kV, bins=100, color="b", alpha=0.7, edgecolor="k", linewidth=1, label="N+"
)
kehist.hist(
    N2ke / wp.kV,
    bins=100,
    color="r",
    alpha=0.7,
    edgecolor="k",
    linewidth=1,
    label="N2+",
)
kehist.set_xlabel("End Energy [KeV]")
kehist.set_ylabel("Number of Particles")
kehist.legend()
plt.tight_layout()
plt.savefig("stats", dpi=300)
plt.show()

# wp.window(4)
# # Plot kinetic energy for each species at the diagnostic.
# wp.limits(0,  tmax_limit, ekininit, KEmax_limit)
# wp.plg(
#     keselect,
#     tdiagn_select,
#     color = wp.blue,
#     type = 'dot',
#     msize = 8
# )
# wp.plg(
#     keother,
#     tdiagn_other,
#     color = wp.red,
#     type = 'dot',
#     msize = 8
# )
# title = "Kinetic Energy in Z, N+(Blue), N2+(Red)"
# xlabel = "time [s]"
# ylabel = "Kinetic Energy [eV]"
# wp.ptitles(title, xlabel, ylabel)
# wp.fma()
#
# # Plot the current for each species at diagnostic
# wp.limits(0, tmax_limit, 0, currmax_limit)
# wp.plg(
#     currselect,
#     tdiagn_select,
#     color = wp.blue,
#     type = 'dot',
#     msize = 8
# )
# wp.plg(
#     currother,
#     tdiagn_other,
#     color = wp.red,
#     type = 'dot',
#     msize = 8
# )
# title = "Current N+(Blue), N2+(Red)"
# xlabel = "time [s]"
# ylabel = "Current [A] "
# wp.ptitles(title, xlabel, ylabel)
# wp.fma()
#
# # Plot the number of particles for eachvspecies at diagnostic
# wp.plg(
#     npdiagn_select,
#     tdiagn_select,
#     color = wp.blue,
#     marker = '\4',
#     type = 'dot',
#     msize = 8
# )
# wp.plg(
#     npdiagn_other,
#     tdiagn_other,
#     color = wp.red,
#     marker = '\5',
#     type = 'dot',
#     msize = 8
# )
# title = "Particle Counts N+(Blue), N2+(Red)"
# xlabel = "time [s]"
# ylabel = "Number of Particles "
# wp.ptitles(title, xlabel, ylabel)
# wp.fma()

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
