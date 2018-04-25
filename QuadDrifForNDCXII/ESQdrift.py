"""
Simplfied ESQ model
"""
import warpoptions
warpoptions.parser.add_argument('--esq_voltage', dest='Vesq', type=float, default='2203.2')
warpoptions.parser.add_argument('--numESQ', dest='numESQ', type=int, default='2')
#potential to have all input parameters in one line? maybe input parameters is list?

#warp.options.parser.add_argument('--input', dest=)
from warp import *

import numpy as np

import geometry
from geometry import Aperture, ESQ, RF_stack3, Gap
#from helper import gitversion
import matplotlib.pyplot as plt

# which geometry to use 2d or 3d
# w3d.solvergeom = w3d.RZgeom
w3d.solvergeom = w3d.XYZgeom

# define some strings that go into the output file
top.pline1 = "ESQ model"
#top.pline2 = " " + gitversion()
# top.runmaker = "Arun Persaud (apersaud@lbl.gov)"


top.dt = 5e-11

#often tweeked
selectedIons = Species(charge_state=1, name='H2', mass=2*amu, color=green)
ekininit = 200e3#1100e3#35.6e3#
Vesq = warpoptions.options.Vesq
numESQ = warpoptions.options.numESQ
ibeaminit = 1e-6




# --- Invoke setup routine for the plotting
setup(prefix="esq-V{}-gap-{}um-{}-{}".format(int(Vesq), int(geometry.RF_gap*1e6), selectedIons.name, numESQ),cgmlog=0)

# --- Set basic beam parameters
emittingradius = 1*mm#0.055307*mm


top.a0 = emittingradius
top.b0 = emittingradius
top.ap0 = 1e-3#2.919e-3#27.021e-3
top.bp0 = -1e-3#-2.919e-3#-27.021e-3
top.vbeam = .0e0
top.emit = 9.96635e-7
top.ibeam = ibeaminit
top.ekin = ekininit
#top.aion = selectedIons.type.A
top.zion = selectedIons.charge_state
#top.vthz = 0.0
top.lrelativ = False
top.linj_efromgrid = True
derivqty()

# --- Set input parameters describing the 3d simulation
w3d.l4symtry = True
w3d.l2symtry = False

# --- Set boundary conditions

# ---   for field solve
w3d.bound0 = neumann
w3d.boundnz = neumann
w3d.boundxy = neumann

# ---   for particles
top.pbound0 = absorb
top.pboundnz = absorb
#top.prwall = np.sqrt(2)*1.5*mm/2.0
top.prwall = 1*mm#90*um*2

# --- Set field grid size
w3d.xmmin = -0.02/8.
w3d.xmmax = +0.02/8.
w3d.ymmin = -0.02/8.
w3d.ymmax = +0.02/8.
w3d.zmmin = 0.0
w3d.zmmax = 20*mm

# set grid spacing
w3d.nx = 50.
w3d.ny = 50.
w3d.nz = 400.

if w3d.l4symtry:
    w3d.xmmin = 0.
    w3d.nx /= 2
if w3d.l2symtry or w3d.l4symtry:
    w3d.ymmin = 0.
    w3d.ny /= 2

# --- Select plot intervals, etc.
top.npmax = 300
top.inject = 1  # 2 means space-charge limited injection
top.rinject = 5000  # 9999.
top.npinject = 30  # 300  # needed!! macro particles per time step or cell
top.linj_eperp = False  # Turn on transverse E-fields near emitting surface
#top.zinject = 1*mm  # w3d.zmmin#w3d.zmmin
top.zinject = w3d.zmmin
top.vinject = 1.0
print("--- Ions start at: ", top.zinject)

top.nhist = 5  # Save history data every N time step
top.itmomnts[0:4] = [0, 1000000, top.nhist, 0]  # Calculate moments every N steps
# --- Save time histories of various quantities versus z.
top.lhpnumz = True
top.lhcurrz = True
top.lhrrmsz = True
top.lhxrmsz = True
top.lhyrmsz = True
top.lhepsnxz = True
top.lhepsnyz = True
top.lhvzrmsz = True

# --- Set up fieldsolver - 7 means the multigrid solver
solver = MRBlock3D()
registersolver(solver)
solver.mgtol = 1.0  # Poisson solver tolerance, in volts
solver.mgparam = 1.5
solver.downpasses = 2
solver.uppasses = 2

# --- Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.)
package("w3d")
generate()

# --- define voltages
Vground = 0.0e3
VRF = 1000.0

ESQs = []
RFs = []
ID_ESQ = 100
ID_RF = 201

def gen_volt_esq(Vesq, inverse=False, toffset=0):
    def ESQvoltage(time):
        if inverse:
            return -Vesq
        else:
            return Vesq
    return ESQvoltage


# this loop generates the geometry
ESQ_toffset = 0
gaplength = 8*mm#16*mm

for i in range(numESQ):
    Gap(gaplength/2)
    E1 = ESQ(voltage=gen_volt_esq(Vesq, False, ESQ_toffset), condid=[ID_ESQ, ID_ESQ+1])
    Gap(geometry.ESQ_gap)
    E2 = ESQ(voltage=gen_volt_esq(Vesq, True, ESQ_toffset), condid=[ID_ESQ+2, ID_ESQ+3])
    Gap(gaplength/2)

    ESQs.append(E1)
    ESQs.append(E2)
    ID_ESQ += 4


conductors = sum(ESQs)

velo = np.sqrt(2*ekininit*selectedIons.charge/selectedIons.mass)
length = geometry.pos
tmax = length/velo
zrunmax = length + 4*mm

# define the electrodes
installconductors(conductors)

# --- Recalculate the fields
fieldsol(-1)

# I want contour plots for levels between 0 and 1kV
contours = range(0, int(Vesq), int(Vesq/10))

winon(xon=0)

# some plots of the geometry
pfzx(fill=1, filled=1, plotphi=0)
fma()
pfzx(fill=1, filled=1, plotphi=1)
fma()

zmin = w3d.zmmin
zmax = w3d.zmmax
zmid = 0.5*(zmax+zmin)

# make a circle to show the beam pipe
R = 1*mm
t = np.linspace(0, 2*np.pi, 100)
X = R*np.sin(t)
Y = R*np.cos(t)
deltaKE = 10e3
time = []
numsel = []

beamwidth=[]
energy_time = []

dist = 2.5*mm
distN = 0

KE_select = []

while (zmax < zrunmax):
    step(10)
    time.append(top.time)
    numsel.append(len(selectedIons.getke()))
    KE_select.append(np.mean(selectedIons.getke()))
# saving beam data for eventual optimization
    top.pline1 = "V_esq: {:.0f}".format(gen_volt_esq(Vesq, False, ESQ_toffset)(top.time))

    # inject only for 1 ns, so that we can get onto the rising edge of the RF
    if 0*ns < top.time < 1e-9:
        top.inject = 1
        #top.finject[0,selectedIons.jslist[0]] = 1
    else:
        top.inject = 0

    Z = selectedIons.getz()
    if Z.mean() > zmid:
        top.vbeamfrm = selectedIons.getvz().mean()
        solver.gridmode = 0

    zmin = top.zbeam+w3d.zmmin
    zmax = top.zbeam+w3d.zmmax

    # # create some plots
    KE = selectedIons.getke()
    print(np.mean(KE))
    if len(KE) > 0:
        selectedIons.ppzke(color=blue)
        KEmin, KEmax = KE.min(), KE.max()
        while KEmax-KEmin > deltaKE:
            deltaKE += 10e3
    ylimits(0.95*KEmin, 0.95*KEmin+deltaKE)
    fma()

    pfzx(fill=1, filled=1, plotselfe=True, comp='E', titles=0,
         cmin=0, cmax=1.2*Vesq/geometry.RF_gap)
    selectedIons.ppzx(color=green, titles=0)
    ptitles("Particles and Fields", "Z [m]", "X [m]", "")
    limits(zmin, zmax)
    fma()

    pfxy(iz=w3d.nz//2, fill=0, filled=1, plotselfe=2, comp='E', titles=0, cmin=0, cmax=5e6*Vesq/125)
    limits(-w3d.xmmax, w3d.xmmax)
    ylimits(-w3d.ymmax, w3d.ymmax)
    ptitles("Geometry and Fields", "X [m]", "Y [m]", "")
    fma()

    selectedIons.ppxy(color=red, titles=0)
    limits(-R, R)
    ylimits(-R, R)
    plg(Y, X, type="dash")
    fma()


plg(numsel, time, color=blue)
ptitles("Number of Particles vs Time", "Time (s)", "Number of Particles")
fma()

plg(KE_select, time, color=blue)
ptitles("kinetic energy vs time")
fma()

hpxrms(color=red, titles=0)
hpyrms(color=blue, titles=0)
hprrms(color=green, titles=0)
ptitles("X(red), Y(blue), R(green)", "Z [m]", "X/Y/R [m]", "")
fma()

# save history information, so that we can plot all cells in one plot
t = np.trim_zeros(top.thist, 'b')
hepsny = selectedIons.hepsny[0]
hepsnz = selectedIons.hepsnz[0]
hep6d = selectedIons.hepsx[0] * selectedIons.hepsy[0] * selectedIons.hepsz[0]
hekinz = 1e-6*0.5*top.aion*amu*selectedIons.hvzbar[0]**2/jperev

u = selectedIons.hvxbar[0]**2 + selectedIons.hvybar[0]**2 + selectedIons.hvzbar[0]**2
hekin = 1e-6 * 0.5*top.aion*amu*u/jperev

hxrms = selectedIons.hxrms[0]
hyrms = selectedIons.hyrms[0]
hrrms = selectedIons.hrrms[0]
hpnum = selectedIons.hpnum[0]


out = np.stack((t, hepsny, hepsnz, hep6d, hekinz, hekin, hxrms, hyrms, hrrms, hpnum))
np.save("esqhist.npy", out)
