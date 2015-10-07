"""
Simplfied ESQ model
"""
from __future__ import print_function

import warpoptions
warpoptions.parser.add_argument('--Vesq', dest='Vesq', type=float, default=548.)

from warp import *
from warp.egun_like import *
from warp.ionization import *
from warp.timedependentvoltage import TimeVoltage

import numpy as np

import geometry
from geometry import Aperture, ESQ, RF_stack2, Gap
from helper import gitversion

# which geometry to use 2d or 3d
# w3d.solvergeom = w3d.RZgeom
w3d.solvergeom = w3d.XYZgeom

# define some strings that go into the output file
top.pline1 = "ESQ model"
top.pline2 = " " + gitversion()
top.runmaker = "Arun Persaud (apersaud@lbl.gov)"

# --- Invoke setup routine for the plotting
gap = 500*um
Vesq = warpoptions.options.Vesq
setup(prefix="esq-V{}-gap-{}um".format(int(Vesq), int(gap*1e6)))

# --- Set basic beam parameters
emittingradius = 48.113*um
ibeaminit = 20e-6
ekininit = 40e3

ions = Species(type=Xenon, charge_state=1, name='Xe')

top.a0 = emittingradius
top.b0 = emittingradius
top.ap0 = 14.913e-3
top.bp0 = -14.913e-3
top.vbeam = .0e0
top.emit = 0.77782e-6
top.ibeam = ibeaminit
top.ekin = ekininit
top.aion = ions.type.A
top.zion = ions.charge_state
#top.vthz = 0.0
top.lrelativ = False
derivqty()

# --- Set input parameters describing the 3d simulation
top.dt = 5e-11
w3d.l4symtry = False
w3d.l2symtry = True

# --- Set boundary conditions

# ---   for field solve
w3d.bound0 = dirichlet
w3d.boundnz = neumann
w3d.boundxy = neumann

# ---   for particles
top.pbound0 = absorb
top.pboundnz = absorb
top.prwall = np.sqrt(2)*1.5*mm/2.0
top.prwall = 90*um

# --- Set field grid size
w3d.xmmin = -0.0005/2.
w3d.xmmax = +0.0005/2.
w3d.ymmin = -0.0005/2.
w3d.ymmax = +0.0005/2.
w3d.zmmin = 0.0
w3d.zmmax = 0.002

# set grid spacing
w3d.nx = 100.
w3d.ny = 100.
w3d.nz = 100.

if w3d.l4symtry:
    w3d.xmmin = 0.
    w3d.nx /= 2
if w3d.l2symtry or w3d.l4symtry:
    w3d.ymmin = 0.
    w3d.ny /= 2

# --- Select plot intervals, etc.
top.npmax = 300
top.inject = 1  # 2 means space-charge limited injection
top.rinject = 9999.
top.npinject = 300  # needed!!
top.linj_eperp = True  # Turn on transverse E-fields near emitting surface
top.zinject = w3d.zmmin
top.vinject = 1.0

top.nhist = 1  # Save history data every time step
top.itmomnts[0:4] = [0, 1000000, top.nhist, 0]  # Calculate moments every step
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
solver.mgtol = 1.0e-2  # Poisson solver tolerance, in volts
solver.mgparam = 1.5
solver.downpasses = 2
solver.uppasses = 2

# --- Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.)
package("w3d")
generate()

# --- define voltages
Vground = 0.0e3
VRF = 0.0

ESQs = []
RFs = []
ID_ESQ = 100
ID_RF = 201
geometry.pos = -0.5*gap-(2*um+500*um+2*um)-50*um
print("starting pos:", geometry.pos)
A1 = Aperture(0, 95, width=50*um)
for i in range(6):
    RF = RF_stack2(voltage=VRF, condid=list(range(ID_RF, ID_RF+4)), rfgap=gap)
    Gap(gap)
    E1 = ESQ(voltage=Vesq, condid=[ID_ESQ, ID_ESQ+1])
    Gap(gap)
    E2 = ESQ(voltage=-Vesq, condid=[ID_ESQ+2, ID_ESQ+3])
    Gap(gap)

    ESQs.append(E1)
    ESQs.append(E2)
    RFs.append(RF)
    ID_ESQ += 4
    ID_RF += 4
Gap(gap)
E1 = ESQ(voltage=Vesq, condid=[ID_ESQ, ID_ESQ+1])
Gap(gap)
E2 = ESQ(voltage=-Vesq, condid=[ID_ESQ+2, ID_ESQ+3])
Gap(gap)
A2 = Aperture(0, 95, width=50*um)

Apertures = [A1, A2]

print("total length", geometry.pos)

#scraper = ParticleScraper(ESQs +  RFs)
conductors = sum(ESQs) + sum(RFs) + sum(Apertures)

velo = np.sqrt(2*ekininit*ions.charge/ions.mass)
length = geometry.pos
tmax = length/velo
zrunmax = length

# set up time varying fields on the RF electrodes
toffset = 2.5e-9
Vmax = 0.5*10e3
freq = 100e6
def RFvoltage1(time):
    return 0
    return -Vmax*np.sin(2*np.pi*freq*(time-toffset))

def RFvoltage2(time):
    return -RFvoltage1(time)

def RFvoltage3(time):
    return -RFvoltage1(time)

RF1a = TimeVoltage(202, voltfunc=RFvoltage1)
RF1b = TimeVoltage(203, voltfunc=RFvoltage1)
RF2a = TimeVoltage(206, voltfunc=RFvoltage2)
RF2b = TimeVoltage(207, voltfunc=RFvoltage2)
RF3a = TimeVoltage(210, voltfunc=RFvoltage3)
RF3b = TimeVoltage(211, voltfunc=RFvoltage3)

# define the electrodes
installconductors(conductors)

# --- Recalculate the fields
fieldsol(-1)

# I want contour plots for levels between 0 and 1kV
contours = range(0, int(Vesq), int(Vesq/10))

winon(xon=1)

# some plots of the geometry
pfzx(fill=1, filled=1, plotphi=0)
fma()
pfzx(fill=1, filled=1, plotphi=1)
fma()

zmin = w3d.zmmin
zmax = w3d.zmmax
zmid = 0.5*(zmax+zmin)

# make a circle to show the beam pipe
R = 90*um
t = np.linspace(0, 2*np.pi, 100)
X = R*np.sin(t)
Y = R*np.cos(t)

# check the fields in one ESQ

#for i in range(100):
#    fma()
#    pfxy(iz=i,fill=0, filled=1, plotselfe=2, comp='E', cmin=-5e6, cmax=5e6)
#    limits(-w3d.xmmax, w3d.xmmax)
#    ylimits(-w3d.ymmax, w3d.ymmax)
#for i in range(100):
#    fma()
#    pfzx(iy=i,fill=0, filled=1, plotselfe=2, comp='x', cmin=-5e6, cmax=5e6)
#    limits(w3d.zmmin, w3d.zmmax)
#    ylimits(-w3d.xmmax, w3d.xmmax)
#for i in range(100):
#    fma()
#    pfzy(ix=i,fill=0, filled=1, plotselfe=2, comp='y', cmin=-5e6, cmax=5e6)
#    limits(w3d.zmmin, w3d.zmmax)
#    ylimits(-w3d.ymmax, w3d.ymmax)
#
#import sys
#sys.exit()

zrunmax = 5*mm

while (top.time < tmax and zmax < zrunmax):
    step(10)

#    tmp = " Voltages: {} {} {}".format(RF1a.getvolt(top.time), RF2a.getvolt(top.time), RF3a.getvolt(top.time))
    tmp = " Voltage: {}V gap: {}um".format(int(Vesq), int(1e6*gap))
    top.pline1 = tmp

    # inject only for 1 ns, so that we can get onto the rising edge of the RF
    if 0 < top.time < 1e-9:
        top.inject = 1
    else:
        top.inject = 0

    Z = ions.getz()
    if Z.mean() > zmid:
        top.vbeamfrm = velo
        solver.gridmode = 0

    zmin = top.zbeam+w3d.zmmin
    zmax = top.zbeam+w3d.zmmax

    # create some plots
    ions.ppzvz(color=red)
    ylimits(0.95*ekininit, 1.05*ekininit)
    fma()
    pfxy(iz=w3d.nz//2, fill=0, filled=1, plotselfe=2, comp='E', titles=0, cmin=0, cmax=5e6*Vesq/125)
    limits(-w3d.xmmax, w3d.xmmax)
    ylimits(-w3d.ymmax, w3d.ymmax)
    ptitles("Geometry and Fields","X [m]", "Y [m]", "")
    fma()
    pfzx(fill=1, filled=1, plotselfe=2, comp='E', titles=0, cmin=0, cmax=5e6)
    ions.ppzx(color=red, titles=0)
    ptitles("Particles and Fields","Z [m]", "X [m]", "")
    limits(zmin, zmax)
    fma()
    ions.ppxy(color=red, titles=0)
    limits(-R, R)
    ylimits(-R, R)
    plg(Y,X,type="dash")
    fma()
    refresh()

#plot particle vs time
#hzepsnxz()
#fma()
#hzepsnyz()
#hzepsnx()
#fma()
#hzepsny()
#hzepsnz()
#hzeps6d()
#hztotalke()
#hztotale()
hzxrms(color=red, titles=0)
hzyrms(color=blue, titles=0)
hzrrms(color=green, titles=0)
ptitles("X(red), Y(blue), R(green)", "Z [m]", "X/Y/R [m]", "")

fma()
hzpnum()

fma()
#hzlinechg()
#fma()
