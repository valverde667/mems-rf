"""
Simplfied ESQ model
"""

from warp import *

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

# Parameters available for scans
gap = 500*um # distance between grounded and potential RF plates
Vesq = 548.0 # voltage of ESQs
top.dt = 5e-11 # time beam is between ground and potential???

# --- Invoke setup routine for the plotting
setup(prefix="esq-V{}-gap-{}um".format(int(Vesq), int(gap*1e6)))

# --- Set basic beam parameters
emittingradius = 40*um
ibeaminit = 20e-6
ekininit = 40e3


# selectedIons = Species(type=Xenon, charge_state=1, name='Xe', color=green)
# rejectedIons = Species(type=Xenon, charge_state=1, name='Xe', color=red)

selectedIons = Species(type=Phosphorus, charge_state=1, name='P', color=green)
rejectedIons = Species(type=Sulfur, charge_state=1, name='S', color=red)

 # number of rf cells
 #each cell has 2 acceleration stages
rfCount = 10

top.a0 = emittingradius
top.b0 = emittingradius
top.ap0 = 14.913e-3
top.bp0 = -14.913e-3
top.vbeam = .0e0
top.emit = 0.77782e-6
top.ibeam = ibeaminit
top.ekin = ekininit
top.zion = selectedIons.charge_state
#top.vthz = 0.0
top.lrelativ = False
derivqty()

# --- Set input parameters describing the 3d simulation
w3d.l4symtry = True
w3d.l2symtry = False

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
w3d.nx = 50.
w3d.ny = 50.
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
VRF = 0.0

ESQs = []
RFs = []
ID_ESQ = 100
ID_RF = 201
geometry.pos = -0.5*gap-(2*um+500*um+2*um)-50*um
print("starting pos:", geometry.pos)

# set up time varying fields on the RF electrodes
Vmax = 5e3
freq = 100e6


def gen_volt(toffset=0):
    def RFvoltage(time):
        return Vmax*np.sin(2*np.pi*freq*(time+toffset-1.5e-9))
    return RFvoltage

#origional time offsets for Xe
#toffsets = [3.5e-9, 0.0e-9, 5.5e-9, 0.5e-9, 4.5e-9, 8.5e-9]

toffsets = [0] * rfCount
Ekin = ekininit
#rfgaps = [rfgap] * rfCount

# Origional manualy adjusted gaps for Xe
# rfgaps = [rfgap+0.5*mm, rfgap+0.144*mm, rfgap+0.146*mm,
#           rfgap+0.375*mm, rfgap+0.2992*mm, rfgap+0.6986*mm]


Vpos = []

for i, toffset in zip(range(rfCount), toffsets):
    rfgap = np.sqrt(2*Ekin*selectedIons.charge/selectedIons.mass)/freq/2-0.5*mm
    Ekin += 2 * 0.8 * Vmax
    RF = RF_stack2(condid=[ID_RF, ID_RF+1, ID_RF+2, ID_RF+3],
                   rfgap=rfgap, voltage=gen_volt(toffset))
    Vpos.append(geometry.pos)
    Gap(gap)
    E1 = ESQ(voltage=Vesq, condid=[ID_ESQ, ID_ESQ+1])
    Gap(gap)
    E2 = ESQ(voltage=-Vesq, condid=[ID_ESQ+2, ID_ESQ+3])
    Gap(gap)

    Vesq *= 1.02

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

print("total length", geometry.pos)

#scraper = ParticleScraper(ESQs +  RFs)
conductors = sum(ESQs) + sum(RFs)

velo = np.sqrt(2*ekininit*selectedIons.charge/selectedIons.mass)
length = geometry.pos
tmax = length/velo
zrunmax = length

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
zmid = 0.4*(zmax+zmin)

# make a circle to show the beam pipe
R = 90*um
t = np.linspace(0, 2*np.pi, 100)
X = R*np.sin(t)
Y = R*np.cos(t)
deltaKE = 10e3

while (top.time < tmax and zmax < zrunmax):
    step(10)

    Volts = []
    for i, t in zip(range(rfCount), toffsets):
        func = gen_volt(t)
        Volts.append(func(top.time))

    tmp = "V: {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} {:.0f} ".format(Volts[0],
                                                                 Volts[1],
                                                                 Volts[2],
                                                                 Volts[3],
                                                                 Volts[4],
                                                                 Volts[5])
##    tmp = " Voltage: {}V gap: {}um".format(int(Vesq), int(1e6*gap))
    top.pline1 = tmp

    # inject only for 1 ns, so that we can get onto the rising edge of the RF
    if 0 < top.time < 1.5e-9:
        top.finject[0,selectedIons.jslist[0]] = 1 #?
        top.finject[0,rejectedIons.jslist[0]] = 1 #?
        #top.inject = 1
    else:
        top.inject = 0

    #which species to follow
    Z = selectedIons.getz()
    if Z.mean() > zmid:
        top.vbeamfrm = selectedIons.getvz().mean()
        solver.gridmode = 0

    zmin = top.zbeam+w3d.zmmin
    zmax = top.zbeam+w3d.zmmax

    # create some plots
    KE = selectedIons.getke()
    if len(KE) > 0:
        selectedIons.ppzke(color=red)
        KEmin, KEmax = KE.min(), KE.max()
        while KEmax-KEmin > deltaKE:
            deltaKE += 10e3
        ylimits(0.95*KEmin, 0.95*KEmin+deltaKE)
        fma()

    pfxy(iz=w3d.nz//2, fill=0, filled=1, plotselfe=2, comp='E', titles=0, cmin=0, cmax=5e6*Vesq/125)
    limits(-w3d.xmmax, w3d.xmmax)
    ylimits(-w3d.ymmax, w3d.ymmax)
    ptitles("Geometry and Fields", "X [m]", "Y [m]", "")
    fma()

    # moveie plot
    pfzx(fill=1, filled=1, plotselfe=2, comp='z', titles=0, cmin=0, cmax=5e6)
    selectedIons.ppzx(color=green, titles=0)
    rejectedIons.ppzx(color=red, titles=0)
    ptitles("Partcles and Fields", "Z [m]", "X [m]", "")
    limits(zmin, zmax)
    fma()

    selectedIons.ppxy(color=red, titles=0)
    limits(-R, R)
    ylimits(-R, R)
    plg(Y, X, type="dash")
    fma()
    refresh()


# plot particle vs time
# hpepsnxz()
# fma()
# hpepsnyz()
# hpepsnx()
# fma()

hpepsny()
fma()
hpepsnz()
fma()
hpeps6d()
fma()
hpekinz()
ylimits(35e-3, KEmax*1e-6*1.2)
fma()
hpekin()
ylimits(35e-3, KEmax*1e-6*1.2)
fma()
hpxrms(color=red, titles=0)
hpyrms(color=blue, titles=0)
hprrms(color=green, titles=0)
ptitles("X(red), Y(blue), R(green)", "Z [m]", "X/Y/R [m]", "")

fma()

# plot number of particles over time
hppnum()
fma()

# hplinechg()
# fma()

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

for i in (t, hepsny, hepsnz, hep6d, hekinz, hekin, hxrms, hyrms, hrrms, hpnum):
    print(len(i))

out = np.stack((t, hepsny, hepsnz, hep6d, hekinz, hekin, hxrms, hyrms, hrrms, hpnum))
np.save("esqhist.npy", out)
