"""
Simplfied ESQ model
"""
import warpoptions
warpoptions.parser.add_argument('--esq_voltage', dest='Vesq', type=float, default='200')

from warp import *

import numpy as np

import geometry
from geometry import Aperture, ESQ, RF_stack3, Gap
from helper import gitversion

# which geometry to use 2d or 3d
# w3d.solvergeom = w3d.RZgeom
w3d.solvergeom = w3d.XYZgeom

# define some strings that go into the output file
top.pline1 = "ESQ model"
top.pline2 = " " + gitversion()
# top.runmaker = "Arun Persaud (apersaud@lbl.gov)"

# Parameters available for scans
geometry.RF_gap = 500*um
Vesq = warpoptions.options.Vesq
top.dt = 5e-11

# --- Invoke setup routine for the plotting
setup(prefix="esq-V{}-gap-{}um".format(int(Vesq), int(geometry.RF_gap*1e6)))

# --- Set basic beam parameters
emittingradius = 20*um
ibeaminit = 20e-12
ekininit = 40e3

selectedIons = Species(type=Phosphorus, charge_state=1, name='P', color=green)
rejectedIons = Species(type=Sulfur, charge_state=1, name='S', color=red)

top.a0 = emittingradius
top.b0 = emittingradius
top.ap0 = 0  # -5e-3#10e-3
top.bp0 = 0  # -5e-3#-10e-3
#top.ap0 = 14.913e-3
#top.bp0 = -14.913e-3
top.vbeam = .0e0
top.emit = 0  # 0.77782e-6
top.ibeam = ibeaminit
top.ekin = ekininit
#top.aion = selectedIons.type.A
top.zion = selectedIons.charge_state
#top.vthz = 0.0
top.lrelativ = False
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
top.prwall = np.sqrt(2)*1.5*mm/2.0
top.prwall = 90*um

# --- Set field grid size
w3d.xmmin = -0.0005/2.
w3d.xmmax = +0.0005/2.
w3d.ymmin = -0.0005/2.
w3d.ymmax = +0.0005/2.
w3d.zmmin = 0.0
w3d.zmmax = 4*mm

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
top.rinject = 5000  # 9999.
top.npinject = 30  # 300  # needed!! macro particles per time step or cell
top.linj_eperp = False  # Turn on transverse E-fields near emitting surface
top.zinject = 1*mm  # w3d.zmmin#w3d.zmmin
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
geometry.pos = 2*mm
print("starting pos:", geometry.pos)

# set up time varying fields on the RF electrodes
Vmax = 5e3
freq = 50e6


def gen_volt(toffset=0):
    """ A sin voltage function with variable offset"""
    def RFvoltage(time):
        return Vmax*np.sin(2*np.pi*freq*(time+toffset))
    return RFvoltage


def gen_volt_esq(toffset=0, inverse=False):
    def ESQvoltage(time):
        if inverse:
            return -Vesq*np.sin(2*np.pi*freq*(time+toffset))
        else:
            return Vesq*np.sin(2*np.pi*freq*(time+toffset))
    return ESQvoltage


RF_toffset = 8*ns
ESQ_toffset = 13*ns
numRF = 2*8  # the total number of accelertion gaps (must be a multiple of 2)

# calculate beta*lambda/2 distances
distances = []
energies = []
mass = selectedIons.mass
energy = ekininit
for i in range(numRF):
    energy += 0.8*Vmax  # the .8 coefficent is from the ions arriving at .8 of the maximum
    velocity = np.sqrt(2*energy*selectedIons.charge/mass)
    distance = velocity/(freq*2)
    distances.append(distance)
    energies.append(energy)

print(distances)
print(selectedIons.charge)
print(selectedIons.mass)

Vpos = []
thickness = 2*um


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
    (rf_bl2, E1), (esq_bl2, E2) = bl2s

    # use first betalamba_half for the RF unit
    RF = RF_stack3(condid=[ID_RF, ID_RF+1, ID_RF+2],
                   betalambda_half=rf_bl2, voltage=gen_volt(RF_toffset))
    Vpos.append(geometry.pos)

    # scale esq voltages
    voltage = Vesq * E2/ekininit

    # and second betalamba_half for ESQ unit
    gaplength = esq_bl2-geometry.RF_gap-2*geometry.RF_thickness
    gaplength = gaplength/2-geometry.RF_gap/2-geometry.ESQ_wafer_length
    assert gaplength > 0
    Gap(gaplength)
    E1 = ESQ(voltage=gen_volt_esq(ESQ_toffset), condid=[ID_ESQ, ID_ESQ+1])
    Gap(geometry.RF_gap)
    E2 = ESQ(voltage=gen_volt_esq(ESQ_toffset, inverse=True), condid=[ID_ESQ+2, ID_ESQ+3])
    Gap(gaplength)

    ESQs.append(E1)
    ESQs.append(E2)
    RFs.append(RF)
    ID_ESQ += 4
    ID_RF += 3


# scraper = ParticleScraper(ESQs +  RFs)
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
zmid = 0.5*(zmax+zmin)

# make a circle to show the beam pipe
R = 90*um
t = np.linspace(0, 2*np.pi, 100)
X = R*np.sin(t)
Y = R*np.cos(t)
deltaKE = 10e3
time = []
numsel = []
numrej = []
energy_time = []
dist = 2.5*mm
distN = 0

while (top.time < tmax and zmax < zrunmax):
    step(10)
    time.append(top.time)
    numsel.append(len(selectedIons.getke()))
    numrej.append(len(rejectedIons.getke()))

    top.pline1 = "V_RF: {:.0f}   V_ESQ: {:.0f}".format(
        gen_volt(RF_toffset)(top.time), gen_volt_esq(ESQ_toffset)(top.time))

    # inject only for 1 ns, so that we can get onto the rising edge of the RF
    if 0 < top.time < 1e-9:
        top.inject = 1
    else:
        top.inject = 0

    Z = selectedIons.getz()
    if Z.mean() > zmid:
        top.vbeamfrm = selectedIons.getvz().mean()
        solver.gridmode = 0

    # record time when we cross certain points
    if Z.mean() > dist + np.cumsum(distances)[distN]:
        energy_time.append(top.time)
        distN += 1

    zmin = top.zbeam+w3d.zmmin
    zmax = top.zbeam+w3d.zmmax

    # # create some plots
    KE = selectedIons.getke()

    if len(KE) > 0:
        selectedIons.ppzke(color=blue)
        KEmin, KEmax = KE.min(), KE.max()
        while KEmax-KEmin > deltaKE:
            deltaKE += 10e3
    ylimits(0.95*KEmin, 0.95*KEmin+deltaKE)
    fma()

    pfzx(fill=1, filled=1, plotselfe=True, comp='E', titles=0,
         cmin=0, cmax=1.2*Vmax/geometry.RF_gap)
    selectedIons.ppzx(color=green, titles=0)
    rejectedIons.ppzx(color=red, titles=0)
    ptitles("Partcles and Fields", "Z [m]", "X [m]", "")
    limits(zmin, zmax)
    fma()

    # geometry and fields
    pfxy(iz=w3d.nz//2, fill=0, filled=1, plotphi=1, titles=0)
    limits(-w3d.xmmax, w3d.xmmax)
    ylimits(-w3d.ymmax, w3d.ymmax)
    ptitles("Geometry and Potentials", "X [m]", "Y [m]", "")
    fma()

    # Z = rejectedIons.getz()
    # if Z.mean() > zmid:
   #	top.vbeamfrm = rejectedIons.getvz().mean()
   # 	solver.gridmode = 0
    pfzx(fill=1, filled=1, plotselfe=2, comp='z', titles=0, cmin=0, cmax=5e6)
    selectedIons.ppzx(color=green, titles=0)
    rejectedIons.ppzx(color=red, titles=0)
    ptitles("Particles and Fields", "Z [m]", "X [m]", "")
    limits(zmin, zmax)
    fma()

    Z = rejectedIons.getz()
    if Z.mean() > zmid:
        top.vbeamfrm = rejectedIons.getvz().mean()
        solver.gridmode = 0

    zmin = top.zbeam+w3d.zmmin
    zmax = top.zbeam+w3d.zmmax

    pfzx(fill=1, filled=1, plotselfe=2, comp='z', titles=0, cmin=0, cmax=5e6)
    selectedIons.ppzx(color=green, titles=0)
    rejectedIons.ppzx(color=red, titles=0)
    ptitles("Particles and Fields", "Z [m]", "X [m]", "")
    limits(zmin, zmax)
    fma()


# plot particle vs time
# hpepsnxz()
# fma()
# hpepsnyz()
# hpepsnx()
# fma()
hpepsny()
fma()


plg(numrej, time, color=red)
plg(numsel, time, color=blue)
ptitles("Attenuation vs Time", "Time (s)", "Number of Particles")
fma()

hpepsnz()
fma()

hpeps6d()
fma()

hpekinz(color=red)
l = min(len(energies), len(energy_time))
pla(np.array(energies[:l])*1e-6, energy_time[:l], color=green)
ylimits(ekininit*0.8, KEmax*1e-6*1.2)
fma()

hpekin(color=red)
pla(np.array(energies[:l])*1e-6, energy_time[:l], color=green)
ylimits(ekininit*0.8, KEmax*1e-6*1.2)
fma()

hpxrms(color=red, titles=0)
hpyrms(color=blue, titles=0)
hprrms(color=green, titles=0)
ptitles("X(red), Y(blue), R(green)", "Z [m]", "X/Y/R [m]", "")

fma()
hppnum(js=selectedIons.js, color=green)
hppnum(js=rejectedIons.js, color=red)

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

out = np.stack((t, hepsny, hepsnz, hep6d, hekinz, hekin, hxrms, hyrms, hrrms, hpnum))
np.save("esqhist-V{}-gap-{}um.{}.npy".format(int(Vesq), int(geometry.RF_gap*1e6), setup.pnumb), out)

l = min(len(energies), len(energy_time))
np.save("esq-loss-V{}-gap-{}um.{}.npy".format(int(Vesq), int(geometry.RF_gap*1e6), setup.pnumb),
        np.stack((energies[:l], energy_time[:l])))
