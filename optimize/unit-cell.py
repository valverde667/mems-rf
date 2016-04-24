"""Simulate single unit cell for ALPHA project.

Simulation is as follows
 E = ESQ wafer, g = gap, R=rf wafer, d=rfgap drift


rf_stack2:
 E g R d R g E g E g R
     ^^^^^^^^^^^^^^^
         unitcell

rf_stack3:
 E g RgR d RgR g E g E g R
     ^^^^^^^^^^^^^^^^^^^
         unitcell

beginning and end overlaps with the next unitcell, so that we can save
particels and load them

"""

import warpoptions
warpoptions.parser.add_argument('--cell', dest='cellnr', type=int, default=0)
warpoptions.parser.add_argument('--toffset', dest='toffset', type=float, default=2e-9)
warpoptions.parser.add_argument('--rfgap', dest='rfgap', type=float, default=3e-3)
warpoptions.parser.add_argument('--zoffset', dest='zoffset', type=float, default=0)
warpoptions.parser.add_argument('--Vesq', dest='Vesq', type=float, default=350.0)
warpoptions.parser.add_argument('--Vesqold', dest='Vesqold', type=float, default=350.0)
warpoptions.parser.add_argument('--ap', dest='ap', type=float, default=0.0)
warpoptions.parser.add_argument('--bp', dest='bp', type=float, default=0.0)
warpoptions.parser.add_argument('--zt', dest='zt', type=float, default=1e-9)
warpoptions.parser.add_argument('--t', dest='time', type=float, default=0.0)

from warp import *
from warp.utils.timedependentvoltage import TimeVoltage

import numpy as np
import json

from helper import gitversion
import geometry
from geometry import RF_stack2, RF_stack3, Gap, ESQ

cellnr = warpoptions.options.cellnr
toffset = warpoptions.options.toffset
zoffset = warpoptions.options.zoffset
rfgap = warpoptions.options.rfgap
Vesq = warpoptions.options.Vesq
Vesqold = warpoptions.options.Vesqold
ap = warpoptions.options.ap
bp = -warpoptions.options.bp
zt = warpoptions.options.zt
mytime = warpoptions.options.time

# which geometry to use 2d or 3d
# w3d.solvergeom = w3d.RZgeom
w3d.solvergeom = w3d.XYZgeom

# define some strings that go into the output file
top.pline1 = "Unit cell {}".format(cellnr)
top.pline2 = " " + gitversion()
top.runmaker = "Arun Persaud (apersaud@lbl.gov)"

# Parameters available for scans
gap = 500*um
top.dt = 5e-11

# --- Invoke setup routine for the plotting
setup(prefix="unitcell-{:03d}".format(cellnr))


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

# want a step size in Z of 20um
ESQwaferthickness = 522e-6
RFwaferthickness = 504e-6
#Lunit = 6*waferthickness + 4*gap + rfgap
Lunit = 2*RFwaferthickness + 2*ESQwaferthickness + 3*gap + rfgap
#L = 2*waferthickness + 2*gap + Lunit
L = ESQwaferthickness + gap + Lunit + RFwaferthickness
if cellnr == 0:
    L += 2*RFwaferthickness + ESQwaferthickness + 2*gap + rfgap  # use complete ESQ doublet + RFgap
N = int(L/20e-6)

if cellnr == 0:
    #    w3d.zmmin = zoffset - 3*waferthickness - 2*gap - rfgap
    w3d.zmmin = -0.0008040000000000001
else:
    w3d.zmmin = zoffset - ESQwaferthickness - gap
w3d.zmmax = w3d.zmmin + L

# set grid spacing
w3d.nx = 50.
w3d.ny = 50.
w3d.nz = N

if w3d.l4symtry:
    w3d.xmmin = 0.
    w3d.nx /= 2
if w3d.l2symtry or w3d.l4symtry:
    w3d.ymmin = 0.
    w3d.ny /= 2

print("min: ", w3d.zmmin, " max: ", w3d.zmmax)

# --- Set basic beam parameters
emittingradius = 40*um
ibeaminit = 20e-6
ekininit = 40e3

ions = Species(type=Xenon, charge_state=1, name='Xe')

if cellnr == 0:
    top.a0 = emittingradius
    top.b0 = emittingradius
    #    top.ap0 = ap
    #    top.bp0 = bp
    #    top.vbeam = .0e0
    #    top.emit = 0
    #    top.ap0 = 14.913e-3   # for testing
    top.bp0 = -14.913e-3
    top.vbeam = .0e0
    top.emit = 0.77782e-6  # end for testing
    top.ibeam = ibeaminit
    top.ekin = ekininit
    top.aion = ions.type.A
    top.zion = ions.charge_state
    # top.vthz = 0.0
    top.lrelativ = False
    derivqty()
else:
    fin = PRpickle.PR("save-{:03d}.pkl".format(cellnr-1))
    print("particles X: ", len(fin.x), "minmax", fin.x.min(), fin.x.max())
    print("particles Y: ", len(fin.y), "minmax", fin.y.min(), fin.y.max())
    print("particles Z: ", len(fin.z), "minmax", fin.z.min(), fin.z.max())
    ions.addparticles(fin.x, fin.y, fin.z, fin.vx, fin.vy, fin.vz,
                      lallindomain=True, resetrho=False, resetmoments=False)
    fin.close()
    Z = ions.getz()
    print("NR: ", len(Z), "minmax", Z.min(), Z.max())

# --- Select plot intervals, etc.
top.npmax = 300
if cellnr == 0:
    top.inject = 1  # 2 means space-charge limited injection
    top.rinject = 9999.
    top.npinject = 300  # needed!!
    top.linj_eperp = True  # Turn on transverse E-fields near emitting surface
    top.zinject = 0.0   # start between ESQ wafers
    top.vinject = 1.0
    print("--- Ions start at: ", top.zinject)

top.nhist = 5  # Save history data every N time step
top.itmomnts[0:4] = [0, 1000000, top.nhist, 0]  # Calculate moments every N step
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

Vmax = 5e3
freq = 100e6


def gen_volt(toffset=0):
    def RFvoltage(time):
        return Vmax*np.sin(2*np.pi*freq*(time-toffset+mytime))
    return RFvoltage

ID = 100

geometry.pos = w3d.zmmin

elements = []

# add full ESQ doublet
if cellnr == 0:
    Ekin = ekininit
    rfgap0 = 0.00121234121477
    toffset0 = 3.5e-9
    RF = RF_stack2(condid=[ID, ID+1, ID+2, ID+3],
                   rfgap=rfgap0, voltage=0)
    TimeVoltage(condid=ID+1, voltfunc=gen_volt(toffset0))
    TimeVoltage(condid=ID+2, voltfunc=gen_volt(toffset0))
    elements.append(RF)
    ID += 4
    Gap(dist=gap)
    Vesqold = 548.0
    e = ESQ(Vesqold, [ID, ID+1])
    elements.append(e)
    ID += 2
    Gap(dist=gap)

e = ESQ(-Vesqold, [ID, ID+1])
elements.append(e)
ID += 2
Gap(dist=gap)

# now the unit cell begins
# RF = RF_stack3(condid=[ID, ID+1, ID+2, ID+3],
#               rfgap=rfgap, gap=gap, voltage=0.0)
UnitCellStart = geometry.pos

RF = RF_stack2(condid=[ID, ID+1, ID+2, ID+3],
               rfgap=rfgap, voltage=0)
TimeVoltage(condid=ID+1, voltfunc=gen_volt(toffset))
TimeVoltage(condid=ID+2, voltfunc=gen_volt(toffset))
ID += 4
Gap(dist=gap)
elements.append(RF)

e = ESQ(Vesq, [ID, ID+1])
elements.append(e)
ID += 2
Gap(dist=gap)

e = ESQ(-Vesq, [ID, ID+1])
elements.append(e)
ID += 2

Zend = geometry.pos  # if all particles passed this position, we end the simulation

Gap(dist=gap)

UnitCellEnd = geometry.pos
zoffsetout = geometry.pos  # offset for next iteration

# RF = RF_stack2(condid=[ID, ID+1, ID+2, ID+3],
#               rfgap=rfgap, gap=gap, voltage=0)
RF = RF_stack2(condid=[ID, ID+1, ID+2, ID+3],
               rfgap=rfgap, voltage=0)
ID += 4
elements.append(RF)

conductors = sum(elements)

# define the electrodes
installconductors(conductors)

# --- Recalculate the fields
fieldsol(-1)

winon(xon=0)


VZ = ions.getvz()
EZ = 0.5*ions.mass*VZ**2
if cellnr == 0:
    Ekinold = ekininit
else:
    Ekinold = EZ.mean()/ions.charge

deltaKE = 10e3
R = 90*um
t = np.linspace(0, 2*np.pi, 100)
X = R*np.sin(t)
Y = R*np.cos(t)
zmin = w3d.zmmin

# start and stop time when the mid of the beam passes the unitcell boundaries.
# These values are used to save the history data to a file
Tstart = None
Tend = None

while (zmin < Zend):
    # inject only for dt=zt, so that we can get onto the rising edge of the RF
    if cellnr == 0 and 0 <= top.time < zt:
        top.inject = 1
    else:
        top.inject = 0

    step(10)

    if cellnr == 0:
        V1 = gen_volt(toffset0)(top.time)
        V2 = gen_volt(toffset)(top.time)
        top.pline1 = ("V = {} V  {} V".format(V1, V2))
    else:
        V = gen_volt(toffset)(top.time)
        top.pline1 = ("V = {} V".format(V))

    Z = ions.getz()
    zmin = Z.min()
    zmean = Z.mean()
    if Tstart is None and zmean >= UnitCellStart:
        Tstart = top.time
    if Tend is None and zmean >= UnitCellEnd:
        Tend = top.time

    print("NR: ", len(Z), "minmax", Z.min(), Z.max())
    # create some plots
    KE = ions.getke()
    if len(KE) > 0:
        ions.ppzke(color=red)
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
    pfzx(fill=1, filled=1, plotselfe=2, comp='E', titles=0, cmin=0, cmax=5e6)
    ions.ppzx(color=red, titles=0)
    ptitles("Particles and Fields", "Z [m]", "X [m]", "")
    limits(w3d.zmmin, w3d.zmmax)
    fma()
    ions.ppxy(color=red, titles=0)
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
hppnum()

fma()

# make sure that we are not close the the right edge
zmax = Z.max()
if zmax > w3d.zmmax - 1.1*RFwaferthickness:
    print("WARNING: too close to edge")

print("saving data")
fout = PWpickle.PW("save-{:03d}.pkl".format(cellnr))
fout.x = ions.getx()
fout.y = ions.gety()
fout.z = ions.getz()
fout.vx = ions.getvx()
fout.vy = ions.getvy()
fout.vz = ions.getvz()
fout.close()

VZ = ions.getvz()
EZ = 0.5*ions.mass*VZ**2
Ekin = EZ.mean()/ions.charge
zoffsetout

out = {'cell': cellnr,
       'Ekin': Ekin, 'Ekinold': Ekinold,
       'zoffsetout': zoffsetout, 'zoffset': zoffset,
       'Vesq': Vesq, 'Vesqold': Vesqold,
       'rfgap': rfgap, 'toffset': toffset,
       'time': top.time+mytime}
with open("results{:03d}.json".format(cellnr), "w") as f:
    json.dump(out, f, sort_keys=True)

# save history information, so that we can plot all cells in one plot
t = np.trim_zeros(top.thist, 'b') + mytime
hepsny = ions.hepsny[0]
hepsnz = ions.hepsnz[0]
hep6d = ions.hepsx[0] * ions.hepsy[0] * ions.hepsz[0]
hekinz = 1e-6*0.5*top.aion*amu*ions.hvzbar[0]**2/jperev

u = ions.hvxbar[0]**2 + ions.hvybar[0]**2 + ions.hvzbar[0]**2
hekin = 1e-6 * 0.5*top.aion*amu*u/jperev

hxrms = ions.hxrms[0]
hyrms = ions.hyrms[0]
hrrms = ions.hrrms[0]
hpnum = ions.hpnum[0]

out = np.stack((t, hepsny, hepsnz, hep6d, hekinz, hekin, hxrms, hyrms, hrrms, hpnum))
np.save("hist{:03d}.npy".format(cellnr), out)
print("done")
