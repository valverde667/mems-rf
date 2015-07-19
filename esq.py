"""
Simplfied ESQ model
"""
from __future__ import print_function

from warp import *
from warp.egun_like import *
from warp.ionization import *
from warp.timedependentvoltage import TimeVoltage

import numpy as np

from geometry import ESQ, RF_stack, Gap, getpos
from helper import gitversion

# which geometry to use 2d or 3d
# w3d.solvergeom = w3d.RZgeom
w3d.solvergeom = w3d.XYZgeom

# define some strings that go into the output file
top.pline1 = "ESQ model"
top.pline2 = " " + gitversion()
top.runmaker = "Arun Persaud (apersaud@lbl.gov)"

# --- Invoke setup routine for the plotting
setup()

# --- Set basic beam parameters
emittingradius = 25*um
ibeaminit = 20e-6
ekininit = 20e3

ions = Species(type=Xenon, charge_state=1, name='Xe')

top.a0 = emittingradius
top.b0 = emittingradius
top.ap0 = .0e0
top.bp0 = .0e0
top.vbeam = .0e0
top.emit = .0e0
top.ibeam = ibeaminit
top.ekin = ekininit
top.aion = ions.type.A
top.zion = ions.charge_state
top.vthz = 0.0
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

# --- Set field grid size
w3d.xmmin = -0.0015/2.
w3d.xmmax = +0.0015/2.
w3d.ymmin = -0.0015/2.
w3d.ymmax = +0.0015/2.
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
solver.mgtol = 1.0  # Poisson solver tolerance, in volts
solver.mgparam = 1.5
solver.downpasses = 2
solver.uppasses = 2

# --- Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.)
package("w3d")
generate()

# --- define voltages
Vground = 0.0e3
Vesq = 100.0
VRF = 1000.0

Gap(500*um)
conductors  = ESQ(voltage=Vesq, condid=[100, 101])
Gap(500*um)
conductors += ESQ(voltage=-Vesq, condid=[102, 103])
Gap(500*um)
conductors += RF_stack(voltage=VRF, condid=[201, 202, 203, 204])
Gap(500*um)
conductors += ESQ(voltage=Vesq, condid=[106, 107])
Gap(500*um)
conductors += ESQ(voltage=-Vesq, condid=[108, 109])
Gap(500*um)
conductors += RF_stack(voltage=VRF, condid=[205, 206, 207, 208])
Gap(500*um)
conductors += ESQ(voltage=Vesq, condid=[112, 113])
Gap(500*um)
conductors += ESQ(voltage=-Vesq, condid=[114, 115])
Gap(500*um)
print("total length", getpos())

velo = np.sqrt(2*ekininit*ions.charge/ions.mass)
length = getpos()
tmax = length/velo
zrunmax = length


# set up time varying fields on the RF electrodes
toffset = 2.5e-9
Vmax = 0.5*10e3
freq = 100e6
def RFvoltage1(time):
    return -Vmax*np.sin(2*np.pi*freq*(time-toffset))

def RFvoltage2(time):
    return -RFvoltage1(time)

RF1 = TimeVoltage(202, voltfunc=RFvoltage1)
RF2 = TimeVoltage(203, voltfunc=RFvoltage2)
RF3 = TimeVoltage(206, voltfunc=RFvoltage1)
RF4 = TimeVoltage(207, voltfunc=RFvoltage2)

# define the electrodes
installconductors(conductors)

# --- Recalculate the fields
fieldsol(-1)

# I want contour plots for levels between 0 and 1kV
contours = range(0, int(Vesq), int(Vesq/10))

winon()

# some plots of the geometry
pfzx(fill=1, filled=1, plotphi=0)
fma()
pfzx(fill=1, filled=1, plotphi=1)
fma()

zmin = w3d.zmmin
zmax = w3d.zmmax
zmid = 0.5*(zmax+zmin)

while (top.time < tmax and zmax < zrunmax):
    step(10)

    tmp = " Voltages: {} {} {} {}".format(RF1.getvolt(top.time), RF2.getvolt(top.time), RF3.getvolt(top.time), RF4.getvolt(top.time))
    top.pline1 = tmp

    # inject only for 1 ns, so that we can get onto the rising edge of the RF
    if 0 < top.time < 1e-9:
        top.inject = 1
    else:
        top.inject = 0

    if top.time > 2.e-9:
        top.vbeamfrm = velo
        solver.gridmode = 0

    zmin = top.zbeam+w3d.zmmin
    zmax = top.zbeam+w3d.zmmax

    # create some plots
    ions.ppzvz(color=red)
    fma()
    pfzx(fill=1, filled=1, plotselfe=2, comp='E', titles=0)
    ions.ppzx(color=red, titles=0)
    ptitles("Particles and Fields","Z [m]", "X [m]", "")
    limits(zmin, zmax)
    fma()
    refresh()

#plot particle vs time
hzpnum()
fma()
#hzepsnxz()
#hzepsnyz()
#hzepsnx()
#hzepsny()
#hzepsnz()
#hzeps6d()
#hztotalke()
#hztotale()
#hzxrms()
#hzyrms()
#hzrrms()
#hzlinechg()
