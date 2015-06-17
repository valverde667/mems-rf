"""
Simplfied ESQ model
"""
from warp import *
from egun_like import *
from ionization import *
import numpy as np

from geometry import ESQ
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
top.ibeam_s[ions.jslist[0]] = ibeaminit
top.ekin = ekininit
top.aion = ions.type.A
top.zion = ions.charge_state
top.vthz = 0.0
top.lrelativ = False
derivqty()

# --- Set input parameters describing the 3d simulation
top.dt = 1.e-11
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
w3d.zmmax = 0.01

if w3d.l4symtry:
    w3d.xmmin = 0.
if w3d.l2symtry or w3d.l4symtry:
    w3d.ymmin = 0.

# set grid spacing
w3d.dx = (w3d.xmmax-w3d.xmmin)/100.
w3d.dy = (w3d.ymmax-w3d.ymmin)/100.
w3d.dz = (w3d.zmmax-w3d.zmmin)/1000.

# --- Field grid dimensions - note that nx and ny must be even.
w3d.nx = 2*int((w3d.xmmax - w3d.xmmin)/w3d.dx/2.)
w3d.xmmax = w3d.xmmin + w3d.nx*w3d.dx
w3d.ny = 2*int((w3d.ymmax - w3d.ymmin)/w3d.dy/2.)
w3d.ymmax = w3d.ymmin + w3d.ny*w3d.dy
w3d.nz = int((w3d.zmmax - w3d.zmmin)/w3d.dz)
w3d.zmmax = w3d.zmmin + w3d.nz*w3d.dz

# --- Select plot intervals, etc.
top.npmax = 300
top.inject = 1  # 2 means space-charge limited injection
top.rinject = 9999.
top.npinject = 300  # needed!!
top.linj_eperp = True  # Turn on transverse E-fields near emitting surface
top.zinject = w3d.zmmin
top.vinject = 1.0

top.nhist = 1  # Save history data every time step
top.itplfreq[0:4] = [0, 1000000, 25, 0]  # Make plots every 25 time steps
top.itmomnts[0:4] = [0, 1000000, top.nhist, 0]  # Calculate moments every step
# --- Save time histories of various quantities versus z.
top.lhcurrz = True
top.lhrrmsz = True
top.lhxrmsz = True
top.lhyrmsz = True
top.lhepsnxz = True
top.lhepsnyz = True
top.lhvzrmsz = True

# --- Set up fieldsolver - 7 means the multigrid solver
top.fstype = 7
f3d.mgtol = 1.0  # Poisson solver tolerance, in volts
f3d.mgparam = 1.5
f3d.downpasses = 2
f3d.uppasses = 2

# --- Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.)
package("w3d")
generate()

# --- define voltages
Vground = 0.0e3
Vesq = 100.0

conductors = ESQ(voltage=Vesq, zcenter=1*mm, condid=[100, 101])
conductors += ESQ(voltage=-Vesq, zcenter=2.0*mm, condid=[102, 103])
conductors += ESQ(voltage=Vesq, zcenter=3*mm, condid=[104, 105])
conductors += ESQ(voltage=-Vesq, zcenter=4*mm, condid=[106, 107])
conductors += ESQ(voltage=Vesq, zcenter=5*mm, condid=[108, 109])
conductors += ESQ(voltage=-Vesq, zcenter=6*mm, condid=[110, 111])
conductors += ESQ(voltage=Vesq, zcenter=7*mm, condid=[112, 113])
conductors += ESQ(voltage=-Vesq, zcenter=8*mm, condid=[114, 115])

# define the electrodes
installconductors(conductors)

# --- Recalculate the fields
fieldsol(-1)

# I want contour plots for levels between 0 and 1kV
contours = range(0, int(Vesq), int(Vesq/10))

winon()

fma()
pfzx(fill=1, filled=1, plotphi=0)
fma()
pfzx(fill=1, filled=1, plotphi=1)

fma()
pfxy(fill=0, filled=1, plotphi=0, iz=50)
fma()
pfxy(fill=0, filled=1, plotphi=1, iz=74)
fma()
pfxy(fill=0, filled=1, plotphi=1, iz=123)
fma()
pfxy(fill=0, filled=1, plotphi=1, iz=124)
fma()
pfxy(fill=0, filled=1, plotphi=1, iz=125)

for i in range(2):
    gun(1, ipstep=1, lvariabletimestep=1, ipsave=300000)
    fma()
    pfzx(fill=1, filled=1, plotselfe=2, comp='E')
    ions.ppzx(color=red, titles=0)
    refresh()
