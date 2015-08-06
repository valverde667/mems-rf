"""
Source matching
"""
from __future__ import print_function

from warp import *
from warp.egun_like import *
from warp.ionization import *
from warp.timedependentvoltage import TimeVoltage

import numpy as np

from geometry import ESQ, Gap, getpos
from helper import gitversion

# which geometry to use 2d or 3d
# w3d.solvergeom = w3d.RZgeom
w3d.solvergeom = w3d.XYZgeom

# define some strings that go into the output file
top.pline1 = "Source matching model"
top.pline2 = " " + gitversion()
top.runmaker = "Arun Persaud (apersaud@lbl.gov)"

# --- Invoke setup routine for the plotting
setup()

# --- Set basic beam parameters
emittingradius = 846*um
ibeaminit = 225e-6
ekininit = 20e3

ions = Species(type=Xenon, charge_state=1, name='Xe')

top.a0 = emittingradius
top.b0 = emittingradius
top.ap0 = .0e0
top.bp0 = .0e0
top.vbeam = .0e0
top.emit = 8.0e-6*(emittingradius/846e-6)   # 846e-6 from Peter's simulations
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
w3d.zmmax = 0.007

# set grid spacing
w3d.nx = 100.
w3d.ny = 100.
w3d.nz = 700.

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

Gap(500*um)
E1 = ESQ(2.74751E2, [101, 102])
Gap(500*um)
E2 = ESQ(-5.145606E2, [103, 104])
Gap(500*um)
E3 = ESQ(1.55086E3, [104, 105])
Gap(500*um)
E4 = ESQ(-1.968585E2, [106, 107])
Gap(500*um)
E5 = ESQ(5.226E2, [108, 109])
Gap(500*um)
E6 = ESQ(-5.226E2, [110, 111])
Gap(500*um)

velo = np.sqrt(2*ekininit*ions.charge/ions.mass)
length = getpos()
tmax = length/velo
zrunmax = length
Vmax = 2e3
print("total length", length)

# define the electrodes
conductors = E1 + E2 + E3 + E4 + E5 + E6
scraper = ParticleScraper([E1, E2, E3, E4, E5, E6])
installconductors(conductors)

# --- Recalculate the fields
fieldsol(-1)

winon()

# some plots of the geometry
pfzx(fill=1, filled=1, plotphi=0)
fma()
pfzx(fill=1, filled=1, plotphi=1)
fma()

zmin = w3d.zmmin
zmax = w3d.zmmax
zmid = 0.5*(zmax+zmin)

while (top.time < tmax):
    step(10)

    # inject only for 1 ns, so that we can get onto the rising edge of the RF
    if 0 < top.time < 1e-9:
        top.inject = 1
    else:
        top.inject = 0

    # create some plots
    ppzke(color=green)
    fma()
    ppzke(color=green)
    old = limits()
    limits(old[0], old[1], 10e3, 30e3)
    fma()
    pfzx(fill=1, filled=1, plotphi=1, titles=0, cmin=-Vmax, cmax=Vmax)
    ions.ppzx(color=red, titles=0)
    ptitles("Particles and Potentials -- Source matching gap","Z [m]", "X [m]", "")
    limits(zmin, zmax)
    fma()
    refresh()

#plot particle vs time
hzpnum(color=red)
fma()
hpepsz(color=blue)
fma()
hpepsnz(color=blue)
fma()
hpepsr(color=blue)
fma()
hpepsnr(color=blue)
fma()
