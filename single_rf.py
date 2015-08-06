"""
RF-gap model
"""
from __future__ import print_function

from warp import *
from warp.egun_like import *
from warp.ionization import *
from warp.timedependentvoltage import TimeVoltage

import numpy as np

from geometry import RF_stack, Aperture, Gap, getpos
from helper import gitversion

# which geometry to use 2d or 3d
# w3d.solvergeom = w3d.RZgeom
w3d.solvergeom = w3d.XYZgeom

# define some strings that go into the output file
top.pline1 = "RF-gap model"
top.pline2 = " " + gitversion()
top.runmaker = "Arun Persaud (apersaud@lbl.gov)"

# --- Invoke setup routine for the plotting
setup()

ions = Species(type=Xenon, charge_state=1, name='Xe')

# --- Set basic beam parameters
emittingradius = 25*um
# use 20uA at 20keV and then keep the current and beamlength constant
beampulse = 1*ns
ibeaminit = 20e-6  # change these below iff needed
ekininit = 20e3   # change these below iff needed
velo = np.sqrt(2*ekininit*ions.charge/ions.mass)
L = velo*beampulse

ekininit = 20e3
velo = np.sqrt(2*ekininit*ions.charge/ions.mass)
Lnew = velo*beampulse

# update to new values
beampulse *= L/Lnew
ibeaminit *= Lnew/L

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
w3d.zmmax = 0.00232

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
VRF = 0.0

conductors = Aperture(0, 101, width=50*um)
Gap(500*um)
conductors += RF_stack(voltage=VRF, condid=[201, 202, 203, 204])
Gap(500*um)
conductors += Aperture(0, 101, width=50*um)
print("total length", getpos())

velo = np.sqrt(2*ekininit*ions.charge/ions.mass)
length = getpos()
tmax = length/velo
zrunmax = length

# set up time varying fields on the RF electrodes
toffset = tmax*0.5+0.5*beampulse  # offset so that maximum would be when beam
                    # arrives in the gap
Vmax = 5e3
offset = 30./360.  # we don't want to actually be at the maximum, but
                   # 30 degree off for some focusing

freq1 = (1/4.-offset)/(522e-6/velo)
velo2 = np.sqrt(2*(ekininit+2*Vmax*np.cos(offset))*ions.charge/ions.mass)
freq2 = (1/4.+offset)/(522e-6/velo2)

def RFvoltage1(time):
    return Vmax*np.cos(2*np.pi*freq1*(time-toffset)-2*np.pi*offset)

def RFvoltage2(time):
    return -Vmax*np.cos(2*np.pi*freq2*(time-toffset)-2*np.pi*offset)

RF1 = TimeVoltage(202, voltfunc=RFvoltage1)
RF2 = TimeVoltage(203, voltfunc=RFvoltage2)

# define the electrodes
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

# save data to plot the bunch length
blength = []
@callfromafterstep
def myhist():
    global blength
    Z = ions.getz()
    if len(Z) >0:
        blength.append([top.time, Z.ptp()])

while (top.time < tmax and zmax < zrunmax):
    step(10)

    tmp = " Voltages: {:.0f} V  {:.0f} V\n".format(RF1.getvolt(top.time), RF2.getvolt(top.time))
    tmp += " Freq: {:.2f} MHz  {:.2f} MHz\n".format(freq1*1e-6, freq2*1e-6)
    top.pline1 = tmp

    # inject only for 1 ns, so that we can get onto the rising edge of the RF
    if 0 < top.time < beampulse:
        top.inject = 1
    else:
        top.inject = 0

#    if top.time > 2.e-9:
#        top.vbeamfrm = velo
#        solver.gridmode = 0
#
#    zmin = top.zbeam+w3d.zmmin
#    zmax = top.zbeam+w3d.zmmax

    # create some plots
    ppzke(color=green)
    fma()
    ppzke(color=green)
    old = limits()
    limits(old[0], old[1], ekininit*0.95, (ekininit+12e3)*1.05)
    fma()
    pfzx(fill=1, filled=1, plotphi=1, titles=0, cmin=-Vmax, cmax=Vmax)
    ions.ppzx(color=red, titles=0)
    ptitles("Particles and Potentials -- Single RF gap","Z [m]", "X [m]", "")
    limits(zmin, zmax)
    fma()
    refresh()

#plot particle vs time
blength = np.array(blength)
T = blength[:, 0]
L = blength[:, 1]
plg(L, T)
ptitles("Bunch length","time","L","")
fma()
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
