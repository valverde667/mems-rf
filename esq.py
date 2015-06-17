"""
Simplfied ESQ model
"""
from warp import *
from egun_like import *
from ionization import *
import numpy as np

# which geometry to use 2d or 3d
#w3d.solvergeom = w3d.RZgeom
w3d.solvergeom = w3d.XYZgeom


#define some strings that go into the output file
top.pline1     = "ESQ model"
top.pline2     = " "
top.runmaker   = "Arun Persaud (apersaud@lbl.gov)"

# --- Invoke setup routine for the plotting
setup()

# --- Set basic beam parameters
emittingradius =  25*um
ibeaminit = 20e-6
ekininit = 20e3

ions = Species(type=Xenon,charge_state=1,name='Xe')

top.a0       =    emittingradius
top.b0       =    emittingradius
top.ap0      =    .0e0
top.bp0      =    .0e0
top.vbeam    =    .0e0
top.emit     =    .0e0
top.ibeam_s[ions.jslist[0]] = -ibeaminit
top.ekin     =    ekininit
top.aion     =    ions.type.A
top.zion     =    ions.charge_state
top.vthz     =    0.0
top.lrelativ =    false
derivqty()

# --- Set input parameters describing the 3d simulation
top.dt = 1.e-11
w3d.l4symtry = false
w3d.l2symtry = true

# --- Set boundary conditions

# ---   for field solve
w3d.bound0  = dirichlet
w3d.boundnz = neumann
w3d.boundxy = neumann

# ---   for particles
top.pbound0  = absorb
top.pboundnz = absorb
top.prwall   = np.sqrt(2)*1.5*mm/2.0

# --- Set field grid size
w3d.xmmin = -0.0015
w3d.xmmax = +0.0015
w3d.ymmin = -0.0015
w3d.ymmax = +0.0015
w3d.zmmin = 0.0
w3d.zmmax = 0.01

if w3d.l4symtry: w3d.xmmin = 0.
if w3d.l2symtry or w3d.l4symtry: w3d.ymmin = 0.

# set grid spacing
w3d.dx = (w3d.xmmax-w3d.xmmin)/100.
w3d.dy = (w3d.ymmax-w3d.ymmin)/100.
w3d.dz = (w3d.zmmax-w3d.zmmin)/100.

# --- Field grid dimensions - note that nx and ny must be even.
w3d.nx    = 2*int((w3d.xmmax - w3d.xmmin)/w3d.dx/2.)
w3d.xmmax = w3d.xmmin + w3d.nx*w3d.dx
w3d.ny    = 2*int((w3d.ymmax - w3d.ymmin)/w3d.dy/2.)
w3d.ymmax = w3d.ymmin + w3d.ny*w3d.dy
w3d.nz    = int((w3d.zmmax - w3d.zmmin)/w3d.dz)
w3d.zmmax = w3d.zmmin + w3d.nz*w3d.dz

# --- Select plot intervals, etc.
top.npmax    = 300
top.inject   = 1 # 2 means space-charge limited injection
top.rinject = 9999.
top.npinject = 300  # needed!!
top.linj_eperp = true # Turn on transverse E-fields near emitting surface
top.zinject = w3d.zmmin
top.vinject = 1.0

top.nhist = 1 # Save history data every time step
top.itplfreq[0:4]=[0,1000000,25,0] # Make plots every 25 time steps
top.itmomnts[0:4]=[0,1000000,top.nhist,0] # Calculate moments every step
# --- Save time histories of various quantities versus z.
top.lhcurrz  = true
top.lhrrmsz  = true
top.lhxrmsz  = true
top.lhyrmsz  = true
top.lhepsnxz = true
top.lhepsnyz = true
top.lhvzrmsz = true

# --- Set up fieldsolver - 7 means the multigrid solver
top.fstype     = 7
f3d.mgtol      = 1.0 # Poisson solver tolerance, in volts
f3d.mgparam    =  1.5
f3d.downpasses =  2
f3d.uppasses   =  2

# --- Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.)
package("w3d")
generate()

# --- define voltages
voltage_ground     =   0.0e3
voltage_max        =   100.0

# turn this into a function, so that we can add several ESQ one after the other and rotate them easily (or exchange voltages)
# should be create_esq(voltage1, voltage2, zpos)

def esq_wafer(v1, v2, zcenter, condid):
    condidA, condidB = condid

    electrodeA1 = ZCylinder(voltage=v1, radius=75*um, length=500*um,
                            zcent=zcenter, xcent=-337*um, ycent= 125*um, condid=condidA)
    electrodeA2 = ZCylinder(voltage=v1, radius=75*um, length=500*um,
                            zcent=zcenter, xcent=-337*um, ycent=-125*um, condid=condidA)
    electrodeA3 = ZCylinder(voltage=v1, radius=96*um, length=500*um,
                            zcent=zcenter, xcent=-187*um, ycent=0., condid=condidA)
    electrodeA = electrodeA1 + electrodeA2 + electrodeA3

    electrodeB1 = ZCylinder(voltage=v2, radius=75*um, length=500*um,
                            zcent=zcenter, ycent=-337*um, xcent= 125*um, condid=condidB)
    electrodeB2 = ZCylinder(voltage=v2, radius=75*um, length=500*um,
                            zcent=zcenter, ycent=-337*um, xcent=-125*um, condid=condidB)
    electrodeB3 = ZCylinder(voltage=v2, radius=96*um, length=500*um,
                            zcent=zcenter, ycent=-187*um, xcent=0., condid=condidB)
    electrodeB = electrodeB1 + electrodeB2 + electrodeB3

    electrodeC1 = ZCylinder(voltage=v1, radius=75*um, length=500*um,
                            zcent=zcenter, xcent=337*um, ycent= 125*um, condid=condidA)
    electrodeC2 = ZCylinder(voltage=v1, radius=75*um, length=500*um,
                            zcent=zcenter, xcent=337*um, ycent=-125*um, condid=condidA)
    electrodeC3 = ZCylinder(voltage=v1, radius=96*um, length=500*um,
                            zcent=zcenter, xcent=187*um, ycent=0., condid=condidA)
    electrodeC = electrodeC1 + electrodeC2 + electrodeC3

    electrodeD1 = ZCylinder(voltage=v2, radius=75*um, length=500*um,
                            zcent=zcenter, ycent=337*um, xcent= 125*um, condid=condidB)
    electrodeD2 = ZCylinder(voltage=v2, radius=75*um, length=500*um,
                            zcent=zcenter, ycent=337*um, xcent=-125*um, condid=condidB)
    electrodeD3 = ZCylinder(voltage=v2, radius=96*um, length=500*um,
                            zcent=zcenter, ycent=187*um, xcent=0.0, condid=condidB)
    electrodeD = electrodeD1 + electrodeD2 + electrodeD3

    return electrodeA + electrodeB + electrodeC + electrodeD

conductors = esq_wafer(v1=voltage_max, v2=-voltage_max, zcenter=1*mm, condid=[100, 101])
conductors += esq_wafer(v1=-voltage_max, v2=voltage_max, zcenter=2.0*mm, condid=[102, 103])

conductors += esq_wafer(v1=voltage_max, v2=-voltage_max, zcenter=3*mm, condid=[104, 105])
conductors += esq_wafer(v1=-voltage_max, v2=voltage_max, zcenter=4*mm, condid=[106, 107])
conductors += esq_wafer(v1=voltage_max, v2=-voltage_max, zcenter=5*mm, condid=[108, 109])
conductors += esq_wafer(v1=-voltage_max, v2=voltage_max, zcenter=6*mm, condid=[110, 111])
conductors += esq_wafer(v1=voltage_max, v2=-voltage_max, zcenter=7*mm, condid=[112, 113])
conductors += esq_wafer(v1=-voltage_max, v2=voltage_max, zcenter=8*mm, condid=[114, 115])

# define the electrodes
installconductors(conductors)

# --- Recalculate the fields
fieldsol(-1)

#I want contour plots for levels between 0 and 1kV
contours = range(0, int(voltage_max), int(voltage_max/10))

winon()

# this is a small helper function
#fma()
# plot the geometry
#pfzx(fill=1,filled=0, plotphi=0)
# plot the equipotential lines
#pfzx(fill=1,filled=0,contours=contours)
#pfzx(fill=1,filled=1,plotselfe=2,comp='E')
#pfxy(fill=1,filled=1,plotselfe=2,iz=40, comp='E')

#for i in range(1):
#for i in range(20):
#    fma()
#    pfxy(fill=1, filled=1, plotselfe=2, iz=i, comp='E')
#    pfxy(fill=1, filled=0, plotphi=0, iz=i)
#for i in range(20):
#    fma()
#    pfxy(fill=1,filled=0,contours=contours, iz=i)
#

fma()
pfzx(fill=1, filled=1, plotphi=0)
fma()
pfzx(fill=1, filled=1, plotphi=1)

fma()
pfxy(fill=0, filled=1, plotphi=0, iz=10)
fma()
pfxy(fill=0, filled=1, plotphi=1, iz=10)

for i in range(9):
    gun(1, ipstep=1, lvariabletimestep=1, ipsave=300000)
    fma()
    pfzx(fill=1, filled=1, plotselfe=2, comp='E')
    ions.ppzx(color=red, titles=0)
    refresh()

