"""
    RF model for quarter wavelength system
    Created by Grace Woods on 4/1/2018

    Script accesses geometry.py

    Dependent variables in examining beam dynamic dependence on accelerator aperature:
        in geometry.py + quartwavelength.py : r_aperture
        in quartwavelength.py : selectedIons (either H or H2)
"""

# execute script in command line with something along the "python3 -i accelerator.py --ap value"
# where value is the desired aperture radius

import warpoptions

warpoptions.parser.add_argument('--ap' , dest='ap' , type=float , default='.001')

from warp import *
import numpy as np
import geometry_grace as geometry
import matplotlib.pyplot as plt


w3d.solvergeom = w3d.XZgeom # cylindrical symmetry
top.pline1 = "1/4 Wavelength Simulation"

# Parameters available for scans
top.dt = 5e-11
freq = 230e6

#Species should be Hydrogen, Deuterium/Dihydrogen

selectedIons = Species(type=Hydrogen, charge_state=1, name='H', color=green)
#selectedIons = Species(type=Dihydrogen, charge_state=1, name='H2', color=red)

#ap = 1*mm
ap = warpoptions.options.ap

""" File name will be RF-H/H2-freq-Hz.cgm """

setup(prefix="RF-{}-{}-Hz".format(selectedIons.name,int(ap*1e3)))

# --- Set basic beam parameters
emittingradius = ap*0.75 # scaling wrt aperature
ibeaminit = 20e-12
ekininit = 10e3

# --- Make injected species
top.a0 = emittingradius
top.b0 = emittingradius
top.ap0 = 0
top.bp0 = 0
top.vbeam = 0e0
top.emit = 0
top.ibeam = ibeaminit
top.ekin = ekininit
top.zion = selectedIons.charge_state
top.lrelativ = False
derivqty()

# --- Set input parameters describing the 3d simulation
w3d.l4symtry = True
w3d.l2symtry = False

# ---   for field solve
w3d.bound0 = dirichlet # value zero at bndy
w3d.boundnz = dirichlet # value zero at bndy
w3d.boundxy = neumann # derivative zero at bndy

# ---   for particles
top.pbound0 = absorb
top.pboundnz = absorb
top.prwall = .004

# --- Set field grid size
w3d.xmmin = -0.005
w3d.xmmax = +0.005
w3d.ymmin = -0.005
w3d.ymmax = +0.005
w3d.zmmin = 0.0
w3d.zmmax = 0.015

# set grid spacing
w3d.nx = 50*4. # should scale wrt beam radius for resolution
w3d.ny = 50*4. # should scale wrt beam radius for resolution
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
top.rinject = 5000
top.npinject = 300 # needed!! macro particles per time step or cell
top.linj_eperp = False  # Turn on transverse E-fields near emitting surface
top.zinject = 0 # injection point of ions
top.vinject = 1.0
top.linj_efromgrid = true
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

# --- Set up fieldsolver
solver = MultiGrid2DDielectric()
registersolver(solver)

package("w3d")
generate()

ID_RF = 201
geometry.pos = 0
print("starting pos:", geometry.pos)

# RF sinusoid
Vmax = 200 # V
freq = 230e6 # Hz

def gen_volt(toffset=0):
    def RFvoltage(time):
        return Vmax*np.sin(2*np.pi*freq*(time+toffset))
    return RFvoltage

# toffset becomes important when the pulse length is less than 1/4 period of RF
toffset = 0

# rfgap is the drift distance
rfgap = 1.5e-3 - 2*5e-6

# setting up geometry of single RF unit

geometry.Gap(dist=3e-3)

RF = geometry.quart_stack(condid=[ID_RF,ID_RF+1,ID_RF+2,ID_RF+3], r_aperture = ap, rfgap = rfgap, drift = 0.7e-3, voltage=gen_volt(toffset))

geometry.Gap(dist=8e-3)

installconductors(RF)

velo = np.sqrt(2*ekininit*selectedIons.charge/selectedIons.mass)
length = geometry.pos
tmax = length/velo
zrunmax = length + 3e-3

""" dielectric grid defined between conducting plates (manually calculated) """

# r is the transverse radial distance or acceleration gap aperture radius

# r = 1*mm
solver.epsilon[20:50,20:30]*=2.2
solver.epsilon[20:50,35:44]*=2.2


# r = .75*mm
#solver.epsilon[15:50,20:30]*=2.2
#solver.epsilon[15:50,35:44]*=2.2

# r= .50*mm
#solver.epsilon[10:50,20:30]*=2.2
#solver.epsilon[10:50,35:44]*=2.2

# r = .25*mm
#solver.epsilon[5:50,20:30]*=2.2
#solver.epsilon[5:50,35:44]*=2.2

# r = .15*mm
#solver.epsilon[3:50,20:30]*=2.2
#solver.epsilon[3:50,35:44]*=2.2


# recalculate the fields
fieldsol(-1)

winon(xon=0)

# some plots of the geometry
zmin = w3d.zmmin
zmax = w3d.zmmax
zmid = 0.4*(zmax+zmin)

deltaKE = 10e3
time=[]
top.vbeamfrm = 0.001
solver.gridmode = 0
while (top.time < tmax):
    step(1)
    time.append(top.time)

    Volts2 = []
    func2 = gen_volt(toffset)
    Volts2.append(func2(top.time))

    tmp2 = "RF_voltage: {:.0f}".format(Volts2[0])

    top.pline1 = tmp2

    #inject
    if 0 < top.time < (1/230e6):
        top.finject[0,selectedIons.jslist[0]] = 1
    else:
        top.inject = 0

    Z = selectedIons.getz()

    zmin = top.zbeam+w3d.zmmin
    zmax = top.zbeam+w3d.zmmax

    # plot energy vs. z
    KE = selectedIons.getke()
    KEmin, KEmax = KE.min(), KE.max()

    if len(KE) > 0:
        selectedIons.ppzke(color=blue)
        KEmin, KEmax = KE.min(), KE.max()
        while KEmax-KEmin > deltaKE:
            deltaKE += 10e3
    ylimits(0.5*KEmin, 0.5*KEmin+deltaKE)
    fma()

    # plot particles and E field vs z
    pfzx(fill=1, filled=1, plotselfe=2, comp='E', titles=0, cmin=0, cmax=1e6)
    selectedIons.ppzx(color=green, titles=0)
    ptitles("Particles and Fields", "Z [m]", "X [m]", "")
    limits(zmin, zmax)
    fma()

pfzx()
limits(w3d.zmmin,w3d.zmmax)
fma()


""" Saving data to plot with aspectratio.py """
E = selectedIons.getke()
#np.savetxt('H_E_er',E)

""" Energy distribution histogram """
plt.hist(E,100)
plt.xlabel("Ion Energy (eV)")
plt.ylabel("Number of Particles")
plt.show()

""" Plot number of ions passed vs Grid """
Grid = np.arange(np.min(E)-400,np.max(E)+400,10)

num=[]
for G in Grid:
  num.append(sum(G<Energy for Energy in E))

plt.plot(Grid,num,marker='o')
plt.xlabel("Retarding voltage (V)")
plt.ylabel("Number of ions")
plt.show()
