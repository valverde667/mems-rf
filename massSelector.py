import warpoptions

#   get input parameters
#   you don't have to put all parameters into the command line,
#   only the ones you want to change from their default vaues
#   you can get a good idea of what these values should be from the ExcelSimulationOfMassSeperation.xlsx
#   or from interactivePlot.ipynb
#   use mathmatica to find out esq_voltage

#   the input will look like:
"""
python massSelector.py --esq_voltage=500 --fraction=.8 --selectedMass=20 --ekininit=15e3
"""
#   voltage on the focusing quads
warpoptions.parser.add_argument('--esq_voltage', dest='Vesq', type=float, default='813')
#   the total number of RF acceleration gaps (must be a multiple of 2)
warpoptions.parser.add_argument('--numRF', dest='numRF', type=int, default='40')
#   the votage on the RF gaps at the peak of the sinusoid
warpoptions.parser.add_argument('--rf_voltage', dest='Vmax', type=float, default='500')
#   the fraction of the max voltage at which the selected ions cross the gap
warpoptions.parser.add_argument('--fraction', dest='V_arrival', type=float, default='.8')
#   the mass of the ions currently being accelerated
warpoptions.parser.add_argument('--mass', dest='current_mass', type=int, default='13')
#   the mass of the ions that the accelerator is built to select for
warpoptions.parser.add_argument('--selected_mass', dest='selectedMass', type=int, default='14')
#   the injeciton energy in eV
warpoptions.parser.add_argument('--ekininit', dest='ekininit', type=float, default='20e3')
#   the frequency of the RF
warpoptions.parser.add_argument('--freq', dest='freq', type=float, default='15e6')

from warp import *
import numpy as np
import geometry
from geometry import Aperture, ESQ, RF_stack3, Gap
from helper import gitversion
import matplotlib.pyplot as plt
import TransitTimeEffectCalculator as tte

w3d.solvergeom = w3d.XYZgeom
top.dt = 5e-11

#set input parameters
selectedMass = warpoptions.options.selectedMass*amu
selectedIons = Species(charge_state=1, name='C14', mass=warpoptions.options.current_mass*amu, color=green)
ekininit = warpoptions.options.ekininit
Vesq = warpoptions.options.Vesq
numRF = warpoptions.options.numRF
Vmax = warpoptions.options.Vmax #RF voltage
freq = warpoptions.options.freq #RF freq #change freq in TTE
V_arrival = warpoptions.options.V_arrival #the fraction of the total voltage gained across each gap

freq = (1/3.4e-2)*np.sqrt(ekininit*selectedIons.charge/(2*selectedMass))#automaticaly set first distance
freq_multiplier = 3 # multiplies the frequency by a constant while leaveing the geometry the same to improve mass separation

# --- Invoke setup routine for the plotting
setup(prefix="injected-mass-{}-selected-for-mass-{}-num-gaps-{}".format(selectedIons.mass/amu, selectedMass/amu, numRF),cgmlog=0)

# --- Set basic beam parameters these should be calculated in mathmatica first
emittingradius = 0.41065e-3
ibeaminit = 1e-6

top.a0 = emittingradius
top.b0 = emittingradius
top.ap0 = .48743e-3
top.bp0 = -.48743e-3
top.vbeam = .0e0
top.emit = 9.45E-7
top.ibeam = ibeaminit
top.ekin = ekininit
top.zion = selectedIons.charge_state
top.lrelativ = False
top.linj_efromgrid = True
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
top.prwall = 1*mm

# --- Set field grid size, this is the width of the window
w3d.xmmin = -0.02/8.
w3d.xmmax = +0.02/8.
w3d.ymmin = -0.02/8.
w3d.ymmax = +0.02/8.
w3d.zmmin = 0.0
w3d.zmmax = 18*mm

# set grid spacing, this is the number of mesh elements in one window
w3d.nx = 50.
w3d.ny = 50.
w3d.nz = 180.

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
top.npinject = 30  # 300  # needed!! macro particles per time step or cell
top.linj_eperp = False  # Turn on transverse E-fields near emitting surface
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
VRF = 1000.0

ESQs = []
RFs = []
ID_ESQ = 100
ID_RF = 201

# --- generates voltage for the RFs
def gen_volt(toffset=0):
    """ A sin voltage function with variable offset"""
    def RFvoltage(time):
        return Vmax*np.sin(2*np.pi*freq*(time+toffset)*freq_multiplier)
    return RFvoltage

# --- generates voltage for the ESQs
def gen_volt_esq(Vesq, inverse=False, toffset=0):
    def ESQvoltage(time):
        if inverse:
            return -Vesq#*np.sin(2*np.pi*freq*(time+toffset))
        else:
            return Vesq#*np.sin(2*np.pi*freq*(time+toffset))
    return ESQvoltage

# --- calculate the time ofset for the RFs
energies = [ekininit + V_arrival*Vmax*i for i in range(numRF)]
distances = []
for energy in energies:
    distances.append(sqrt(energy*selectedIons.charge/(2*selectedMass))*1/freq)
geometry.pos = -0.5*distances[0] - .5*geometry.RF_gap - geometry.RF_thickness
d_mid = distances[0]*.5 - .5*geometry.RF_gap - geometry.RF_thickness
RF_toffset = np.arcsin(V_arrival)/(2*np.pi*freq*freq_multiplier)-d_mid/np.sqrt(2*ekininit*selectedIons.charge/selectedMass) - .5*ns

ESQ_toffset = 0

Vpos = []

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
    gaplength = gaplength/2-geometry.ESQ_gap/2-geometry.ESQ_wafer_length
    assert gaplength > 0
    Gap(gaplength)
    E1 = ESQ(voltage=gen_volt_esq(Vesq, False, ESQ_toffset), condid=[ID_ESQ, ID_ESQ+1])
    Gap(geometry.ESQ_gap)
    E2 = ESQ(voltage=gen_volt_esq(Vesq, True, ESQ_toffset), condid=[ID_ESQ+2, ID_ESQ+3])
    Gap(gaplength)

    ESQs.append(E1)
    ESQs.append(E2)
    RFs.append(RF)
    ID_ESQ += 4
    ID_RF += 3

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
R = 1*mm # beam radius
t = np.linspace(0, 2*np.pi, 100)
X = R*np.sin(t)
Y = R*np.cos(t)
deltaKE = 10e3
time = []
numsel = []
KE_select = []
beamwidth=[]
energy_time = []

dist = 2.5*mm
distN = 0

sct = [] #when the particles cross the acceleration gaps
gap_num_select = 0

while (top.time < tmax and zmax < zrunmax):
    step(10) # each plotting step is 10 timesteps
    time.append(top.time)

    numsel.append(len(selectedIons.getke()))
    KE_select.append(np.mean(selectedIons.getke()))

    top.pline1 = "V_RF: {:.0f}   V_esq: {:.0f}".format(
        gen_volt(RF_toffset)(top.time), gen_volt_esq(Vesq, False, ESQ_toffset)(top.time))

    # inject only for 1 ns, so that we can get onto the rising edge of the RF
    if 0*ns < top.time < 1e-9:
        top.finject[0,selectedIons.jslist[0]] = 1
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

    # the instantanious kinetic energy plot
    KE = selectedIons.getke()
    print(np.mean(KE))
    if len(KE) > 0:
        selectedIons.ppzke(color=blue)
        KEmin, KEmax = KE.min(), KE.max()
        while KEmax-KEmin > deltaKE:
            deltaKE += 10e3
    ylimits(0.95*KEmin, 0.95*KEmin+deltaKE)
    fma()

    # the side view particle and field plot
    pfzx(fill=1, filled=1, plotselfe=True, comp='E', titles=0,
         cmin=0, cmax=1.2*Vmax/geometry.RF_gap)
    selectedIons.ppzx(color=green, titles=0)
    ptitles("Particles and Fields", "Z [m]", "X [m]", "")
    limits(zmin, zmax)
    fma()

    # keep track of when the beam crosses the gaps (for the phase plot at the end)
    if gap_num_select < len(distances) and len(selectedIons.getz() > 0):
        if np.max(selectedIons.getz()) > np.cumsum(distances)[gap_num_select] -0.5*distances[0]:
            sct.append(top.time)
            gap_num_select += 1

    # the head on particle plot
    selectedIons.ppxy(color=red, titles=0)
    limits(-R, R)
    ylimits(-R, R)
    plg(Y, X, type="dash")
    fma()

# particles in beam plot
plg(numsel, time, color=blue)
ptitles("Particle Count vs Time", "Time (s)", "Number of Particles")
fma()

# rms envelope plot
hpxrms(color=red, titles=0)
hpyrms(color=blue, titles=0)
hprrms(color=green, titles=0)
ptitles("X(red), Y(blue), R(green)", "Z [m]", "X/Y/R [m]", "")
fma()

# kinetic energy plot
plg(KE_select, time, color=blue)
ptitles("kinetic energy vs time")
fma()

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
np.save("esqhist.npy", out)

# # --- makes the sin wave phase plot at the end can un-comment if you are interested in this
# s_phases = ((np.array(sct) + RF_toffset) % (1/freq))
# offset = -.07*Vmax # move down each wave so they are not on top of eachother
#
# # the sin wave
# T = np.linspace(0, 1/freq, 1000)
# V = Vmax*np.sin(2*np.pi*freq*T)
#
# # the dots
# for i, phase in enumerate(s_phases, 0):
#     o_Y = Vmax*np.sin(2*np.pi*phase*freq)
#     plt.plot(T, V+i*offset, "k-")
#     plt.plot(phase, o_Y+i*offset, 'bo')
# plt.show()

f= open("injected-mass-"+str(selectedIons.mass/amu)+"-selected-for-mass-"+str(selectedMass/amu)+".txt","a+")
f.write(str(numsel))
f.close
