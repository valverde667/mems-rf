import warpoptions

#   get input parameters
#   you don't have to put all parameters into the command line,
#   only the ones you want to change from their default vaues
#   you can get a good idea of what these values should be from the ExcelSimulationOfMassSeperation.xlsx
#   or from interactivePlot.ipynb
#   use mathmatica to find out esq_voltage

#   only specify the parameters you want to change from their default values
#   the input will look like:
"""
python single-species-simulation.py --esq_voltage=500 --fraction=.8 --speciesMass=20 --ekininit=15e3
"""
#   Bunch length
warpoptions.parser.add_argument('--bunch_length', dest='Lbunch', type=float, default='50.e-9')
#   voltage on the focusing quads
warpoptions.parser.add_argument('--esq_voltage', dest='Vesq', type=float, default='.01') #850
#   the total number of RF acceleration gaps (must be a multiple of 2)
warpoptions.parser.add_argument('--numRF', dest='numRF', type=int, default='4')
#   the votage on the RF gaps at the peak of the sinusoid
warpoptions.parser.add_argument('--rf_voltage', dest='Vmax', type=float, default='5000') #we can play with this
#   the fraction of the max voltage at which the selected ions cross the gap
warpoptions.parser.add_argument('--fraction', dest='V_arrival', type=float, default='1') #.8
#   the mass of the ions currently being accelerated
warpoptions.parser.add_argument('--mass', dest='current_mass', type=int, default='40')
#   the mass of the ions that the accelerator is built to select for
warpoptions.parser.add_argument('--species_mass', dest='speciesMass', type=int, default='40')
#   the injeciton energy in eV
warpoptions.parser.add_argument('--ekininit', dest='ekininit', type=float, default='10e3')
#   the frequency of the RF
warpoptions.parser.add_argument('--freq', dest='freq', type=float, default='13.56e6') #27e6

import warp as wp
import numpy as np
import geometry
from geometry import ESQ, RF_stack3, Gap, mid_gap
from helper import gitversion
import matplotlib.pyplot as plt
import datetime

wp.w3d.solvergeom = wp.w3d.XYZgeom
wp.top.dt = 5e-11

# --- keep track of when the particles are born
wp.top.ssnpid = wp.nextpid()
wp.top.tbirthpid = wp.nextpid()

#set input parameters
L_bunch = warpoptions.options.Lbunch
speciesMass = warpoptions.options.speciesMass*wp.amu
selectedIons = wp.Species(charge_state=1, name='Ar', mass=warpoptions.options.current_mass*wp.amu, color=wp.green)
ekininit = warpoptions.options.ekininit
Vesq = warpoptions.options.Vesq
numRF = warpoptions.options.numRF
Vmax = warpoptions.options.Vmax #RF voltage
freq = warpoptions.options.freq #RF freq
V_arrival = warpoptions.options.V_arrival #fraction of the total voltage gained across each gap

# --- Invoke setup routine for the plotting (name the cgm output file)

#add a date & timestamp to the cgm file
now = datetime.datetime.now()
datetimestamp = datetime.datetime.now().strftime('%m-%d-%y_%H:%M:%S')

wp.setup(prefix="injected-mass-{}-num-gaps-{}-date-{}-".format(selectedIons.mass/wp.amu,numRF,datetimestamp), cgmlog= 0)

# --- Set basic beam parameters these should be calculated in mathmatica first
emittingradius = .25e-3 #0.41065e-3
ibeaminit = 10e-6 # inital beam current

wp.top.a0 = emittingradius
wp.top.b0 = emittingradius
wp.top.ap0 = 30e-3#divergence angle
wp.top.bp0 = 30e-3#divergence angle
wp.top.vbeam = .0e0
wp.top.emit = 9.45E-7 #what is this? Do we need it; it is not being called anywhere here but maybe warp uses it? -MWG
wp.top.ibeam = ibeaminit
wp.top.ekin = ekininit
wp.top.zion = selectedIons.charge_state
wp.top.lrelativ = False
wp.top.linj_efromgrid = True
wp.derivqty()

# --- Set input parameters describing the 3d simulation
wp.w3d.l4symtry = True
wp.w3d.l2symtry = False

# --- Set boundary conditions

# ---   for field solve
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.neumann

# ---   for particles
wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb
wp.top.prwall = .5*wp.mm #this changes where the particles get absorbed. We should find a more elegant way to do this

# --- Set field grid size, this is the width of the window
wp.w3d.xmmin = -0.02/8.
wp.w3d.xmmax = +0.02/8.
wp.w3d.ymmin = -0.02/8.*1.2
wp.w3d.ymmax = +0.02/8.*1.2
wp.w3d.zmmin = 0.0
wp.w3d.zmmax = 53*wp.mm #18 #changes the length of the gist output window

# set grid spacing, this is the number of mesh elements in one window
wp.w3d.nx = 50.
wp.w3d.ny = 50.
wp.w3d.nz = 180.

if wp.w3d.l4symtry:
    wp.w3d.xmmin = 0.
    wp.w3d.nx /= 2
if wp.w3d.l2symtry or wp.w3d.l4symtry:
    wp.w3d.ymmin = 0.
    wp.w3d.ny /= 2

# --- Select plot intervals, etc.
wp.top.npmax = 300
wp.top.inject = 1  # 2 means space-charge limited injection
wp.top.rinject = 5000 #what is this?
wp.top.npinject = 30  # 300  # needed!! macro particles per time step or cell
wp.top.linj_eperp = False  # Turn on transverse E-fields near emitting surface
wp.top.zinject = wp.w3d.zmmin
wp.top.vinject = 1.0 #what does this do?
print("--- Ions start at: ", wp.top.zinject)

wp.top.nhist = 5  # Save history data every N time step
wp.top.itmomnts[0:4] = [0, 1000000, wp.top.nhist, 0]  # Calculate moments every N steps
# --- Save time histories of various quantities versus z.
wp.top.lhpnumz = True
wp.top.lhcurrz = True
wp.top.lhrrmsz = True
wp.top.lhxrmsz = True
wp.top.lhyrmsz = True
wp.top.lhepsnxz = True
wp.top.lhepsnyz = True
wp.top.lhvzrmsz = True

# --- Set up fieldsolver - 7 means the multigrid solver
solver = wp.MRBlock3D()
wp.registersolver(solver)
solver.mgtol = 1.0  # Poisson solver tolerance, in volts
solver.mgparam = 1.5
solver.downpasses = 2
solver.uppasses = 2

# --- Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.)
wp.package("w3d")
wp.generate()

# --- define voltages
Vground = 0.0e3 #do we actually need to do this? It is not being defined anywhere
VRF = 1000.0 #" "

ESQs = []
RFs = []
ID_ESQ = 100
ID_RF = 201

# --- generates voltage for the RFs
def gen_volt(toffset=0): #0
    """ A sin voltage function with variable offset"""
    def RFvoltage(time):
        return Vmax*np.sin(2*np.pi*freq*(time+toffset))
    return RFvoltage

# --- generates voltage for the ESQs
def gen_volt_esq(Vesq, inverse=False, toffset=0):
    def ESQvoltage(time):
        if inverse:
            return -Vesq#*np.sin(2*np.pi*freq*(time+toffset))
        else:
            return Vesq#*np.sin(2*np.pi*freq*(time+toffset))
    return ESQvoltage

# --- calculate the distances and time offset for the RFs

energies = [ekininit + V_arrival*Vmax*(i+1) for i in range(numRF)] #(i+1) to start at proper 15,000 energy gain
distances = [wp.sqrt(energy*selectedIons.charge/(2*speciesMass))*1/freq for energy in energies] #beta lambda/2
drifti = 10.05*wp.mm #added to start the particles well before the first RF wafer
geometry.pos = -0.5*distances[0] - .5*geometry.RF_gap - geometry.RF_thickness + drifti #starting position of particles
d_mid = geometry.pos + geometry.RF_thickness + .5*geometry.RF_gap #the middle of the first gap
veloinit = np.sqrt(2*ekininit*selectedIons.charge/speciesMass) #initial velocity of particles
t_offset = (d_mid - geometry.RF_thickness - .5*geometry.RF_gap)/veloinit #phase offset to account for moving the injection beam farther to the left
print("TIME OFFSET = {}".format(t_offset)) #not the same as 8e-9
RF_toffset = np.arcsin(V_arrival)/(2*np.pi*freq) - (d_mid + drifti)/veloinit + 8e-9#.5*L_bunch - 1e-9# L_bunch - 8e-9
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
    (rf_bl2, E1), (esq_bl2, E2) = bl2s #calling distances rf_bl2s
    # use first betalamba_half for the RF unit
    RF = RF_stack3(condid=[ID_RF, ID_RF+1, ID_RF+2],
            betalambda_half=rf_bl2 ,voltage=gen_volt(RF_toffset))
    Vpos.append(geometry.pos)

    # scale esq voltages
    voltage = Vesq * E2/ekininit #why do we need to do this? -MWG

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

conductors = wp.sum(ESQs) + wp.sum(RFs) #names all ESQs and RFs conductors in order to feed into warp

velo = np.sqrt(2*ekininit*selectedIons.charge/selectedIons.mass) #used to caluclate tmax
length = geometry.pos + 2.5*wp.cm #2.5mm added to allow particles to completely pass through last RF gap
tmax = length/velo #this is used for the maximum time for timesteps to take place
zrunmax = length #this is used for the maximum distance for timesteps to take place

# define the electrodes
wp.installconductors(conductors)

# --- Recalculate the fields
wp.fieldsol(-1)

solver.gridmode = 0 #makes the fields oscillate properly at the beginning -MWG

# I want contour plots for levels between 0 and 1kV
#contours = range(0, int(Vesq), int(Vesq/10))

wp.winon(xon=0)

# some plots of the geometry
wp.pfzx(fill=1, filled=1, plotphi=0)
wp.fma()
wp.pfzx(fill=1, filled=1, plotphi=1)
wp.fma()

zmin = wp.w3d.zmmin
zmax = wp.w3d.zmmax
zmid = 0.5*(zmax+zmin)

# make a circle to show the beam pipe
R = 1*wp.mm #beam radius
t = np.linspace(0, 2*np.pi, 100)
X = R*np.sin(t)
Y = R*np.cos(t)
deltaKE = 10e3
time = []
numsel = []
KE_select = []
beamwidth=[]
energy_time = []

dist = 2.5*wp.mm
distN = 0

sct = [] #when the particles cross the acceleration gaps
gap_num_select = 0

geometry.mid_gap

while (wp.top.time < tmax and zmax < zrunmax):
    wp.step(10) # each plotting step is 10 timesteps
    time.append(wp.top.time)

    numsel.append(len(selectedIons.getke()))
    KE_select.append(np.mean(selectedIons.getke()))

    wp.top.pline1 = "V_RF: {:.0f}   V_esq: {:.0f}".format(
        gen_volt(RF_toffset)(wp.top.time), gen_volt_esq(Vesq, False, ESQ_toffset)(wp.top.time))
        
    # inject only for 1 ns, so that we can get onto the rising edge of the RF
    if 0*wp.ns < wp.top.time < L_bunch: #changes the beam length
        wp.top.finject[0,selectedIons.jslist[0]] = 1
    else:
        wp.top.inject = 0

    Z = selectedIons.getz()
    if Z.mean() > zmid:
        wp.top.vbeamfrm = selectedIons.getvz().mean()
        solver.gridmode = 0

    # record time when we cross certain points
    if Z.mean() > dist + np.cumsum(distances)[distN]:
        energy_time.append(wp.top.time)
        distN += 1

    zmin = wp.top.zbeam+wp.w3d.zmmin
    zmax = wp.top.zbeam+wp.w3d.zmmax #scales the window length

    # create some plots

    # the instantaneous kinetic energy plot
    KE = selectedIons.getke()
    print(np.mean(KE))
    if len(KE) > 0:
        selectedIons.ppzke(color=wp.blue)
        KEmin, KEmax = KE.min(), KE.max()
        while KEmax-KEmin > deltaKE:
            deltaKE += 10e3
    wp.ylimits(0.95*KEmin, 0.95*KEmin+deltaKE) #is this fraction supposed to match with V_arrival?
    wp.fma()

    # the side view field plot
    wp.pfzx(fill=1, filled=1, plotselfe=True, comp='E', titles=0, cmin=0, cmax=1.2*Vmax/geometry.RF_gap)
    #cmax adjusts the righthand scale of the voltage, not sure how

    #keep track of minimum and maximum birth times of particles
    t_birth_min = selectedIons.tbirth.min()
    t_birth_max = selectedIons.tbirth.max()
    tarray = np.linspace(t_birth_min, t_birth_max, 6)

    #mask to sort particles by birthtime
    mask = []
    for i in range(len(tarray)-1):
        m = (selectedIons.tbirth > tarray[i])*(selectedIons.tbirth < tarray[i+1])
        mask.append(m)

    #plot particles on top of feild plot, sort by birthtime and color them accordingly
    colors = [wp.red, wp.yellow, wp.green, wp.blue, wp.magenta]
    for m, c in zip(mask, colors):
        wp.plp(selectedIons.getx()[m], selectedIons.getz()[m], msize=1.0, color = c)
    wp.limits(zmin, zmax)
    wp.ptitles("Particles and Fields", "Z [m]", "X [m]")
    wp.fma()

    # keep track of when the beam crosses the gaps (for the phase plot at the end)
    if gap_num_select < len(distances) and len(selectedIons.getz() > 0):
        if np.max(selectedIons.getz()) > np.cumsum(distances)[gap_num_select] -0.5*distances[0]:
            sct.append(wp.top.time)
            gap_num_select += 1

    # the head on particle plot
    selectedIons.ppxy(color=wp.red, titles=0) #ppxy (particle plot x horizontal axis, y on vertical axis)
    wp.limits(-R, R)
    wp.ylimits(-R, R)
    wp.plg(Y, X, type="dash")
    wp.fma()

# particles in beam plot
wp.plg(numsel, time, color=wp.blue)
wp.ptitles("Particle Count vs Time", "Time (s)", "Number of Particles")
wp.fma()

# rms envelope plot
wp.hpxrms(color=wp.red, titles=0)
wp.hpyrms(color=wp.blue, titles=0)
wp.hprrms(color=wp.green, titles=0)
wp.ptitles("X(red), Y(blue), R(green)", "Z [m]", "X/Y/R [m]", "")
wp.fma()

# kinetic energy plot
wp.plg(KE_select, time, color=wp.blue)
wp.ptitles("kinetic energy vs time")
wp.fma()

# save history information, so that we can plot all cells in one plot
t = np.trim_zeros(wp.top.thist, 'b')
hepsny = selectedIons.hepsny[0]
hepsnz = selectedIons.hepsnz[0]
hep6d = selectedIons.hepsx[0] * selectedIons.hepsy[0] * selectedIons.hepsz[0]
hekinz = 1e-6*0.5*wp.top.aion*wp.amu*selectedIons.hvzbar[0]**2/wp.jperev

u = selectedIons.hvxbar[0]**2 + selectedIons.hvybar[0]**2 + selectedIons.hvzbar[0]**2
hekin = 1e-6 * 0.5*wp.top.aion*wp.amu*u/wp.jperev

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

#wp.setup(prefix="injected-mass-{}-num-gaps-{}-date{}-".format(selectedIons.mass/wp.amu,numRF,datetimestamp),cgmlog=0)

f= open("injected-mass-"+str(selectedIons.mass/wp.amu)+"-selected-for-mass-"+str(speciesMass/wp.amu)+".txt","a+")

f.write(str(numsel))
f.close
