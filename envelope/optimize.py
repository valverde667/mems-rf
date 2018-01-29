"""Simulate a simple RF-ESQ lattice.

optimize voltages, rf gaps, rf phase.
"""

from warp import *
from helper import gitversion

top.pline2 = "MEQALAC envelope optimizer"
top.pline1 = " " + gitversion()
top.runmaker = "Arun Persaud"

setup()

# --- Set basic beam parameters
emittingradius = 40*um
ibeaminit = 20e-6
ekininit = 40e3

ions = Species(type=Xenon, charge_state=1, name='Xe')

top.a0 = emittingradius
top.b0 = emittingradius
top.ap0 = 14.913e-3
top.bp0 = -14.913e-3
top.vbeam = .0e0
top.emit = 0.77782e-6
top.ibeam = ibeaminit
top.ekin = ekininit
top.aion = ions.type.A
top.zion = ions.charge_state
#top.vthz = 0.0
top.lrelativ = False
derivqty()

beam.vthz = 0.5*ions.vbeam*ions.emit/sqrt(ions.a0*ions.b0)


# geometry

# set up time varying fields on the RF electrodes
Vmax = 5e3
freq = 100e6


def gen_volt(toffset=0):
    def RFvoltage(time):
        return Vmax*np.sin(2*np.pi*freq*(time-toffset))
    return RFvoltage

rfgap = 3*mm

addnewaccl(zs=0, ze=0.5*mm, func=gen_volt())
addnewaccl(zs=rfgap+0, ze=rfgap+0.5*mm, func=gen_volt())


# --- Setup the FODO lattice
# --- These are user created python variables describing the lattice.
hlp = 36.*cm   # half lattice period length
piperad = 3.445*cm  # pipe radius
quadlen = 11.*cm   # quadrupole length

# --- Magnetic quadrupole field gradient - calculated to give sigma0 = 72 degrees.
dbdx = 0.93230106124518164/quadlen

# --- Set up the quadrupoles. Only one lattice period is defined.
# --- This period is repeated to fill all space.
# --- The lattice consists of two quadrupoles, one defocusing, one focusing.
addnewquad(zs=0.5*hlp - quadlen/2.,
           ze=0.5*hlp + quadlen/2.,
           db=+dbdx)
addnewquad(zs=1.5*hlp - quadlen/2.,
           ze=1.5*hlp + quadlen/2.,
           db=-dbdx)


# --- zlatstrt is the start of the periodicity, relative to the quadrupoles position.
top.zlatstrt = 0.

# --- zlatperi is the length of the lattice period, the length of the periodic repeat.
top.zlatperi = 2.0*hlp

# ------------------------------------------------------------------------
# --- The next section sets up and run the envelope equation solver.
# --- Given the initial conditions specified above (a0, b0 etc.),
# --- the envelope package solves the KV envelope equations.
# --- The envelope solution will be used to specify the transverse
# --- shape of the beam where simulation particles will be loaded.

# --- The lattice period length, used to calculate phase advances.
top.tunelen = 2.*hlp

# --- The start and end of the envelope calculation. The initial conditions
# --- are the values at env.zl. Note that zl and zu must cover
# --- the longitudinal extent where the beam particles will be loaded.
# --- dzenv is the step size used in the envelope solver.
env.zl = -2.5*hlp  # z-lower
env.zu = -env.zl  # z-upper
env.dzenv = top.tunelen/100.

# --- Select the envelope solver, do any initialization, and solve the equations.
package("env")
generate()
step()

# --- Make a plot of the resulting envelope solution.
penv()
fma()

# ------------------------------------------------------------------------
# --- Now, set up the parameters describing the 3D simulation.

# --- Specify the time step size. In this case, it is set so that
# --- it takes the specified number of time steps per lattice period.
steps_p_perd = 50
top.dt = (top.tunelen/steps_p_perd)/beam.vbeam

# --- Specify the number of grid cells in each dimension.
w3d.nx = 32
w3d.ny = 32
w3d.nz = 128

# --- Specify the extent of the field solve grid.
w3d.xmmin = -piperad
w3d.xmmax = piperad
w3d.ymmin = -piperad
w3d.ymmax = piperad
w3d.zmmin = -hlp*2
w3d.zmmax = +hlp*2

# --- Specify the boundary conditions on the outer sides of the grid.
# --- Possible values are dirichlet, periodic, and neumann.
w3d.bound0 = dirichlet  # at iz == 0
w3d.boundnz = dirichlet  # at iz == nz
w3d.boundxy = dirichlet  # at all transverse sides

# --- Set the particle boundary conditions at the outer sides of the grid.
# --- Possible values are absorb, periodic, and reflect.
top.pbound0 = absorb
top.pboundnz = absorb
top.pboundxy = absorb

# --- Set the beam pulse length.
# --- Here, it is set to 80% of the grid length.
beam.zimin = w3d.zmmin*0.8
beam.zimax = w3d.zmmax*0.8

# --- Setup the parameters describing how the beam is created.
# --- npmax is the number of simulation particles to create.
top.npmax = 200000

# --- The distribution of the beam.
# --- There are a number of possible values, including "semigauss", "KV", and "WB".
w3d.distrbtn = "semigaus"

# --- The longitudinal velocity distribution of the beam.
w3d.distr_l = "gaussian"

# --- Turn on the "cigar" loading option This imposes a parabolic taper in the line-charge
# --- at the ends of the beam, adjusting the beam envelope to stay matched.
# --- beam.straight specifies the fraction of the beam that is in the middle, without the tapering.
# --- The length of each end will be (1 - beam.straight)/2.
w3d.cigarld = true
beam.straight = 0.5

# --- Set up field solver.
# --- fstype == 0 species the FFT solver.
top.fstype = 0

# --- Optional symmetries can be imposed on the solver.
# --- If l4symtry is true, the fields are calculated in only in transverse
# --- quadrant, and are replicated in the other quadrants.
# --- Note that the particles still occupy all of transverse space.
# --- When the charge is deposited, it would be mapped into the one quadrant.
w3d.l4symtry = false

# --- Setup various diagnostics and plots.
# --- By default, Warp calculates all 1st and 2nd order moments of the particles
# --- as a function of z position.

# --- Warp can save histories of the values of these moments at a selected number
# --- of z-locations relative to the beam frame. These locations are specified
# --- by zwindows. Note that the moments at the center of the window are saved.
# --- The zwindows are given a finite extent since that can also be used to
# --- select particles within the range, for plotting for example.
# --- Note that top.zwindows[:,0] always includes the whole longitudinal extent
# --- and should not be changed.
top.zwindows[:, 1] = [-0.35, -0.3]
top.zwindows[:, 2] = [-0.25, 0.25]
top.zwindows[:, 3] = [0.3, 0.35]

# --- Since it can use a significant amount of memory, only time histories of the
# --- line-charge and vzbar are saved by default. These lines turn on the saving
# --- of time histories of other quantities.
top.lhxrmsz = true
top.lhyrmsz = true
top.lhepsnxz = true
top.lhepsnyz = true
top.lhcurrz = true

# --- nhist specifies the period, in time steps, of saving histories of
# --- the particle moments.
top.nhist = 1

# --- Define some plots to make and the frequency.
# --- zzplalways defines how often certain plots are generated.
# --- zzplalways = [zstart, zend, zperiod, extra_z_values, ...]
top.zzplalways[0:4] = [0., 100000., 2*hlp, 0.]

# --- These specify that the plots ppzxy and ppzvz will be called as specified by zzplalways.
top.ipzxy[-2] = always
top.ipzvz[-2] = always

# --- User defined functions can be called from various points within the time step loop.
# --- This "@callfromafterstep" is a python decorator that says that this function will
# --- be called after every time step.


@callfromafterstep
def runtimeplots(nsteps=steps_p_perd):
    "Make user defined plots, every steps_p_perd steps"
    if top.it % nsteps != 0:
        return
    # --- Create overlaid plots in subframes of the plot window.
    plsys(9)
    pfzx(cellarray=1, contours=0, centering='cell')
    pzxedges(color=red, titles=False)
    plsys(10)
    pfzy(cellarray=1, contours=0, centering='cell')
    pzyedges(color=red, titles=False)
    fma()

    # --- Make plots of the transverse distribution in two zwindows.
    plsys(3)
    ppxy(iw=1)
    limits(-0.02, +0.02, -0.02, +0.02)
    plsys(4)
    ppxxp(iw=1)
    limits(-0.02, +0.02, -0.04, +0.04)
    plsys(5)
    ppxy(iw=3)
    limits(-0.02, +0.02, -0.02, +0.02)
    plsys(6)
    ppxxp(iw=3)
    limits(-0.02, +0.02, -0.04, +0.04)
    fma()

# --- Switch to the w3d package, which runs the 3D PIC model.
# --- The generate command does the initialization, including creating
# --- the particles, doing the initial Poisson solve, and calculating
# --- initial diagnostics and moments.
package("w3d")
generate()

# --- Directly call the user defined function, producing plots of the initial conditions.
runtimeplots()

# --- Run for 50 time steps.
# --- Note that after each time step, the routine runtimeplots will be automatically called.
step(50)

# --- Make various post processing diagnostic plots.
ptitles('Beam X envelope history, in beam frame', 'Lattice periods',
        'Beam frame location (m)', 'Envelope is 2*Xrms')
ppgeneric(gridt=2.*top.hxrmsz[:, :top.jhist, 0], xmin=0.,
          xmax=top.zbeam/(2.*hlp), ymin=w3d.zmmin, ymax=w3d.zmmax)
fma()

ptitles('Beam X normalized emittance history, in beam frame',
        'Lattice periods', 'Beam frame location (m)')
ppgeneric(gridt=top.hepsnxz[:, :top.jhist, 0], xmin=0.,
          xmax=top.zbeam/(2.*hlp), ymin=w3d.zmmin, ymax=w3d.zmmax)
fma()

hpepsnx()
hpepsny(titles=0)
fma()
