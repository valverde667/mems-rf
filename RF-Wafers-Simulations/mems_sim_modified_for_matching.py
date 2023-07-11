# Warp simulation to focusing on modeling the matching section. The section is
# placed in negative z with the assumption that the acceleration lattice lattice
# starts at z=0.

import warpoptions


# --Python packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import scipy.constants as SC
import time
import datetime
import os

import pdb

import mems_simulation_utility as mems_utils

mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.minor.top"] = True
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["ytick.right"] = True
mpl.rcParams["ytick.major.right"] = True
mpl.rcParams["ytick.minor.right"] = True
mpl.rcParams["figure.max_open_warning"] = 60

# --Import third-party packages
import warp as wp
from warp.particles.extpart import ZCrossingParticles
from warp.particles.singleparticle import TraceParticle, NoninteractingParticles
from warp.diagnostics.gridcrossingdiags import GridCrossingDiags

wp.setup()
start_time = time.time()

# Define useful constants
mrad = 1e-3
mm = 1e-3
um = 1e-6
nm = 1e-9
kV = 1e3
eV = 1.0
keV = 1e3
ms = 1e-3
us = 1e-6
ns = 1e-9
MHz = 1e6
uA = 1e-6
twopi = 2.0 * np.pi

# ------------------------------------------------------------------------------
#    Functions and Classes
# This section defines the various functions and classes used within the script.
# Eventually this will be moved into a seperate script and then imported via the
# mems package for this system.
# ------------------------------------------------------------------------------


def set_lhistories():
    """Utility function to set all the history flags wanted for sim.

    These flags can be found in the top.v file. I believe some may get autoset
    when others are turned on. I'd rather be explicit and set them all to
    avoid confusion.
    """

    wp.top.lspeciesmoments = True
    wp.top.itlabwn = 1  # Sets how often moment calculations are done
    wp.top.nhist = 1  # Save history data every N time step
    wp.top.itmomnts = wp.top.nhist

    wp.top.lhnpsimz = True
    wp.top.hnpinject = True
    wp.top.lhcurrz = True

    wp.top.lhrrmsz = True
    wp.top.lhxrmsz = True
    wp.top.lhyrmsz = True

    wp.top.lhvxrmsz = True
    wp.top.lhvyrmsz = True
    wp.top.lhvzrmsz = True

    wp.top.lhepsxz = True
    wp.top.lhepsyz = True
    wp.top.lhepsnxz = True
    wp.top.lhepsnyz = True

    wp.top.lsavelostpart = True


# ------------------------------------------------------------------------------
#    Script inputs
# Parameter inputs for running the script. Initially were set as command line
# arguments. Setting to a designated section for better organization and
# overview. Eventually this will be stripped and turn into an input file.
# ------------------------------------------------------------------------------
# Specify conductor characteristics
Vg = 1 * kV
lq = 0.696 * mm
esq_space = 3.0 * mm
Vq_match = np.array([-122.4192, 203.9838, -191.2107, 107.0136])  # Volts
Nq_match = len(Vq_match)
Fcup_dist = 10 * mm

# Operating parameters
freq = 13.6 * MHz
phase_shift = 0.0
hrf = SC.c / freq
period = 1.0 / freq
emittingRadius = 0.25 * mm
aperture = 0.55 * mm

# Beam Paramters
init_E = 7.0 * keV
Tb = 0.1  # eV
div_angle = 3.78 * mrad
emit = 1.336 * mm * mrad
init_I = 10 * uA
Np_injected = 0  # initialize injection counter
Np_max = int(1e5)

# Specify Species and ion type
beam = wp.Species(type=wp.Argon, charge_state=1, name="Ar+", color=wp.blue)
tracked_ions = wp.Species(type=wp.Argon, charge_state=0, name="Track", color=wp.red)
mass_eV = beam.mass * pow(SC.c, 2) / wp.jperev

rf_volt = lambda time: Vg * np.cos(twopi * freq * time + phase_shift)
# ------------------------------------------------------------------------------
#     ESQ Centers
# ESQs are placed according to their length and the spacing provided above.
# If the spacing is d, then the setup goes as d-lq-2d-lq-2d-...-lq-d.
# ------------------------------------------------------------------------------
match_centers = mems_utils.calc_zmatch_sect(lq, esq_space, Nq=Nq_match)
match_centers -= match_centers.max() + lq / 2 + esq_space

# ------------------------------------------------------------------------------
#    Mesh setup
# Specify mesh sizing and time stepping for simulation. The zmmin and zmmax
# settings will set the length of the lab window. This window will move with
# the trace particle. It should be long enough to encompass the beam and
# provide enough spacing between the sim box edges where the field solver is
# screwy.
# ------------------------------------------------------------------------------
# Specify  simulation mesh
wp.w3d.xmmax = 1.5 * mm
wp.w3d.xmmin = -wp.w3d.xmmax
wp.w3d.ymmax = wp.w3d.xmmax
wp.w3d.ymmin = -wp.w3d.ymmax

wp.w3d.zmmin = match_centers[0] - lq / 2.0 - esq_space
wp.w3d.zmmax = match_centers[-1] + lq / 2.0 + Fcup_dist
wp.w3d.nx = 100
wp.w3d.ny = wp.w3d.nx
wp.w3d.nz = 400
dz = (wp.w3d.zmmax - wp.w3d.zmmin) / wp.w3d.nz
lab_center = (wp.w3d.zmmax + wp.w3d.zmmin) / 2.0

# Set boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.periodic

# Reflection boundary used so that particle are not absorbed at injection
wp.top.pbound0 = wp.reflect
wp.top.pboundnz = wp.absorb
wp.top.prwall = aperture

# ------------------------------------------------------------------------------
#     Beam and ion specifications
# ------------------------------------------------------------------------------
beam.a0 = emittingRadius
beam.b0 = emittingRadius
beam.ap0 = div_angle
beam.bp0 = div_angle
beam.ibeam = init_I
beam.vbeam = 0.0
beam.ekin = init_E
beam.emit = emit
vth = np.sqrt(Tb * wp.jperev / beam.mass)

# keep track of when the particles are born
wp.top.inject = 1
wp.top.ibeam_s = init_I
wp.top.ekin_s = init_E
wp.derivqty()
wp.top.dt = 0.7 * dz / beam.vbeam
inj_dz = beam.vbeam * wp.top.dt

# Calculate and set the weight of particles. Weight calculated for desired
# current to be correct out of source.
Np_inject = int(Np_max / (period / wp.top.dt))
pweight = wp.top.dt * init_I / beam.charge / Np_inject
beam.pgroup.sw[0] = pweight
wp.top.zinject = wp.w3d.zmmin


# Create injection scheme. A uniform cylinder will be injected with each time
# step.
def injection():
    """A uniform injection to be called each time step.
    The injection is done each time step and the width of injection is
    calculated above using vz*dt.
    """
    global Np_injected
    global Np_max
    global Np_inject

    # Calculate number to inject to reach max
    Np_injected += Np_inject

    beam.add_uniform_cylinder(
        np=Np_inject,
        rmax=emittingRadius,
        zmin=wp.top.zinject[0],
        zmax=wp.top.zinject[0] + inj_dz,
        vthx=vth,
        vthy=vth,
        vthz=vth,
        vzmean=beam.vbeam,
        vxmean=0.0,
        vymean=0.0,
        vrmax=beam.vbeam * div_angle,
    )


wp.installuserinjection(injection)

# ------------------------------------------------------------------------------
#    History Setup and Conductor generation
# Tell Warp what histories to save and when (in units of iterations) to do it.
# There are also some controls for the solver.
# ------------------------------------------------------------------------------

# Setup Histories and moment calculations
set_lhistories()

# Set the z-windows to calculate moment date at select windows relative to the
# beam frame. top.zwindows[:,0] always includes the who longitudinal extent
# and should not be changed. Here, the window length is calculated for the beam
# extend at the initial condition.
zwin_length = beam.vbeam * period
wp.top.zwindows[:, 1] = [lab_center - zwin_length / 2, lab_center + zwin_length / 2]

# Set up lab window for collecting whole beam diagnostics such as current and
# RMS values. Also set the diagnostic for collecting individual particle data
# as they cross.
ilws = []
zdiagns = []
ilws.append(wp.addlabwindow(2.0 * dz))  # Initial lab window
zdiagns.append(ZCrossingParticles(zz=2.0 * dz, laccumulate=1))

# Loop through quad centers in matching section and place diagnostics at center
# point between quads.
for i in range(Nq_match - 1):
    zloc = (match_centers[i + 1] + match_centers[i]) / 2.0
    ilws.append(wp.addlabwindow(zloc))
    zdiagns.append(ZCrossingParticles(zz=zloc, laccumulate=1))

# Set up fieldsolver
solver = wp.MRBlock3D()
solver.ldosolve = True  # Enable self-fields.
wp.registersolver(solver)
solver.mgtol = 1.0  # Poisson solver tolerance, in volts
solver.mgparam = 1.5
solver.downpasses = 2
solver.uppasses = 2

# Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.)
wp.package("w3d")
wp.generate()
x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh

# Add tracker beam that will record full history. This must be set after
# generate is called.
tracker = TraceParticle(
    js=tracked_ions.js, x=0.0, y=0.0, z=wp.top.zinject[0], vx=0.0, vy=0.0, vz=beam.vbeam
)

# For unknown reasons, the tracer cannot be placed arbitrarily in the injection
# scheme. Thus, it is created early on, disabled, then renabled at the desired
# point in injection.
tracker.disable()

# Recalculate fields and conductor information every time step. Needed for
# oscillating fields in a moving frame.
solver.gridmode = 0


# Create list of conductors to hold the created gaps and ESQs
conductors = []

# Create matching section consisting of four quadrupoles.
for i, pos in enumerate(match_centers):
    this_zc = pos
    this_Vq = Vq_match[i]

    this_ESQ = mems_utils.Mems_ESQ_SolidCyl(this_zc, this_Vq, -this_Vq, chop=True)
    this_ESQ.set_geometry(rp=aperture, R=1.3 * aperture, lq=lq)
    this_cond = this_ESQ.generate()
    conductors.append(this_cond)

# Create and intialize the scraper that will collect lost particle data.
aperture_wall = wp.ZCylinderOut(
    radius=aperture, zlower=-wp.top.largepos, zupper=wp.top.largepos
)
Fcup = wp.Box(
    xsize=wp.top.largepos,
    ysize=wp.top.largepos,
    zsize=wp.w3d.zmmax - 2.0 * dz,
    zcent=match_centers[-1] + lq / 2 + Fcup_dist,
)

for cond in conductors:
    wp.installconductors(cond)


scraper = wp.ParticleScraper(aperture_wall, lcollectlpdata=True)
scraper.registerconductors(conductors)
scraper.registerconductors(Fcup)
# Recalculate the fields
wp.fieldsol(-1)

# Create cgm windows for plotting
wp.winon(winnum=2, suffix="pzx", xon=False)
wp.winon(winnum=3, suffix="pxy", xon=False)

# ------------------------------------------------------------------------------
#    Injection and advancement
# Here the particles are injected. The first while-loop injects particles for
# a half-period in time. Afterward, a trace particle is added and then the
# second while-loop finishes the injection so that a full period of beam is
# injected. The particles are advanced. After the trace (design) particle has
# arrived at the Fcup, the particle advancement is done for 2 more RF periods
# to capture remaining particles.
# TODO: Devise better injection and advancment scheme rather than multiple for-loops.
# ------------------------------------------------------------------------------


# Def plotting routine to be called in stepping
def plotbeam(lplt_tracker=False):
    """Plot particles, conductors, and Ez contours as particles advance."""
    wp.pfzx(
        plotsg=0,
        cond=0,
        fill=1,
        filled=1,
        condcolor="black",
        titles=0,
        contours=80,
        cmin=-1 * kV,
        cmax=1 * kV,
    )
    wp.ptitles("Ez, Ar+(Blue) and Tracker (Red)", "z (m)", "x (m)")
    wp.ppzx(titles=0, color=wp.blue, msize=1)

    # plot magenta lines to mark the zwindow range
    yy = np.linspace(wp.w3d.xmmin, wp.w3d.xmmax, 10)
    xxl = np.ones(yy.shape[0]) * lab_center + wp.top.zbeam - zwin_length / 2.0
    xxr = np.ones(yy.shape[0]) * lab_center + wp.top.zbeam + zwin_length / 2.0
    wp.plg(yy, xxl, color="magenta")
    wp.plg(yy, xxr, color="magenta")

    if lplt_tracker:
        wp.plp(tracker.getx()[-1], tracker.getz()[-1], color=wp.red, msize=3)


# Inject particles for a full-period, then inject the tracker particle and
# continue injection till the tracker particle is at grid center.
tracker.enable()
wp.window(2)
plotbeam()
wp.fma()
wp.step()

# Estimate the stop time for beam to fill simulation window.
stop_time = (wp.w3d.zmmax - wp.w3d.zmmin) / beam.vbeam
while wp.top.time < stop_time + 10 * wp.top.dt:
    wp.window(2)
    if wp.top.it % 10 == 0:
        plotbeam(lplt_tracker=True)
        wp.fma()

    wp.step(1)
end_time = time.time()

# ------------------------------------------------------------------------------
#    Diagnostic Viewing
# Make various plots and grab histories of various data.
# ------------------------------------------------------------------------------
zmnt_ind = np.argmin(abs(wp.w3d.zmesh))
hxrms = wp.top.xrmsz[:zmnt_ind, 0]
hyrms = wp.top.yrmsz[:zmnt_ind, 0]
hxxpbar = wp.top.xxpbarz[:zmnt_ind, 0]
hyypbar = wp.top.yypbarz[:zmnt_ind, 0]
zhist = wp.top.zmntmesh[:zmnt_ind] + abs(z.min())

grad = wp.getselfe(comp="x")[41, 40, :] / wp.w3d.dx
kappa = wp.echarge * grad / 2.0 / init_E / wp.jperev

fig, ax = plt.subplots()
ax.set_xlabel("s (mm)")
ax.set_ylabel(r"$\kappa (s)\, (\mathrm{m}^{-2})$")
plt.plot(z / mm, kappa)

fig, ax = plt.subplots()
ax.set_title(r"Warp Envelope for $r_x$ and $r_y$")
ax.set_xlabel("s (mm)")
ax.set_ylabel("Transverse Position (mm)")
ax.plot(zhist / mm, 2 * hxrms / mm, label=r"2rms-$r_x(s)$")
ax.plot(zhist / mm, 2 * hyrms / mm, label=r"2rms-$r_y(s)$")
ax.legend()

fig, ax = plt.subplots()
ax.set_title(r"Warp $d(\mathrm{envelope})/ds$ Solutions")
ax.set_xlabel("s (mm)")
ax.set_ylabel("Transverse Angle (mrad)")
ax.plot(zhist / mm, 2 * hxxpbar / hxrms / mrad, label=r"2rms-$rp_x(s)$")
ax.plot(zhist / mm, 2 * hyypbar / hyrms / mrad, label=r"2rms-$rp_y(s)$")
ax.legend()
plt.show()

data = np.vstack((hxrms, hyrms, hxxpbar, hyypbar, zhist))
np.save("mems_matching_sect_data", data)
print(f"Total Run Time: {(end_time - start_time)/60:2f} (min)")
