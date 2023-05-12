# Warp simulation for the full mems linear accelerator. The simulation
# injects a full DC beam with initial particles injected at the head and tail
# ends to simulate what is done in the lab – 60us continuous injection.
# The script uses a tracker particle as a control to advance the window as the
# particles move further down the acceleration lattice. The tracker particle is
# also used to center a z-window that will calculate beam moments within a fixed
# range in z.

import warpoptions

#   special cgm name - for mass output / scripting
warpoptions.parser.add_argument("--name", dest="name", type=str, default="multions")

#   special cgm path - for mass output / scripting
warpoptions.parser.add_argument("--path", dest="path", type=str, default="")

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
from warp.particles.singleparticle import TraceParticle
from warp.diagnostics.gridcrossingdiags import GridCrossingDiags

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

# Utility definitions
name = warpoptions.options.name

# --- where to store the outputfiles
cgm_name = name
step1path = "."
# step1path = os.getcwd()

# overwrite if path is given by command
if warpoptions.options.path != "":
    step1path = warpoptions.options.path

wp.setup(prefix=f"{step1path}/{cgm_name}")  # , cgmlog= 0)

### read / write functionality #ToDo: move into helper file
basepath = warpoptions.options.path
if basepath == "":
    basepath = f"{step1path}/"
thisrunID = warpoptions.options.name

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
lq = 0.696 * mm
Vq = 0.05 * kV
gap_width = 2 * mm
Vg = 5 * kV
Ng = 8
Fcup_dist = 10 * mm


# Match section Parameters
esq_space = 2.0 * mm
Vq_match = 0.2 * kV
Nq_match = 4


# Operating parameters
freq = 13.6 * MHz
hrf = SC.c / freq
period = 1.0 / freq
emittingRadius = 0.25 * mm
aperture = 0.55 * mm

# Lattice Controls
do_matching_section = False
do_focusing_quads = False
moving_frame = True
moving_win_size = 13.5 * mm

# Beam Paramters
init_E = 7.0 * keV
Tb = 0.1  # eV
div_angle = 3.78 * mrad
emit = 1.336 * mm * mrad
init_I = 10 * uA
Np_injected = 0  # initialize injection counter
Np_max = int(1e5)

dsgn_phase = -np.pi / 2.0

# Specify Species and ion type
beam = wp.Species(type=wp.Argon, charge_state=1, name="Ar+", color=wp.blue)
mass_eV = beam.mass * pow(SC.c, 2) / wp.jperev
# ------------------------------------------------------------------------------
#     Gap Centers
# Here, the gaps are initialized using the design values listed. The first gap
# is started fully negative to ensure the most compact structure. The design
# particle is initialized at z=0 t=0 so that it arrives at the first gap at the
# synchronous phase while the field is rising. However, note that a full period
# of particles is injected before the design particle (tracker particle) is to
# ensure proper particle-particle interactions at the head of the beam.
# ------------------------------------------------------------------------------
# Design phases are specified with the max field corresponding to phi_s=0.
phi_s = np.ones(Ng) * dsgn_phase * 0
phi_s[1:] = np.linspace(-np.pi / 3, -0.0, Ng - 1) * 0
gap_dist = np.zeros(Ng)
E_s = init_E
for i in range(Ng):
    this_beta = mems_utils.beta(E_s, mass_eV)
    this_cent = this_beta * SC.c / 2 / freq
    cent_offset = (phi_s[i] - phi_s[i - 1]) * this_beta * SC.c / freq / twopi
    if i < 1:
        gap_dist[i] = (phi_s[i] + np.pi) * this_beta * SC.c / twopi / freq
    else:
        gap_dist[i] = this_cent + cent_offset

    dsgn_Egain = Vg * np.cos(phi_s[i])
    E_s += dsgn_Egain

gap_centers = np.array(gap_dist).cumsum()


if do_matching_section:
    match_centers = mems_utils.calc_zmatch_sect(lq, esq_space, Nq=Nq_match)
    match_centers -= match_centers.max() + lq / 2 + esq_space

# Calculate phase shift needed to be in resonance
phase_shift = mems_utils.calc_phase_shift(
    phi_s[0],
    freq,
    gap_centers[0] - wp.top.zinject[0],
    init_E,
    beam.mass * pow(SC.c, 2) / wp.jperev,
)
rf_volt = lambda time: Vg * np.cos(twopi * freq * time + phase_shift)

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

# If the frame is moving then the min and max will need to be the length of the
# moving window. If the frame isn't moving, then the grid will need to encapsulate
# the simulation lattice.
if moving_frame:
    if do_matching_section:
        wp.w3d.zmmin = match_centers[0] - lq / 2.0 - esq_space
        wp.w3d.zmmax = 2.0 * moving_win_size + wp.w3d.zmmin
    else:
        wp.w3d.zmmin = -1 * mm
        wp.w3d.zmmax = wp.w3d.zmmin + moving_win_size
else:
    if do_matching_section:
        wp.w3d.zmmin = match_centers[0] - lq / 2.0 - esq_space
        wp.w3d.zmmax = gap_centers[0] + gap_wdith / 2.0 + Fcup_dist
    else:
        wp.w3d.zmmin = -1 * mm
        wp.w3d.zmmax = gap_centers[0] + gap_width / 2.0 + Fcup_dist

wp.w3d.nx = 75
wp.w3d.ny = 75
wp.w3d.nz = 300
lab_center = (wp.w3d.zmmax + wp.w3d.zmmin) / 2.0
dz = (wp.w3d.zmmax - wp.w3d.zmmin) / wp.w3d.nz

# Set boundary conditions
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.periodic

wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb
wp.top.prwall = aperture

# ------------------------------------------------------------------------------
#     Beam and ion specifications
# ------------------------------------------------------------------------------
beam.a0 = emittingRadius
beam.b0 = emittingRadius
beam.ap0 = 0.0
beam.bp0 = 0.0
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

# Calculate and set the weight of particle
Np_inject = int(Np_max / (period / wp.top.dt))
pweight = wp.top.dt * init_I / beam.charge / Np_inject
beam.pgroup.sw[0] = pweight

# Set z-location of injection. This uses the phase shift to ensure rf resonance
# with the trace particle
if do_matching_section:
    wp.top.zinject = wp.w3d.zmmin + 2 * dz
else:
    wp.top.zinject = 0.0

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
zwin_length = beam.vbeam * period / 2
wp.top.zwindows[:, 1] = [lab_center - zwin_length, lab_center + zwin_length]

# Set up lab window for collecting whole beam diagnostics such as current and
# RMS values. Also set the diagnostic for collecting individual particle data
# as they cross.
ilws = []
zdiagns = []
ilws.append(wp.addlabwindow(2.0 * dz))  # Initial lab window
zdiagns.append(ZCrossingParticles(zz=2.0 * dz, laccumulate=1))

# Loop through quad centers in matching section and place diagnostics at center
# point between quads.
if do_matching_section:
    for i in range(Nq_match - 1):
        zloc = (match_centers[i + 1] + match_centers[i]) / 2.0
        ilws.append(wp.addlabwindow(zloc))
        zdiagns.append(ZCrossingParticles(zz=zloc, laccumulate=1))

# Loop through gap_centers and place diagnostics at center point between gaps.
for i in range(Ng - 1):
    zloc = (gap_centers[i + 1] + gap_centers[i]) / 2.0
    ilws.append(wp.addlabwindow(zloc))
    zdiagns.append(ZCrossingParticles(zz=zloc, laccumulate=1))

ilws.append(wp.addlabwindow(gap_centers[-1] + Fcup_dist))
zdiagns.append(ZCrossingParticles(zz=gap_centers[-1] + Fcup_dist, laccumulate=1))

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
tracked_ions = wp.Species(type=wp.Argon, charge_state=0, name="Track", color=wp.red)
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

for i, pa in enumerate(gap_centers):
    print(f"Unit {i} placed at {pa}")

# Create list of conductors to hold the created gaps and ESQs
conductors = []

# Create acceleration gaps.
for i, pos in enumerate(gap_centers):
    zl = pos - 1 * mm
    zr = pos + 1 * mm
    if i % 2 == 0:
        this_lcond = mems_utils.create_wafer(zl, voltage=0.0)
        this_rcond = mems_utils.create_wafer(zr, voltage=rf_volt)
    else:
        this_lcond = mems_utils.create_wafer(zl, voltage=rf_volt)
        this_rcond = mems_utils.create_wafer(zr, voltage=0.0)

    conductors.append(this_lcond)
    conductors.append(this_rcond)

# Create matching section consisting of four quadrupoles.
Vq_match = np.array([97.3053476, -150.58980624, -32.01017919, 142.00741896])
if do_matching_section:
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
    zsize=5.0 * dz,
    zcent=zdiagns[-1].getzz() + 2 * mm,
)

if do_focusing_quads:
    # Calculate ESQ center positions and then install.
    esq_pos = mems_utils.calc_zESQ(gap_centers, gap_centers[-1] + Fcup_dist, lq=lq)

    # Loop through ESQ positions and place ESQs with alternating bias
    Vq_list = np.ones(shape=len(esq_pos))
    Vq_list[::1] *= -Vq  # Alternate signs in list
    for i, pos in enumerate(esq_pos):
        this_ESQ = mems_utils.Mems_ESQ_SolidCyl(pos, Vq_list[i], -Vq_list[i], chop=True)
        this_ESQ.set_geometry(rp=aperture, R=0.68 * aperture, lq=lq)
        this_cond = this_ESQ.generate()
        conductors.append(this_cond)


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
        comp="z",
        plotselfe=1,
        cmin=-1.25 * Vg / gap_width,
        cmax=1.25 * Vg / gap_width,
    )
    wp.ptitles("Ez, Ar+(Blue) and Tracker (Red)", "z (m)", "x (m)")
    wp.ppzx(titles=0, color=wp.blue, msize=1)

    # plot magenta lines to mark the zwindow range
    yy = np.linspace(wp.w3d.xmmin, wp.w3d.xmmax, 10)
    xxl = np.ones(yy.shape[0]) * lab_center + wp.top.zbeam - zwin_length / 2.0
    xxr = np.ones(yy.shape[0]) * lab_center + wp.top.zbeam + zwin_length / 2.0
    wp.plg(yy, xxl, color="magenta")
    wp.plg(yy, xxr, color="magenta")
    print(f"# ------- Tracker At: {tracker.getz()[-1]/mm:.3f} (mm)")

    if lplt_tracker:
        wp.plp(tracker.getx()[-1], tracker.getz()[-1], color=wp.red, msize=3)


# Inject particles for a full-period, then inject the tracker particle and
# continue injection till the tracker particle is at grid center.
while wp.top.time < 1 * period:
    if wp.top.it % 10 == 0:
        wp.window(2)
        plotbeam()
        wp.fma()

    wp.step(1)

# Turn on tracker
tracker.enable()

# Wait for tracker to get to the center of the cell and then start moving frame
while tracker.getz()[-1] < lab_center:
    if wp.top.it % 5 == 0:
        wp.window(2)
        plotbeam(lplt_tracker=True)
        wp.fma()

    wp.step(1)

# Turn off injection once grid starts moving
wp.top.inject = 0
wp.uninstalluserinjection(injection)

# for i in range(Ng - 2):
#     # wp.top.dt = 0.7 * dz / tracker.getvz()[-1]
#     while tracker.getz()[-1] < gap_centers[i + 2]:
#         wp.top.vbeamfrm = tracker.getvz()[-1]
#         if wp.top.it % 5 == 0:
#             wp.window(2)
#             plotbeam(lplt_tracker=True)
#             wp.fma()

#         wp.step(1)

wp.top.vbeamfrm = tracker.getvz()[-1]
# wp.top.dt = 0.7 * dz / tracker.getvz()[-1]
while tracker.getz()[-1] < zdiagns[-1].getzz():
    wp.top.vbeamfrm = tracker.getvz()[-1]
    if wp.top.it % 5 == 0:
        wp.window(2)
        plotbeam(lplt_tracker=True)
        wp.fma()

    wp.step(1)

tracker_fin_time = tracker.gett()[-1]
final_time = tracker_fin_time + 0.25 * period
wp.top.vbeamfrm = 0.0
while wp.top.time < final_time:
    if wp.top.it % 5 == 0:
        wp.window(2)
        plotbeam(lplt_tracker=True)
        wp.fma()

    wp.step(1)

end_time = time.time()
print(f"----- Run Time: {(end_time - start_time)/60.:.4f} (min)")
# ------------------------------------------------------------------------------
#    Simulation Diagnostics
# Here the final diagnostics are computed/extracted. The diagnostic plots are
# made.
# ------------------------------------------------------------------------------
Np_delivered = zdiagns[-1].getvz().shape[0]
Efin = beam.mass * pow(zdiagns[-1].getvz(), 2) / 2.0 / wp.jperev
tfin = zdiagns[-1].gett()
frac_delivered = Np_delivered / Np_injected
frac_lost = abs(Np_delivered - Np_injected) / Np_injected

Data = mems_utils.Data_Ext(ilws, zdiagns, beam, tracker)
Data.grab_data()

# ------------------------------------------------------------------------------
#    Diagnostic Plotting
# Plots are collected and exported to pdf file. Later this will be incorporated
# into the Data class.
# ------------------------------------------------------------------------------
now = datetime.datetime.now()
fmt = "%Y-%m-%d_%H:%M:%S"
save_prefix = datetime.datetime.strftime(now, fmt)

# Make directory in diagnostics for the current run sim
path_diagnostic = os.path.join(os.getcwd(), "diagnostics")
path = os.path.join(path_diagnostic, save_prefix)
os.mkdir(path)

# Create energy history of trace particle
with PdfPages(path + "/trace.pdf") as pdf:
    Etrace = 0.5 * beam.mass * pow(tracker.getvz(), 2) / wp.jperev
    ttrace = tracker.gett()
    ztrace = tracker.getz()

    fig, ax = plt.subplots()
    ax.plot(ttrace / ns, Etrace / keV)
    ax.set_xlabel("Time (ns)")
    ax.set_ylabel("Kinetic Energy (keV)")
    plt.tight_layout()
    pdf.savefig()
    plt.close

    # Plot gaps and fill
    fig, ax = plt.subplots()
    ax.plot(ztrace / mm, Etrace / keV)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("Kinetic Energy (keV)")
    for i, cent in enumerate(gap_centers):
        x1 = cent - 1 * mm
        x2 = cent + 1 * mm
        ax.axvline(x=x1 / mm, c="k", ls="--", lw=1)
        ax.axvline(x=x2 / mm, c="k", ls="--", lw=1)

    plt.tight_layout()
    pdf.savefig()
    plt.close

# Make Energy histograms
with PdfPages(path + "/Ehists.pdf") as pdf:
    # loop through zcrossings and plot histogram of energy and time
    for i in range(len(zdiagns)):
        this_E = 0.5 * beam.mass * pow(zdiagns[i].getvz(), 2) / wp.jperev
        this_t = zdiagns[i].gett()

        fig, ax = plt.subplots(ncols=2)
        Eax, tax = ax[0], ax[1]
        Ecounts, Eedges = np.histogram(this_E, bins=100)
        tcounts, tedges = np.histogram(this_t, bins=100)
        Np = np.sum(Ecounts)

        # Hist the energies
        Eax.bar(
            Eedges[:-1] / keV,
            Ecounts[:] / Np,
            width=np.diff(Eedges[:] / keV),
            edgecolor="black",
            lw="1",
        )
        Eax.set_xlabel("Kinetic Energy (keV)")
        Eax.set_ylabel("Fraction of Particles")
        Eax.set_title(f"z={zdiagns[i].getzz()/mm:.2f} (mm)")

        tax.bar(
            tedges[:-1] / ns,
            tcounts[:] / Np,
            width=np.diff(tedges[:] / ns),
            edgecolor="black",
            lw="1",
        )
        tax.set_xlabel("Time (ns)")
        tax.set_ylabel("Fraction of Particles")
        tax.set_title(f"z={zdiagns[i].getzz()/mm:.2f} (mm)")

        plt.tight_layout()
        pdf.savefig()
        plt.close()


# Loop through lab windows and plot onto pdf.
for key in Data.data_lw_keys:
    with PdfPages(path + "/" + key + ".pdf") as pdf:
        # Loop through the lab window for this measurment and plot
        scale_time, xlabel = Data.scale_factors["time"]
        for i, lw in enumerate(Data.data_lw.keys()):
            fig, ax = plt.subplots()
            this_t = Data.data_lw[lw]["time"]
            this_y = Data.data_lw[lw][key]
            scale_y, ylabel = Data.scale_factors[key]

            # These lines will do some selection to clean up the plot outputs.
            # The first mask will select the additional entries in the time arrays
            # that are 0. These entries are place holders and not actual
            # information. The second mask will handle the 0 calculations. These
            # values arise because the lab windows are calculating values when
            # no beam is present.
            mask_t = this_t > 0.0
            mask_val = abs(this_y) > abs(this_y.max() * 1e-6)
            mask = mask_t & mask_val

            # Grab time and do some processing to eliminate zero elements
            ax.plot(this_t[mask] / scale_time, this_y[mask] / scale_y)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(f"z={zdiagns[i].getzz()/mm:.2f} (mm)")
            plt.tight_layout()
            pdf.savefig()
            plt.close()

# Create history plots for selected moments
def plot_hist(xvals, yvals, xlabel, ylabel, scalex, scaley, xmark=None):
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xmark != None:
        # This will probably only ever be marking the time the tracker particle
        # hits the target
        ax.axvline(x=xmark / scalex, c="k", ls="--", lw=1, label="Tracker Time")
    ax.plot(xvals / scalex, yvals / scaley)
    ax.legend()
    plt.tight_layout()
    return (fig, ax)


with PdfPages(path + "/" + "histories" + ".pdf") as pdf:
    htime = wp.top.thist[: wp.top.jhist + 1]
    hzbeam = wp.top.hzbeam[: wp.top.jhist + 1]
    xmark = tracker.gett()[-1]

    plot_hist(
        htime,
        wp.top.hzrms[0, : wp.top.jhist + 1, 0],
        "time (ns)",
        "z-rms (mm)",
        ns,
        mm,
        xmark=xmark,
    )
    pdf.savefig()
    plt.close()

    plot_hist(
        htime,
        wp.top.hnpsim[0, : wp.top.jhist + 1, 0],
        "time (ns)",
        "Fraction of Np",
        ns,
        Np_injected,
        xmark=xmark,
    )
    pdf.savefig()
    plt.close()

    plot_hist(
        htime,
        wp.top.hxrms[0, : wp.top.jhist + 1, 0],
        "time (ns)",
        "x-rms (mm)",
        ns,
        mm,
        xmark=xmark,
    )
    pdf.savefig()
    plt.close()

    plot_hist(
        htime,
        wp.top.hyrms[0, : wp.top.jhist + 1, 0],
        "time (ns)",
        "y-rms (mm)",
        ns,
        mm,
        xmark=xmark,
    )
    pdf.savefig()
    plt.close()

    plot_hist(
        htime,
        wp.top.hepsz[0, : wp.top.jhist + 1, 0],
        "time (ns)",
        "emit-z (mm-mrad)",
        ns,
        mm * mrad,
        xmark=xmark,
    )
    pdf.savefig()
    plt.close()

    plot_hist(
        htime,
        wp.top.hepsx[0, : wp.top.jhist + 1, 0],
        "time (ns)",
        "emit-x (mm-mrad)",
        ns,
        mm * mrad,
        xmark=xmark,
    )
    pdf.savefig()
    plt.close()

    plot_hist(
        htime,
        wp.top.hepsy[0, : wp.top.jhist + 1, 0],
        "time (ns)",
        "emit-y (mm-mrad)",
        ns,
        mm * mrad,
        xmark=xmark,
    )
    pdf.savefig()
    plt.close()

with PdfPages(path + "/" + "win-histories" + ".pdf") as pdf:
    htime = wp.top.thist[: wp.top.jhist + 1]
    hzbeam = wp.top.hzbeam[: wp.top.jhist + 1]
    xmark = tracker.gett()[-1]

    plot_hist(
        htime,
        wp.top.hzrms[1, : wp.top.jhist + 1, 0],
        "time (ns)",
        "z-rms (mm)",
        ns,
        mm,
        xmark=xmark,
    )
    pdf.savefig()
    plt.close()

    plot_hist(
        htime,
        wp.top.hnpsim[1, : wp.top.jhist + 1, 0],
        "time (ns)",
        "Fraction of Np",
        ns,
        Np_injected,
        xmark=xmark,
    )
    pdf.savefig()
    plt.close()

    plot_hist(
        htime,
        wp.top.hxrms[1, : wp.top.jhist + 1, 0],
        "time (ns)",
        "x-rms (mm)",
        ns,
        mm,
        xmark=xmark,
    )
    pdf.savefig()
    plt.close()

    plot_hist(
        htime,
        wp.top.hyrms[1, : wp.top.jhist + 1, 0],
        "time (ns)",
        "y-rms (mm)",
        ns,
        mm,
        xmark=xmark,
    )
    pdf.savefig()
    plt.close()

    plot_hist(
        htime,
        wp.top.hepsz[1, : wp.top.jhist + 1, 0],
        "time (ns)",
        "emit-z (mm-mrad)",
        ns,
        mm * mrad,
        xmark=xmark,
    )
    pdf.savefig()
    plt.close()

    plot_hist(
        htime,
        wp.top.hepsx[1, : wp.top.jhist + 1, 0],
        "time (ns)",
        "emit-x (mm-mrad)",
        ns,
        mm * mrad,
        xmark=xmark,
    )
    pdf.savefig()
    plt.close()

    plot_hist(
        htime,
        wp.top.hepsy[1, : wp.top.jhist + 1, 0],
        "time (ns)",
        "emit-y (mm-mrad)",
        ns,
        mm * mrad,
        xmark=xmark,
    )
    pdf.savefig()
    plt.close()
