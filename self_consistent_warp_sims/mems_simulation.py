# Warp simulation for the full mems linear accelerator. The simulation
# injects a full DC beam with initial particles injected at the head and tail
# ends to simulate what is done in the lab – 60us continuous injection.
# The script uses a tracker particle as a control to advance the window as the
# particles move further down the acceleration lattice. The tracker particle is
# also used to center a z-window that will calculate beam moments within a fixed
# range in z.

import numpy as np
import h5py
import pickle
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches
import seaborn as sns
import scipy.constants as SC
import time
import datetime
import os
import pdb

import mems_simulation_utility as mems_utils

import warp as wp
from warp.particles.extpart import ZCrossingParticles
from warp.particles.singleparticle import TraceParticle
from warp.diagnostics.gridcrossingdiags import GridCrossingDiags

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

wp.setup()

start_time = time.time()

# Define useful constants
mrad = 1e-3
cm = 1e-2
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


def reset_tracker(particle_object, index, xpos):
    """Manually reset the tracker object index value to xpos"""
    tracker.spx[0].data()[index] = xpos
    tracker.spy[0].data()[index] = xpos


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
Vq = [-193, 193, -310, 310, -375, 375, -415, 415]
gap_width = 2 * mm
Vg_scale = 1 + 0.031  # Scale plate voltage so that Vg is reached on-axis
Vg = 6 * Vg_scale * kV
phi_s = np.zeros(4)
gap_mode = np.zeros(len(phi_s))
Ng = len(phi_s)
Fcup_dist = 10 * mm

# Match section Parameters and ESQ parameters
focus_after_gap = 6
match_after_gap = 6
esq_space = 3.0 * mm
Vq_match = np.array([-125.24, 326.43, -352.12, 175.68])  # Volts
Nq_match = len(Vq_match)

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
Tbx = 0.1  # eV
Tby = Tbx
Tbz = 0.1  # eV
div_angle = 3.78 * mrad
emit = 1.336 * mm * mrad
init_I = 10 * uA
Np_injected = 0  # initialize injection counter
Np_max = int(1e6)
beam_length = period  # in seconds

# Specify Species and ion type
beam = wp.Species(type=wp.Argon, charge_state=1, name="Ar+", color=wp.blue)
tracked_ions = wp.Species(type=wp.Argon, charge_state=0, name="Track", color=wp.red)
mass_eV = beam.mass * pow(SC.c, 2) / wp.jperev

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
vthx = np.sqrt(Tbx * wp.jperev / beam.mass)
vthy = np.sqrt(Tby * wp.jperev / beam.mass)
vthz = np.sqrt(Tbz * wp.jperev / beam.mass)
wp.derivqty()

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
gap_dist = np.zeros(Ng)
E_s = init_E
if do_matching_section:
    match_after_gap = 6
    match_centers = np.array([10.1, 23.9, 36, 49.5]) * mm
    match_length = 60 * mm
    gap_centers = mems_utils.calc_gap_centers(
        E_s,
        mass_eV,
        phi_s,
        gap_mode,
        freq,
        Vg / Vg_scale,
        match_after_gap=match_after_gap,
        match_length=match_length,
    )
    match_centers += gap_centers[match_after_gap - 1] + gap_width / 2.0
else:
    gap_centers = mems_utils.calc_gap_centers(
        E_s,
        mass_eV,
        phi_s,
        gap_mode,
        freq,
        Vg / Vg_scale,
    )

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
    wp.w3d.zmmin = -period * beam.vbeam
    wp.w3d.zmmax = (gap_centers[-1] - gap_centers[-2]) + abs(wp.w3d.zmmin)
else:
    wp.w3d.zmmin = -3 * mm
    wp.w3d.zmmax = gap_centers[0] + gap_width / 2.0 + Fcup_dist

wp.w3d.nx = 60
wp.w3d.ny = 60
wp.w3d.nz = 130
lab_center = (wp.w3d.zmmax + wp.w3d.zmmin) / 2.0
dz = (wp.w3d.zmmax - wp.w3d.zmmin) / wp.w3d.nz

# Set boundary conditions
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.periodic

wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb
wp.top.prwall = aperture

# keep track of when the particles are born
wp.top.inject = 1
wp.top.ibeam_s = init_I
wp.top.ekin_s = init_E
wp.derivqty()
wp.top.dt = (
    0.7 * dz / mems_utils.calc_velocity((init_E + Vg) * wp.jperev, beam.mass)
)  # CF conditions
inj_dz = beam.vbeam * wp.top.dt

# Calculate and set the weight of particle
Np_inject = int(Np_max / (beam_length / wp.top.dt))
pweight = wp.top.dt * init_I / beam.charge / Np_inject
beam.pgroup.sw[beam.js] = pweight
tracked_ions.pgroup.sw[tracked_ions.js] = 0
# Calculate phase shift needed to be in resonance
phase_shift = mems_utils.calc_phase_shift(
    freq,
    abs(wp.top.zinject[0]),
    beam.vbeam,
)
if beam_length != None:
    phase_shift += mems_utils.calc_phase_shift(
        freq, beam_length * beam.vbeam / 2, beam.vbeam
    )
rf_volt = lambda time: Vg * np.cos(twopi * freq * time - phase_shift)


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
        zmax=wp.top.zinject[0] + beam.vbeam * wp.top.dt,
        vthx=vthx,
        vthy=vthy,
        vthz=vthz,
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
# set_lhistories()

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
diagns_zgap = []
ilws.append(wp.addlabwindow(2.0 * dz))  # Initial lab window
diagns_zgap.append(ZCrossingParticles(zz=wp.top.zinject[0] + 2.0 * dz, laccumulate=1))

# Loop through gap centers and place diagnostics at the geometric center of gap-pairs.
# If the matching section was done, then add a diagnostics for the quadrupoles and
# insert the diagnostic at the correct index in the diagnostic list.
for i in range(Ng - 1):
    zloc = (gap_centers[i + 1] + gap_centers[i]) / 2.0
    ilws.append(wp.addlabwindow(zloc))
    diagns_zgap.append(ZCrossingParticles(zz=zloc, laccumulate=1))

if do_matching_section:
    for i in range(Nq_match - 1):
        zloc = (match_centers[i + 1] + match_centers[i]) / 2.0
        ilws.insert(match_after_gap + i, wp.addlabwindow(zloc))
        diagns_zgap.insert(
            match_after_gap + i, ZCrossingParticles(zz=zloc, laccumulate=1)
        )

ilws.append(wp.addlabwindow(gap_centers[-1] + Fcup_dist))
diagns_zgap.append(ZCrossingParticles(zz=gap_centers[-1] + Fcup_dist, laccumulate=1))

# Set up fieldsolver
solver = wp.MRBlock3D()
solver.ldosolve = True  # Enable self-fields.
wp.registersolver(solver)
solver.mgtol = 1.0  # Poisson solver tolerance, in volts
solver.mgparam = 1.5
solver.downpasses = 2
solver.uppasses = 2
solver.mgntverbose = 10

# Recalculate fields and conductor information every time step. Needed for
# oscillating fields in a moving frame.
solver.gridmode = 0

# Generate the PIC code (allocate storage, load ptcls, t=0 plots, etc.)
wp.package("w3d")

for i, pa in enumerate(gap_centers):
    print(f"RF Gap Center {i+1} placed at {pa/mm:3f} (mm)")

# Create list of conductors to hold the created gaps and ESQs
conductors = []
child_meshes = []

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

    # Add in refined mesh around gap.
    child_mesh = solver.addchild(
        mins=[-aperture, -aperture, zl],
        maxs=[aperture, aperture, zr],
        refinement=[2, 2, 3],
    )
    child_meshes.append(child_mesh)

# Create matching section consisting of four quadrupoles.
if do_matching_section:
    for i, pos in enumerate(match_centers):
        this_zc = pos
        this_Vq = Vq_match[i]

        this_ESQ = mems_utils.Mems_ESQ_SolidCyl(this_zc, this_Vq, -this_Vq, chop=True)
        this_ESQ.set_geometry(rp=aperture, R=1.3 * aperture, lq=lq)
        this_cond = this_ESQ.generate()
        conductors.append(this_cond)

        # Add in refined mesh around ESQs.
        child_mesh = solver.addchild(
            mins=[-aperture, -aperture, zl],
            maxs=[aperture, aperture, zr],
            refinement=[3, 3, 4],
        )
        child_meshes.append(child_mesh)

# Create and intialize the scraper that will collect lost particle data.
aperture_wall = wp.ZCylinderOut(
    radius=aperture, zlower=-wp.top.largepos, zupper=wp.top.largepos
)

if do_focusing_quads:
    # Calculate ESQ center positions and then install.
    esq_pos = mems_utils.calc_zESQ(gap_centers[focus_after_gap:], diagns_zgap[-1].zz)

    # Loop through focusing quads, create them with assigned voltages and store.
    for i, pos in enumerate(esq_pos[0:8]):
        this_ESQ = mems_utils.Mems_ESQ_SolidCyl(pos, Vq[i], -Vq[i], chop=True)
        this_ESQ.set_geometry(rp=aperture, R=1.304 * aperture, lq=lq)
        this_cond = this_ESQ.generate()
        conductors.append(this_cond)

        # Add in refined mesh around ESQs.
        child_mesh = solver.addchild(
            mins=[-aperture, -aperture, pos - lq / 2.0],
            maxs=[aperture, aperture, pos + lq / 2],
            refinement=[3, 3, 4],
        )
        child_meshes.append(child_mesh)

for cond in conductors:
    wp.installconductors(cond)

Fcup = wp.Box(
    xsize=wp.top.largepos,
    ysize=wp.top.largepos,
    zsize=5.0 * dz,
    zcent=diagns_zgap[-1].getzz() + 2 * mm,
)
scraper = wp.ParticleScraper(aperture_wall, lcollectlpdata=True)
scraper.registerconductors(conductors)
scraper.registerconductors(Fcup)
wp.generate()
x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh

# Add tracker beam that will record full history. This must be set after
# generate is called. For unknown reasons, the tracer cannot be placed arbitrarily
# in the injection scheme. Thus, it is created early on, disabled, then renabled
# at the desired point in injection.
tracker = TraceParticle(
    js=tracked_ions.js,
    x=0.0 * mm,
    y=0.0,
    z=0.0,
    vx=0.0,
    vy=0.0,
    vz=beam.vbeam,
)
tracker.disable()

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
# TODO: Find way to plot select number of plots for better gist graphic.
# ------------------------------------------------------------------------------
# Def plotting routine to be called in stepping
def plotbeam(lplt_tracker=True):
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

    print(f"# ------- Tracker x: {tracker.getx()[-1]/mm:.4f} (mm)")
    print(f"# ------- Tracker z: {tracker.getz()[-1]/mm:.3f} (mm)")

    if lplt_tracker:
        wp.plp(tracker.getx()[-1], tracker.getz()[-1], color=wp.red, msize=3)


# Inject particles based on beam length. The injection will be centered around
# the design particle so that the injection cycle will go for t=beam_length/2,
# then enable the design particle. Once the design particle is enabled
# the window begins moving and the injection is continued for another beam_length/2.
while wp.top.time <= beam_length / 2:
    wp.window(2)
    plotbeam(lplt_tracker=True)
    wp.fma()
    wp.step()

tracker.enable()

while wp.top.time <= beam_length:
    wp.top.vbeamfrm = tracker.getvz()[-1]
    wp.window(2)
    plotbeam(lplt_tracker=True)
    wp.fma()
    wp.step()

wp.top.inject = 0
wp.uninstalluserinjection(injection)

for i in range(Ng):
    while tracker.getz()[-1] < gap_centers[i]:
        wp.top.vbeamfrm = tracker.getvz()[-1]
        reset_tracker(tracker, int(len(tracker.getx()) - 1), 0.0)
        if wp.top.it % 1 == 0:
            wp.window(2)
            plotbeam(lplt_tracker=True)
            wp.fma()

        wp.step(1)

    this_E = init_E + (i + 1) * Vg
    wp.top.dt = 0.7 * dz / mems_utils.calc_velocity(this_E * wp.jperev, beam.mass)

while tracker.getz()[-1] < Fcup.zcent - Fcup.zsize:
    wp.top.vbeamfrm = tracker.getvz()[-1]
    reset_tracker(tracker, int(len(tracker.getx()) - 1), 0.0)
    if wp.top.it % 1 == 0:
        wp.window(2)
        plotbeam(lplt_tracker=True)
        wp.fma()

    wp.step()

tracker_fin_time = tracker.gett()[-1]
tracker_fin_E = 0.5 * beam.mass * pow(tracker.getvz()[-1], 2) / wp.jperev
wp.top.vbeamfrm = 0.0
while wp.top.time < tracker_fin_time + period:
    reset_tracker(tracker, int(len(tracker.getx()) - 1), 0.0)
    wp.window(2)
    plotbeam(lplt_tracker=True)
    wp.fma()
    wp.step()

end_time = time.time()
print(f"----- Run Time: {(end_time - start_time)/60.:.4f} (min)")
# ------------------------------------------------------------------------------
#    Simulation Diagnostics
# Here the final diagnostics are computed/extracted. The diagnostic plots are
# made.
# ------------------------------------------------------------------------------
Np_delivered = diagns_zgap[-1].getvz().shape[0]
Efin = beam.mass * pow(diagns_zgap[-1].getvz(), 2) / 2.0 / wp.jperev
tfin = diagns_zgap[-1].gett()
frac_delivered = Np_delivered / Np_injected
frac_lost = abs(Np_delivered - Np_injected) / Np_injected

Data = mems_utils.Data_Ext(ilws, diagns_zgap, beam, tracker)
Data.grab_data()

# ------------------------------------------------------------------------------
#    Diagnostic Plotting
# Plots are collected and exported to pdf file. Later this will be incorporated
# into the Data class.
# ------------------------------------------------------------------------------
path = mems_utils.create_save_path(dir_name="sim_outputs", prefix="sim")
mems_utils.move_cgms(path, match="*.cgm*")

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
    for i in range(len(diagns_zgap)):
        this_E = 0.5 * beam.mass * pow(diagns_zgap[i].getvz(), 2) / wp.jperev
        this_t = diagns_zgap[i].gett()
        this_ts = tracker.gett()[
            np.argmin(abs(diagns_zgap[i].getzz() - tracker.getz()))
        ]

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
        Eax.set_title(f"z={diagns_zgap[i].getzz()/mm:.2f} (mm)")

        tax.bar(
            tedges[:-1] / ns,
            tcounts[:] / Np,
            width=np.diff(tedges[:] / ns),
            edgecolor="black",
            lw="1",
        )
        tax.set_xlabel(f"Time (ns), Tracker time = {this_ts/ns:.2f} (ns)")
        tax.set_ylabel("Fraction of Particles")
        tax.set_title(f"z={diagns_zgap[i].getzz()/mm:.2f} (mm)")

        plt.tight_layout()
        pdf.savefig()
        plt.close()

# Write data in output file
output_file_path = os.path.join(path, "output.txt")
with open(output_file_path, "w") as file:
    file.write("#----- Injected Beam" + "\n")
    file.write(f"{'Ion:':<30} {beam.type.name}" + "\n")
    file.write(f"{'Number of Injected:':<30} {Np_injected:.0e}" + "\n")
    file.write(f"{'Particle weight:':<30} {pweight:.4f}" + "\n")
    file.write(f"{'Injection Energy:':<30} {init_E/keV:.2f} [keV]" + "\n")
    file.write(
        f"{'Injected Average Current Iavg:':<30} {init_I/uA:.4e} [micro-Amps]" + "\n"
    )
    file.write(f"{'Predicted Final Design Energy:':<30} {E_s/keV:.2f} [keV]" + "\n")

    file.write("#----- Acceleration Lattice" + "\n")
    file.write(f"Matching Section: {do_matching_section}" + "\n")
    if do_matching_section:
        file.write(
            f"Matching Section Voltages: {np.array2string(Vq_match/kV, precision=4)}"
            + "\n"
        )
    file.write(f"Focusing Quads: {do_focusing_quads}" + "\n")
    if do_focusing_quads:
        file.write(
            f"Focusing Quads Vq {np.array2string(Vq_match/kV, precision=4)}" + "\n"
        )

    file.write(f"{'Number of Gaps':<30} {int(Ng)}" + "\n")
    file.write(
        f"{'Fcup Distance (from final plate):':<30} {Fcup_dist/mm:.2f} [mm]" + "\n"
    )
    file.write(
        f"{'Gap Centers:':<30} {np.array2string(gap_centers/cm, precision=4)} [cm]"
        + "\n"
    )
    file.write(
        f"{'Gap Distances:':<30} {np.array2string(np.diff(gap_centers/cm), precision=4)} [cm]"
        + "\n"
    )
    file.write(f"{'System Length:':<30} {Fcup.zcent/cm:.3f} [cm]" + "\n")
    file.write(f"{'Gap Voltage:':<30} {Vg/kV:.2f} [kV]" + "\n")
    file.write(f"{'Gap Width:':<30} {gap_width/mm:.2f} [mm]" + "\n")
    file.write(f"{'RF Frequency:':<30} {freq/MHz:.2f} [MHz]" + "\n")
    file.write(f"{'RF Wavelength:':<30} {SC.c/freq:.2f} [m]" + "\n")
    file.write(
        f"{'Sync Phi:':<30} {np.array2string(phi_s*180/np.pi,precision=3)} [deg]" + "\n"
    )

    file.write("#----- Numerical Parameters" + "\n")
    file.write(f"{'Time step:':<30} {wp.top.dt:>.4e} [s]" + "\n")
    file.write(f"{'z Grid spacing:':<30} {dz:>.4e} [m]" + "\n")
    file.write(f"{'x Grid spacing:':<30} {wp.w3d.dx:>.4e} [m]" + "\n")
    file.write(f"{'y Grid spacing:':<30} {wp.w3d.dx:>.4e} [m]" + "\n")

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
            ax.set_title(f"z={diagns_zgap[i].getzz()/mm:.2f} (mm)")
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

# ------------------------------------------------------------------------------
#    Saving Particle Data
# Save particle data at zdiagnostics. The data will be a list of Npi by 8 arrays.
# Npi is the number of particles at each diagnostic that made it through. This
# will generally be different as particles will be lot. The 8 columns are
# x, y, xp, yp, vx, vy, vz, t. Each z-location will be the same and will be recorded
# as the last element in the list of matrices. Thus, the first m-1 elements of the
# list will be the particle data matrices, and the final element an array of
# z-coordinates for the particle diagnostics. Since this data set will be
# inhomogenous, the data will be saved as an HD5 file.
# The tracker particle data will also be saved and is much easier.
# ------------------------------------------------------------------------------
# Create a dictionary to hold the diagnostic data for each diagnostics. The
# structure will be {Diagn#:{'attr':data}} where each 'attr' is given in the
# data vars list that comprises the data each diagnostic collects.
data_vars = ["x", "y", "xp", "yp", "vx", "vy", "vz", "t"]
diagnostic_data = {}
# Loop through the diagnostics and collect the data into a temporary dictionary.
# The main dictionary will be updated after each pass
for i, diag in enumerate(diagns_zgap):
    this_data = {}
    for var in data_vars:
        this_data.update({var: getattr(diag, f"get{var}")()})

    this_data.update({"zloc": diag.getzz()})
    diagnostic_data.update({f"diag{i}": this_data})

# Save as pickle file
with open(os.path.join(path, "diagnostic_data.pkl"), "wb") as file:
    pickle.dump(diagnostic_data, file)

# Create tracker data that is 3byN where N is the number of points and the rows
# are z, vz, t. Note the tracker is always at x=y=0 amd vx=vy=0.
tracker_data = np.array([tracker.getz(), tracker.getvz(), tracker.gett()])
np.save(os.path.join(path, "trace_particle_data.npy"), tracker_data)

# Create plots and export to pdf for each diagnostic. At each diagnostic the
# particles are binned based on time for a fixed dt. The edges of the particle
# binning are used to create masks and grab the indices in the particle arrays for
# each bin of particles. The statistics are then calculated over the bins and
# plotted to give a time evolution of the statistic over the diagnostic.
hist_dt = 0.5 * ns
time_select = 0.05 * period

# Create array to hold the number of particles at each diagnostic
Np_select = np.zeros(len(diagns_zgap))
with PdfPages(path + "/" + "diagnostic_plots" + ".pdf") as pdf:
    # Start the main loop. This will loop through each zdiagnostic and then
    # do the selection and plotting. After each loop, a collection of pdfs is
    # sent is saved and the process repeated.
    particle_counts = np.zeros(len(diagns_zgap))
    for i, zd in enumerate(diagns_zgap):
        # Get index for when design particle crossed.
        dsgn_ind = np.argmin(abs(tracker.getz() - zd.zz))
        this_ts = tracker.gett()[dsgn_ind]
        this_Es = 0.5 * beam.mass * pow(tracker.getvz()[dsgn_ind], 2) / wp.jperev

        # Mask the particles based on time_select. Then, bin the particles
        if time_select != None:
            mask_time = abs(zd.gett() - this_ts) <= time_select
            zdtmask = zd.gett()[mask_time]
            num_bins = int(
                (zd.gett()[mask_time].max() - zd.gett()[mask_time].min()) / hist_dt
            )
            counts, edges = np.histogram(zd.gett()[mask_time], bins=num_bins)
            this_Np = int(np.sum(counts))
            Np_select[i] = this_Np
        else:
            # Create a mask that selects all particles. A bit redundant but the
            # subsequent code is easier if a mask is used.
            mask_time = np.ones(len(zd.gett()), dtype=bool)
            zdtmask = zd.gett()[mask_time]
            num_bins = int((zd.gett().max() - zd.gett().min()) / hist_dt)
            counts, edges = np.histogram(zd.gett(), bins=num_bins)
            this_Np = int(np.sum(counts))
            Np_select[i] = this_Np

        # Extract masked data from the zd outputs.
        zdx = zd.getx()[mask_time]
        zdy = zd.gety()[mask_time]
        zdt = zd.gett()[mask_time]
        zdxp = zd.getxp()[mask_time]
        zdyp = zd.getyp()[mask_time]
        zdvx = zd.getvx()[mask_time]
        zdvy = zd.getvy()[mask_time]
        zdvz = zd.getvz()[mask_time]

        # Calculate the energy and current for this diagnostic
        this_E = 0.5 * beam.mass * pow(zdvz, 2) / wp.jperev
        this_I = counts * wp.echarge * pweight / hist_dt

        # Loop through and get indices of particles belonging to each bin
        bin_indices = []
        for j in range(len(edges) - 1):
            inds = np.where((zdt >= edges[j]) & (zdt < edges[j + 1]))[0]
            bin_indices.append(inds)

        # Each array of indices can no be looped through and the desired statistics
        # computed over all particles within that time bin.
        rx = np.zeros(len(bin_indices))
        rxp = np.zeros(len(bin_indices))
        emitx = np.zeros(len(bin_indices))
        emitnx = np.zeros(len(bin_indices))
        vzbar = np.zeros(len(bin_indices))
        energy = np.zeros(len(bin_indices))
        Q = np.zeros(len(bin_indices))

        for k, indices in enumerate(bin_indices):
            # Compute statistical envelope edge radii and envelope edge angle
            rx[k] = 2 * np.sqrt(np.mean(pow(zdx[indices], 2)))
            rxp[k] = 2 * np.mean(zdx[indices] * zdxp[indices]) / (rx[k] / 2)

            # Compute emittance and normalized emittance in x
            xbarsq = np.mean(pow(zdx[indices], 2))
            xpbarsq = np.mean(pow(zdxp[indices], 2))
            xxpbarsq = pow(xbarsq * xpbarsq, 2)
            emitx[k] = 4 * np.sqrt(xbarsq * xpbarsq - xxpbarsq)
            vzbar[k] = np.mean(zdvz[indices])
            emitnx[k] = vzbar[k] / SC.c * emitx[k]

            # Compute energy
            energy[k] = 0.5 * beam.mass * pow(vzbar[k], 2) / wp.jperev

            # Compute perveance
            Qnum = wp.echarge * this_I[k]
            Qdenom = 4.0 * np.pi * wp.eps0 * vzbar[k] * energy[k] * wp.jperev
            Q[k] = Qnum / Qdenom

        # Now make plots
        plot_title = f"zd{i+1} at {zd.zz/mm:.2f} (mm)"
        time_data = (edges[:-1] - this_ts) / ns

        fig, ax = plt.subplots()
        ax.plot(time_data, this_I / init_I)
        if time_select != None:
            ax.axvline(
                x=time_select / ns, c="g", ls="--", lw=1, label="$t$-filter boundary"
            )
            ax.axvline(x=-time_select / ns, c="g", ls="--", lw=1)
        ax.axhline(y=1, c="k", lw=1)
        ax.axvline(x=0, c="r", ls="--", lw=1, label="Design Particle")
        ax.legend()
        ax.set_xlabel(r"$\Delta t$ (ns)")
        ax.set_ylabel(r"$I/I_0$")
        ax.set_title(plot_title)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(time_data, Q / 1e-5)
        ax.set_xlabel(r"$\Delta t$ (ns)")
        ax.set_ylabel(r"$\bar{Q} \times 10^{-5}$")
        if time_select != None:
            ax.axvline(
                x=time_select / ns, c="g", ls="--", lw=1, label="$t$-filter boundary"
            )
            ax.axvline(x=-time_select / ns, c="g", ls="--", lw=1)
        ax.axhline(y=0, c="k", lw=1)
        ax.axvline(x=0, c="r", ls="--", lw=1, label="Design Particle")
        ax.legend()
        ax.set_title(plot_title)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(time_data, (energy - this_Es) / keV)
        ax.set_xlabel(r"$\Delta t$ (ns)")
        ax.set_ylabel(r"$\Delta\bar{\mathcal{E}}$ (keV)")
        if time_select != None:
            ax.axvline(
                x=time_select / ns, c="g", ls="--", lw=1, label="$t$-filter boundary"
            )
            ax.axvline(x=-time_select / ns, c="g", ls="--", lw=1)
        ax.axhline(y=0, c="r", ls="--", lw=1, label="Design Particle")
        ax.axvline(x=0, c="r", ls="--", lw=1)
        ax.legend()
        ax.set_title(plot_title)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        # Create phase space plots of energy and time. Due to the large number
        # of particles, make a random selection of 10,000. If less than 10,000
        # plot all of them.
        if this_Np < int(1e4):
            rand_ints = np.random.randint(0, high=this_Np - 1, size=int(this_Np))
        else:
            rand_ints = np.random.randint(0, high=this_Np - 1, size=int(1e4))

        g = mems_utils.make_dist_plot(
            (zdt[rand_ints] - this_ts) / period,
            this_E[rand_ints] / keV,
            xlabel=r"Relative Time Difference $\Delta t / \tau_{rf}$",
            ylabel=r"Energy Difference $\mathcal{E}$ (keV)",
            auto_clip=True,
            xref=0.0,
            yref=this_Es / keV,
            levels=15,
            bins=40,
            weight=1 / this_Np,
            dx_bin=0.015,
            dy_bin=0.5,
        )
        pdf.savefig(g.fig)
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(time_data, rx / mm)
        ax.set_xlabel(r"$\Delta t$ (ns)")
        ax.set_ylabel(r"$\bar{r_x}$ (mm)")
        if time_select != None:
            ax.axvline(
                x=time_select / ns, c="g", ls="--", lw=1, label="$t$-filter boundary"
            )
            ax.axvline(x=-time_select / ns, c="g", ls="--", lw=1)
        ax.axhline(y=0, c="k", ls="--", lw=1)
        ax.axhline(y=aperture / mm, c="k", ls="--", lw=1, label="Aperture")
        ax.axvline(x=0, c="r", ls="--", lw=1, label="Design Particle")
        ax.legend()
        ax.set_title(plot_title)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(time_data, rxp / mrad)
        ax.set_xlabel(r"$\Delta t$ (ns)")
        ax.set_ylabel(r"$\bar{r}'_x$ (mrad)")
        if time_select != None:
            ax.axvline(
                x=time_select / ns, c="g", ls="--", lw=1, label="$t$-filter boundary"
            )
            ax.axvline(x=-time_select / ns, c="g", ls="--", lw=1)
        ax.axhline(y=0, c="k", ls="--", lw=1)
        ax.axvline(x=0, c="r", ls="--", lw=1, label="Design Particle")
        ax.legend()
        ax.set_title(plot_title)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(time_data, emitx / mm / mrad)
        ax.set_xlabel(r"$\Delta t$ (ns)")
        ax.set_ylabel(r"$\bar{\epsilon}_x$ (mm-mrad)")
        if time_select != None:
            ax.axvline(
                x=time_select / ns, c="g", ls="--", lw=1, label="$t$-filter boundary"
            )
            ax.axvline(x=-time_select / ns, c="g", ls="--", lw=1)
        ax.axhline(y=0, c="k", ls="--", lw=1)
        ax.axvline(x=0, c="r", ls="--", lw=1, label="Design Particle")
        ax.legend()
        ax.set_title(plot_title)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        fig, ax = plt.subplots()
        ax.plot(time_data, emitnx / mm / mrad)
        ax.set_xlabel(r"$\Delta t$ (ns)")
        ax.set_ylabel(r"$\bar{\epsilon}_{x,n}$ (mm-mrad)")
        if time_select != None:
            ax.axvline(
                x=time_select / ns, c="g", ls="--", lw=1, label="$t$-filter boundary"
            )
            ax.axvline(x=-time_select / ns, c="g", ls="--", lw=1)
        ax.axhline(y=0, c="k", ls="--", lw=1)
        ax.axvline(x=0, c="r", ls="--", lw=1, label="Design Particle")
        ax.legend()
        ax.set_title(plot_title)
        plt.tight_layout()
        pdf.savefig()
        plt.close()

    # Create final plot of fraction of particles at each diagnostic
    fig, ax = plt.subplots()
    ax.plot([zd.zz / mm for zd in diagns_zgap], Np_select / Np_select[0])
    ax.scatter([zd.zz / mm for zd in diagns_zgap], Np_select / Np_select[0])
    for gc in gap_centers[:-1]:
        ax.axvline(x=gc / mm, c="k", ls="--", lw=1)
    ax.axvline(x=gap_centers[-1] / mm, c="k", ls="--", lw=1, label="Gap Center")
    ax.axhline(y=1, c="k", lw=1)
    ax.axhline(y=0, c="k", lw=1)
    ax.set_xlabel(r"Diagnostic Location (mm)")
    ax.set_ylabel(r"Fraction of Remaining Bucket Particles")
    plt.tight_layout()
    pdf.savefig()
    plt.close()
