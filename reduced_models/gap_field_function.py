# This script will load the accelerating gaps and then fit a function to the
# time varying RF gap field.

import numpy as np
import scipy
import scipy.constants as SC
from scipy.optimize import curve_fit
from scipy.special import jv
import matplotlib.pyplot as plt
import os
import pdb
import time

import warp as wp
from warp.utils.timedependentvoltage import TimeVoltage
from warp.particles.singleparticle import TraceParticle


st = time.time()
# different particle masses in eV
# amu in eV
amu = SC.physical_constants["atomic mass constant energy equivalent in MeV"][0] * 1e6

Ar_mass = 39.948 * amu
mass = Ar_mass
He_mass = 4 * amu
p_mass = amu
kV = 1000.0
keV = 1000.0
MHz = 1e6
cm = 1e-2
mm = 1e-3
um = 1e-6
ns = 1e-9  # nanoseconds
uA = 1e-6
twopi = 2 * np.pi
Vscale = 1.0901  # Scale factor to get desired voltage max on field.

wp.setup()
# ------------------------------------------------------------------------------
#     Functions
# This section is dedicated to creating useful functions for the script.
# ------------------------------------------------------------------------------


def getindex(mesh, value, spacing):
    """Find index in mesh for or mesh-value closest to specified value

    Function finds index corresponding closest to 'value' in 'mesh'. The spacing
    parameter should be enough for the range [value-spacing, value+spacing] to
    encompass nearby mesh-entries .

    Parameters
    ----------
    mesh : ndarray
        1D array that will be used to find entry closest to value
    value : float
        This is the number that is searched for in mesh.
    spacing : float
        Dictates the range of values that will fall into the region holding the
        desired value in mesh. Best to overshoot with this parameter and make
        a broad range.

    Returns
    -------
    index : int
        Index for the mesh-value closest to the desired value.
    """

    # Check if value is already in mesh
    if value in mesh:
        return np.where(mesh == value)[0][0]

    # Create array of possible indices
    indices = np.where((mesh > (value - spacing)) & (mesh < (value + spacing)))[0]

    # Compute differences of the indexed mesh-value with desired value
    difference = []
    for index in indices:
        diff = np.sqrt((mesh[index] ** 2 - value**2) ** 2)
        difference.append(diff)

    # Smallest element will be the index closest to value in indices
    i = np.argmin(difference)
    index = indices[i]

    return index


def beta(E, mass=Ar_mass, q=1, nonrel=True):
    """Velocity of a particle with energy E."""
    if nonrel:
        beta = np.sqrt(2 * E / mass)
    else:
        gamma = (E + mass) / mass
        beta = np.sqrt(1 - 1 / gamma / gamma)

    return beta


def calc_pires(energy, freq, mass=Ar_mass, q=1):
    """RF resonance condition in pi-mode"""
    beta_lambda = beta(energy, mass=mass, q=q) * SC.c / freq
    return beta_lambda / 2


def gap_voltage(t):
    """ "Sinusoidal function of the gap voltage"""

    v = 7 * kV * np.cos(2 * np.pi * 13.6 * MHz * t)

    return v


def neg_gap_voltage(t):
    """ "Sinusoidal function of the gap voltage"""

    v = 7 * kV * np.cos(2 * np.pi * 13.6 * MHz * t)

    return v


def create_wafer(
    cent,
    width=2.0 * mm,
    cell_width=3.0 * mm,
    length=0.7 * mm,
    rin=0.55 * mm,
    rout=0.75 * mm,
    xcent=0.0 * mm,
    ycent=0.0 * mm,
    voltage=0.0,
):
    """ "Create a single wafer

    An acceleration gap will be comprised of two wafers, one grounded and one
    with an RF varying voltage. Creating a single wafer without combining them
    (create_gap function) will allow to place a time variation using Warp that
    one mess up the potential fields."""

    prong_width = rout - rin
    ravg = (rout + rin) / 2

    # Left wafer first.

    # Create box surrounding wafer. The extent is slightly larger than 5mm unit
    # cell. The simulation cell will chop this to be correct so long as the
    # inner box separation is correct (approximately 0.2mm thickness)
    box_out = wp.Box(
        xsize=cell_width,
        ysize=cell_width,
        zsize=length,
        zcent=cent,
        xcent=xcent,
        ycent=ycent,
        voltage=voltage,
    )
    box_in = wp.Box(
        xsize=cell_width - 0.0002,
        ysize=cell_width - 0.0002,
        zsize=length,
        zcent=cent,
        xcent=xcent,
        ycent=ycent,
        voltage=voltage,
        condid=box_out.condid,
    )
    box = box_out - box_in

    annulus = wp.Annulus(
        rmin=rin,
        rmax=rout,
        length=length,
        zcent=cent,
        xcent=xcent,
        ycent=ycent,
        voltage=voltage,
        condid=box.condid,
    )

    # Create prongs. This is done using four box conductors and shifting
    # respective x/y centers to create the prong.
    top_prong = wp.Box(
        xsize=prong_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        zcent=cent,
        xcent=xcent,
        ycent=ycent + (cell_width / 2 + ravg) / 2,
        voltage=voltage,
        condid=box.condid,
    )
    bot_prong = wp.Box(
        xsize=prong_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        zcent=cent,
        xcent=xcent,
        ycent=ycent - (cell_width / 2 + ravg) / 2,
        voltage=voltage,
        condid=box.condid,
    )
    rside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=prong_width,
        zsize=length,
        zcent=cent,
        xcent=xcent + (cell_width / 2 + ravg) / 2,
        ycent=ycent,
        voltage=voltage,
        condid=box.condid,
    )
    lside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=prong_width,
        zsize=length,
        zcent=cent,
        xcent=xcent - (cell_width / 2 + ravg) / 2,
        ycent=ycent,
        voltage=voltage,
        condid=box.condid,
    )

    # Add together
    cond = annulus + box + top_prong + bot_prong + rside_prong + lside_prong

    return cond


def create_gap(
    cent,
    left_volt,
    right_volt,
    width=2.0 * mm,
    cell_width=3.0 * mm,
    length=0.7 * mm,
    rin=0.55 * mm,
    rout=0.75 * mm,
    xcent=0.0 * mm,
    ycent=0.0 * mm,
):
    """Create an acceleration gap consisting of two wafers.

    The wafer consists of a thin annulus with four rods attaching to the conducting
    cell wall. The cell is 5mm where the edge is a conducting square. The annulus
    is approximately 0.2mm in thickness with an inner radius of 0.55mm and outer
    radius of 0.75mm. The top, bottom, and two sides of the annulus are connected
    to the outer conducting box by 4 prongs that are of approximately equal
    thickness to the ring.

    Here, the annuli are created easy enough. The box and prongs are created
    individually for each left/right wafer and then added to give the overall
    conductor.

    Note, this assumes l4 symmetry is turned on. Thus, only one set of prongs needs
    to be created for top/bottom left/right symmetry."""

    prong_width = rout - rin
    ravg = (rout + rin) / 2

    # Left wafer first.
    left_wafer = wp.Annulus(
        rmin=rin,
        rmax=rout,
        length=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent,
        ycent=ycent,
    )

    # Create box surrounding wafer. The extent is slightly larger than 5mm unit
    # cell. The simulation cell will chop this to be correct so long as the
    # inner box separation is correct (approximately 0.2mm thickness)
    l_box_out = wp.Box(
        xsize=cell_width * (1 + 0.02),
        ysize=cell_width * (1 + 0.02),
        zsize=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent,
        ycent=ycent,
    )
    l_box_in = wp.Box(
        xsize=cell_width * (1 - 0.02),
        ysize=cell_width * (1 - 0.02),
        zsize=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent,
        ycent=ycent,
    )
    l_box = l_box_out - l_box_in

    # Create prongs. This is done using four box conductors and shifting
    # respective x/y centers to create the prong.
    l_top_prong = wp.Box(
        xsize=prong_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent,
        ycent=ycent + (cell_width / 2 + ravg) / 2,
    )
    l_bot_prong = wp.Box(
        xsize=prong_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent,
        ycent=ycent - (cell_width / 2 + ravg) / 2,
    )
    l_rside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=prong_width,
        zsize=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent + (cell_width / 2 + ravg) / 2,
        ycent=ycent,
    )
    l_lside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=prong_width,
        zsize=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent - (cell_width / 2 + ravg) / 2,
        ycent=ycent,
    )

    # Add together
    left = (
        left_wafer + l_box + l_top_prong + l_bot_prong + l_rside_prong + l_lside_prong
    )

    right_wafer = wp.Annulus(
        rmin=rin,
        rmax=rout,
        length=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent,
        ycent=ycent,
    )

    r_box_out = wp.Box(
        xsize=cell_width * (1 + 0.02),
        ysize=cell_width * (1 + 0.02),
        zsize=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent,
        ycent=ycent,
    )
    r_box_in = wp.Box(
        xsize=cell_width * (1 - 0.02),
        ysize=cell_width * (1 - 0.02),
        zsize=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent,
        ycent=ycent,
    )
    r_box = r_box_out - r_box_in

    r_top_prong = wp.Box(
        xsize=prong_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent,
        ycent=ycent + (cell_width / 2 + ravg) / 2,
    )
    r_bot_prong = wp.Box(
        xsize=prong_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent,
        ycent=ycent - (cell_width / 2 + ravg) / 2,
    )
    r_rside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=prong_width,
        zsize=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent + (cell_width / 2 + ravg) / 2,
        ycent=ycent,
    )
    r_lside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=prong_width,
        zsize=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent - (cell_width / 2 + ravg) / 2,
        ycent=ycent,
    )
    right = (
        right_wafer + r_box + r_top_prong + r_bot_prong + r_rside_prong + r_lside_prong
    )

    gap = left + right
    return gap


def create_filled_gap(
    cent,
    left_volt,
    right_volt,
    hole_radius,
    gap_width=2 * mm,
    xcent=0.0 * mm,
    ycent=0.0 * mm,
    xsize=3 * mm,
    ysize=3 * mm,
    zsize=0.7 * mm,
):
    """Create a metallic box with bore aperture hole."""

    lbox = wp.Box(
        xsize=xsize,
        ysize=ysize,
        zsize=zsize,
        voltage=left_volt,
        zcent=cent - gap_width / 2 - zsize / 2,
        xcent=xcent,
        ycent=ycent,
    )
    lcylinder = wp.ZCylinder(
        radius=hole_radius,
        length=zsize,
        voltage=left_volt,
        zcent=cent - gap_width / 2 - zsize / 2,
        xcent=xcent,
        ycent=ycent,
    )
    lconductor = lbox - lcylinder

    rbox = wp.Box(
        xsize=xsize,
        ysize=ysize,
        zsize=zsize,
        voltage=right_volt,
        zcent=cent + gap_width / 2 + zsize / 2,
        xcent=xcent,
        ycent=ycent,
    )
    rcylinder = wp.ZCylinder(
        radius=hole_radius,
        length=zsize,
        voltage=right_volt,
        zcent=cent + gap_width / 2 + zsize / 2,
        xcent=xcent,
        ycent=ycent,
    )
    rconductor = rbox - rcylinder

    gap = lconductor + rconductor
    return gap


def uniform_particle_load(E, Np, zcent=0.0 * mm, zl=0.0, zr=0.0, mass=Ar_mass):
    """Uniform load particles around zcent with uniform energy

    A uniform particle load centered around zcent that extents to zcent - zl and
    zcent + zr."""

    # Calculate z-velocities
    vz = np.sqrt(2 * E / mass) * SC.c

    # Create position array
    z = np.linspace(zcent - zl, zcent + zr, Np)

    return z, vz


def voltfunc(time):
    """Voltage function to feed warp for time variation"""

    return np.cos(2.0 * np.pi * 13.06 * MHz * time)


# ------------------------------------------------------------------------------
#     Script parameter settings
# This section is dedicated to naming and setting the various parameters of the
# the script. These settings are used throughout the script and thus different
# settings can be tested by changing the values here. This section will also
# perform quick calculations for the script setup up like the gap positions.
# ------------------------------------------------------------------------------
# Find gap positions. The gap positions will be calculated for 12 gaps giving
# three lattice periods.
length = 0.7 * mm
gap_width = 2.0 * mm
zcenter = abs(0.0 - gap_width / 2.0)
f = 13.06 * MHz
Ng = 2
Np = int(1e4)
Ntrack = int(1e1)
Vg = 5.0 * kV * Vscale
Vgset = Vg / Vscale
E_DC = Vg / gap_width / Vscale
dsgn_phase = -np.pi / 2
gap_cent_dist = []
Einit = 7.0 * keV
rf_wave = beta(Einit) * SC.c / f
fcup_dist = 10.0 * mm
Energy = [Einit]

phi_s = np.ones(Ng) * dsgn_phase
phi_s[1:] = np.linspace(-np.pi / 3, -0.0, Ng - 1)
gap_dist = np.zeros(Ng)
E_s = Einit
for i in range(Ng):
    this_beta = beta(E_s, mass)
    this_cent = this_beta * SC.c / 2 / f
    cent_offset = (phi_s[i] - phi_s[i - 1]) * this_beta * SC.c / f / twopi
    if i < 1:
        gap_dist[i] = (phi_s[i] + np.pi) * this_beta * SC.c / twopi / f
    else:
        gap_dist[i] = this_cent + cent_offset

    dsgn_Egain = Vgset * np.cos(phi_s[i])
    E_s += dsgn_Egain

gap_centers = np.array(gap_dist).cumsum()
print("--Gap Centers")
print(gap_centers / mm)

# ------------------------------------------------------------------------------
#     Simulation Paramter settings.
# This section is dedicated to initializing the parameters necessary to run a
# Warp simulation. This section also creates and loads the conductors.
# ! As of right now the resolution on the x and y meshing is done on the fly. In
#   z however the resoultion is done to give dz = 20um which gives about 100 points
#   in the gap.
# ------------------------------------------------------------------------------
# Set up the 3D simulation

# Create mesh
wp.w3d.xmmin = -1.7 * mm
wp.w3d.xmmax = 1.7 * mm
wp.w3d.nx = 100

wp.w3d.ymmin = -1.7 * mm
wp.w3d.ymmax = 1.7 * mm
wp.w3d.ny = 100

# use gap positioning to find limits on zmesh. Add some spacing at end points.
# Use enough zpoint to resolve the wafers. In this case, resolve with 2 points.
wp.w3d.zmmin = -16 * mm
wp.w3d.zmmax = gap_centers[-1] + fcup_dist

# Set resolution to be 20um giving 35 points to resolve plates and 100 pts in gap
wp.w3d.nz = round((wp.w3d.zmmax - wp.w3d.zmmin) / 50 / um)
dz = (wp.w3d.zmmax - wp.w3d.zmmin) / wp.w3d.nz

# Create particle characteristics. Particles need to be loaded later if using
# addparticles attribute
zload, vzload = uniform_particle_load(Einit, Np, zcent=0.0, zl=rf_wave, zr=0.0)
beam = wp.Species(type=wp.Argon, charge_state=+1, name="Argon Beam")
beam.ekin = 7.0 * keV
wp.derivqty()

# Set timing step with cf condition.
wp.top.dt = 0.7 * dz / beam.vbeam

# Add boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.periodic

# Add particle boundary conditions
wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb

conductors = []
for i, cent in enumerate(gap_centers):
    if i % 2 == 0:
        lzc = cent - gap_width / 2 - length / 2
        rzc = cent + gap_width / 2 + length / 2
        left = create_wafer(lzc)
        right = create_wafer(rzc, voltage=Vgset)
        conductors.append(left)
        conductors.append(right)
    else:
        lzc = cent - gap_width / 2 - length / 2
        rzc = cent + gap_width / 2 + length / 2
        left = create_wafer(lzc, voltage=Vgset)
        right = create_wafer(rzc)
        conductors.append(left)
        conductors.append(right)

wp.w3d.l4symtry = True
wp.f3d.mgtol = 1.0e-6
solver = wp.MRBlock3D()
wp.registersolver(solver)

# Refine mesh in z
childs = []
for i, zc in enumerate(gap_centers):
    this_child = solver.addchild(
        mins=[wp.w3d.xmmin, wp.w3d.ymmin, zc - 1.2 * mm],
        maxs=[0.01 * mm, 0.01 * mm, zc + 1.2 * mm],
        refinement=[1, 1, 3],
    )
    childs.append(this_child)

for cond in conductors:
    wp.installconductor(cond)


# Create accleration gaps with correct coordinates and settings. Collect in
# list and then loop through and install on the mesh.
# do_xoff_cents = False
# xoff_cents = np.array([3.0]) * mm
# ycents = np.array([0.0]) * mm
# conductors = []
# for yc in ycents:
#     for i, cent in enumerate(gap_centers):
#         lzc = cent - gap_width / 2 - length / 2
#         rzc = cent + gap_width / 2 + length / 2
#         if i % 2 == 0:
#             l_wafer = create_wafer(lzc, voltage=0.0, ycent=yc)
#             r_wafer = create_wafer(rzc, voltage=Vgset, ycent=yc)
#             TimeVoltage(r_wafer, voltfunc=voltfunc)
#
#             wp.installconductor(l_wafer)
#             wp.installconductor(r_wafer)
#             # this_cond = create_gap(cent, left_volt=0, right_volt=Vgset, ycent=yc)
#             # wp.installconductor(this_cond)
#
#             # cycle through off center gaps
#             if do_xoff_cents:
#                 for xc in xoff_cents:
#                     l_off_cond = create_wafer(lzc, voltage=0.0, xcent=xc, ycent=yc)
#                     r_off_cond = create_wafer(rzc, voltage=Vgset, xcent=xc, ycent=ync)
#                     TimeVoltage(r_off_cond, voltfunc=voltfunc)
#
#                     wp.installconductor(l_off_cond)
#                     wp.installconductor(r_off_cond)
#
#                     # off_cond = create_gap(
#                     #     cent, left_volt=0, right_volt=Vgset, xcent=xc, ycent=yc
#                     # )
#                     # wp.installconductor(off_cond)
#         else:
#             l_wafer = create_wafer(lzc, voltage=Vgset, ycent=yc)
#             r_wafer = create_wafer(rzc, voltage=0.0, ycent=yc)
#             TimeVoltage(l_wafer, voltfunc=voltfunc)
#
#             wp.installconductor(l_wafer)
#             wp.installconductor(r_wafer)
#
#             # this_cond = create_gap(cent, left_volt=Vgset, right_volt=0, ycent=yc)
#             # wp.installconductor(this_cond)
#
#             # cycle through off center gaps
#             if do_xoff_cents:
#                 for xc in xoff_cents:
#                     l_off_cond = create_wafer(lzc, voltage=Vgset, xcent=xc, ycent=yc)
#                     r_off_cond = create_wafer(rzc, voltage=0.0, xcent=xc, ycent=yc)
#                     TimeVoltage(l_off_cond, voltfunc=voltfunc)
#
#                     wp.installconductor(l_off_cond)
#                     wp.installconductor(r_off_cond)
#                     # off_cond = create_gap(
#                     #     cent, left_volt=Vgset, right_volt=0, xcent=xc, ycent=yc
#                     # )
#                     # wp.installconductors(off_cond)
#

# Create diagnostic for absorbing particles and recording final characteristics.
diagnostic = wp.Box(
    xsize=wp.top.largepos,
    ysize=wp.top.largepos,
    zsize=1.0 * dz,
    voltage=0,
    zcent=gap_centers[-1] + 5 * mm,
    xcent=0.0,
    ycent=0.0,
)
scraper = wp.ParticleScraper(diagnostic, lcollectlpdata=True)
wp.top.lsavelostpart = True

# Perform initial field solve for mesh.
wp.package("w3d")
wp.generate()

# Create gridded electric field data
wp.addnewegrd(
    zs=wp.w3d.zmmin,
    ze=wp.w3d.zmmax,
    xs=wp.w3d.xmmin,
    dx=wp.w3d.dx,
    ys=wp.w3d.ymmin,
    dy=wp.w3d.dy,
    nx=wp.w3d.nx,
    ny=wp.w3d.ny,
    nz=wp.w3d.nz,
    ex=wp.getselfe(comp="x"),
    ey=wp.getselfe(comp="y"),
    ez=wp.getselfe(comp="z"),
    func=voltfunc,
)
# Particles need to be added directly this way after the generate()
beam.addparticles(
    x=np.zeros(Np),
    y=np.zeros(Np),
    z=zload,
    vx=np.zeros(Np),
    vy=np.zeros(Np),
    vz=beam.vbeam,
)

# Create trace particle
tracked_ions = wp.Species(type=wp.Argon, charge_state=+1, name="Tracer")
tracker = TraceParticle(
    js=tracked_ions.js,
    x=0.0,
    y=0.0,
    z=0.0,
    vx=0.0,
    vy=0.0,
    vz=beam.vbeam,
)

while tracker.getz()[-1] < gap_centers[-1] + 2 * mm:
    wp.step(20)

# Do a few more additional steps to be sure
wp.step(300)

et = time.time()
print(f"Run Time: {(et-st)}")

E = Ar_mass * 0.5 * pow(tracker.getvz() / SC.c, 2)
Ebeam = Ar_mass * 0.5 * pow(beam.getvz(lost=1) / SC.c, 2)

# Collect data from the mesh and initialize useful variables.
z = wp.w3d.zmesh
x = wp.w3d.xmesh
y = wp.w3d.ymesh

xc_ind = getindex(x, 0.0, wp.w3d.dx)
yc_ind = xc_ind

Ez0 = wp.getselfe(comp="z")[0, 0, :]
phi0 = wp.getphi()[0, 0, :]
final_dsgn_E = Ar_mass * 0.5 * pow(tracker.getvz()[-1] / SC.c, 2)

# Print out Beam energy and plot tracker particles energy along mesh.
fig, ax = plt.subplots()
ax.plot(tracker.getz() / mm, E / keV)
ax.set_ylim(0, 20)
ax.set_xlim(wp.w3d.zmmin / mm, wp.w3d.zmmax / mm)
ax.axhline(y=E_s, c="r", ls="--", lw=1, label="Theoretical Max")
for i, cent in enumerate(gap_centers):
    ax.axvline(x=cent / mm, c="k", ls="--", lw=0.7)

ax.set_xlabel("Postition z [mm]")
ax.set_ylabel("Kinetic Energy [keV]")
ax.set_title(f"Energy Evolution of Design Particle {final_dsgn_E/keV:.2f}[keV]:")
ax.legend()
plt.show()

# Save arrays
np.save("potential_arrays", phi0)
np.save("field_arrays", Ez0)
np.save("gap_centers", gap_centers)
np.save("zmesh", z)
np.save("xmesh", x)
np.save("ymesh", y)

# Plot final energy distribution of particles that made it to diagnostic
final_E = Ar_mass * 0.5 * pow(beam.getvz(lost=1) / SC.c, 2)
Ecounts, Eedges = np.histogram(final_E, bins=100)

fig, ax = plt.subplots()
ax.bar(
    Eedges[:-1] / keV,
    Ecounts[:] / Np,
    width=np.diff(Eedges[:] / keV),
    edgecolor="black",
    lw="1",
)
ax.set_title("Final Energy Distribution")
ax.set_xlabel(r"Energy [keV]")
ax.set_ylabel(r"Fraction of Total Particles")
plt.show()
# load arrays
# Ez0_arrays = np.load('field_arrays.npy')
# phi0_arrays = np.load('potential_arrays.npy')
# gaps_arrays = np.load('gap_centers.npy')
# z_arrays = np.load('zmesh.npy')
#
# newEz = np.vstack((Ez0_arrays, Ez0))
# newphi = np.vstack((phi0_arrays, phi0))
# newgaps = np.vstack((gaps_arrays, gap_centers))
#
# np.save("field_arrays", newEz)
# np.save("potential_arrays", newphi)
# np.save("gap_centers", newgaps)

# ------------------------------------------------------------------------------
#     Post Analysis
# This section is dedicated to some post analysis work. As of right now, plots
# are generated for the electric fields in z and the potential field. The electric
# field is normalized using the ideal geometry: a gap width of 2mm and a potential
# bias of 7kV.
# Plots using Warp's plotting tools are also used so that the conductors can be
# visualized in the cgm files.
# ------------------------------------------------------------------------------
# Plot potential and electric field (z-direction) on-axis.
fig, ax = plt.subplots()
ax.plot(z / mm, phi0 / Vgset)
ax.set_xlabel("z [mm]")
ax.set_ylabel(r"Potential Normalized by Applied Gap Voltage $V_g$")
ymin, ymax = ax.get_ylim()
for i, cent in enumerate(gap_centers):
    left = cent - gap_width / 2 - length / 2
    right = cent + gap_width / 2 + length / 2
    ax.axvline(x=left / mm, c="gray", lw=0.7)
    ax.axvline(x=right / mm, c="gray", lw=0.7)
    ax.axvspan(left / mm, right / mm, color="grey", alpha=0.5)
ax.axhline(y=0, c="k", lw=1)
ax.axhline(y=1, c="k", ls="--", lw=0.7)
plt.savefig("potential.png", dpi=400)

fig, ax = plt.subplots()
ax.plot(z / mm, Ez0 / E_DC)
ax.set_xlabel("z [mm]")
ax.set_ylabel(r"Normed On-axis E-field $E(x=0, y=0, z)/E_{DC}$")
ymin, ymax = ax.get_ylim()
for i, cent in enumerate(gap_centers):
    left = cent - gap_width / 2 - length / 2
    right = cent + gap_width / 2 + length / 2
    ax.axvline(x=left / mm, c="gray", lw=0.7)
    ax.axvline(x=right / mm, c="gray", lw=0.7)
    ax.axvspan(left / mm, right / mm, color="grey", alpha=0.5)
ax.axhline(y=0, c="k", lw=1)
ax.axhline(y=1, c="k", ls="--", lw=0.7)
ax.axhline(y=-1, c="k", ls="--", lw=0.7)
plt.savefig("Efield.png", dpi=400)
plt.show()


# Warp plotting for verification that mesh and conductors were created properly.
warpplots = True
if warpplots:
    wp.pfzx(fill=1, filled=1)
    wp.fma()
    wp.pfzy(fill=1, filled=1)
    wp.fma()
    # plot the right-side wafer
    plate_cent = gap_centers[0] + length / 2 + gap_width / 2
    dz = z[1] - z[0]
    plate_cent_ind = np.where((z >= plate_cent - dz) & (z <= plate_cent + dz))
    zind = plate_cent_ind[0][0]
    wp.pfxy(iz=zind, fill=1, filled=1)
    wp.fma()

    # plot the left-side wafer
    plate_cent = gap_centers[0] - length / 2 - gap_width / 2
    plate_cent_ind = np.where((z >= plate_cent - dz) & (z <= plate_cent + dz))
    zind = plate_cent_ind[0][0]
    wp.pfxy(iz=zind, fill=1, filled=1)
    wp.fma()

    # Plot potential in xy
    wp.pfxy(iz=int(wp.w3d.nz / 2), fill=1, filled=1)
    wp.fma()
