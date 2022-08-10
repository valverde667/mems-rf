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

import warp as wp
from warp.utils.timedependentvoltage import TimeVoltage
from warp.particles.singleparticle import TraceParticle


# different particle masses in eV
# amu in eV
amu = SC.physical_constants["atomic mass constant energy equivalent in MeV"][0] * 1e6

Ar_mass = 39.948 * amu
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

# ------------------------------------------------------------------------------
#     Functions
# This section is dedicated to creating useful functions for the script.
# ------------------------------------------------------------------------------
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


def uniform_particle_load(E, Np, f=13.6 * MHz, zcent=0.0 * mm, mass=Ar_mass):
    """Uniform load particles around zcent with uniform energy"""

    # Calculate RF-wavelength
    lambda_rf = beta(E, mass=mass) * SC.c / 2 / f

    # Calculate z-velocities
    vz = np.sqrt(2 * E / mass) * SC.c

    # Create position array
    zmin = zcent - lambda_rf / 2
    zmax = zcent + lambda_rf / 2
    z = np.linspace(zmin, zmax, Np)

    return z, vz


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
f = 13.6 * MHz
Ng = 2
Vg = 7.0 * kV
Vgset = 7.0 * kV
E_DC = 7 * kV / gap_width
dsgn_phase = -0.0 * np.pi
gap_cent_dist = []
Einit = 7.0 * keV
rf_wave = beta(Einit) * SC.c / f
fcup_dist = 10.0 * mm
Energy = [Einit]

# Evaluate and store gap distances and design energy gains.
for i in range(Ng):
    this_dist = calc_pires(Energy[i], freq=f)
    gap_cent_dist.append(this_dist)
    Egain = Vg * np.cos(dsgn_phase)  # Max acceleration
    Energy.append(Egain + Energy[i])

Energy = np.array(Energy)

# Real gap positions are the cumulative sums
gap_cent_dist = np.array(gap_cent_dist)
gap_centers = gap_cent_dist.cumsum()
# Shift gaps by drift space to allow for the field to start from minimum and climb.
zs = beta(Einit) * SC.c / 2.0 / np.pi / f * (dsgn_phase + np.pi)
gap_centers += zs

print("--Gap Centers")
print(gap_centers / mm)

# Create beam and initialize beam parameters
beam = wp.Species(type=wp.Argon, charge_state=+1, name="Argon Beam")
beam.a0 = 0.25 * mm
beam.b0 = 0.25 * mm
beam.emit = 1.344e-6
beam.ap0 = 0.0
beam.bp0 = 0.0
beam.ibeam = 10 * uA
beam.beam = 0.0
beam.ekin = 7.0 * keV
wp.derivqty()

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
wp.w3d.nx = 110

wp.w3d.ymmin = -1.7 * mm
wp.w3d.ymmax = 1.7 * mm
wp.w3d.ny = 110

# use gap positioning to find limits on zmesh. Add some spacing at end points.
# Use enough zpoint to resolve the wafers. In this case, resolve with 2 points.
wp.w3d.zmmin = -rf_wave / 2
wp.w3d.zmmax = gap_centers[-1] + fcup_dist
# Set resolution to be 20um giving 35 points to resolve plates and 100 pts in gap
wp.w3d.nz = round((wp.w3d.zmmax - wp.w3d.zmmin) / 50 / um)
dz = (wp.w3d.zmmax - wp.w3d.zmmin) / wp.w3d.nz

# Set timing step with cf condition.
wp.top.dt = 0.7 * dz / beam.vbeam

# Add boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.periodic


wp.w3d.l4symtry = True
solver = wp.MRBlock3D()
wp.registersolver(solver)


# Create accleration gaps with correct coordinates and settings. Collect in
# list and then loop through and install on the mesh.
do_xoff_cents = False
xoff_cents = np.array([3.0]) * mm
ycents = np.array([0.0]) * mm
conductors = []
for yc in ycents:
    for i, cent in enumerate(gap_centers):
        if i % 2 == 0:
            this_cond = create_gap(cent, left_volt=0, right_volt=Vgset, ycent=yc)
            wp.installconductor(this_cond)
            # cycle through off center gaps
            if do_xoff_cents:
                for xc in xoff_cents:
                    off_cond = create_gap(
                        cent, left_volt=0, right_volt=Vgset, xcent=xc, ycent=yc
                    )
                    wp.installconductor(off_cond)
        else:
            this_cond = create_gap(cent, left_volt=Vgset, right_volt=0, ycent=yc)
            wp.installconductor(this_cond)
            # cycle through off center gaps
            if do_xoff_cents:
                for xc in xoff_cents:
                    off_cond = create_gap(
                        cent, left_volt=Vgset, right_volt=0, xcent=xc, ycent=yc
                    )
                    wp.installconductors(off_cond)

        conductors.append(this_cond)
# for cond in conductors:
#     wp.installconductor(cond)
diagnostic = wp.Box(
    xsize=wp.top.largepos,
    ysize=wp.top.largepos,
    zsize=3.0 * dz,
    voltage=0,
    zcent=wp.w3d.zmmax - 4 * dz,
    xcent=0.0,
    ycent=0.0,
)
scraper = wp.ParticleScraper(diagnostic, lcollectlpdata=True)
wp.top.lsavelostpart = True
# Perform initial field solve for mesh.
wp.package("w3d")
wp.generate()

# Load particles
zload, vzload = uniform_particle_load(Einit, 10000)
beam.addparticles(
    x=np.zeros(len(zload)),
    y=np.zeros(len(zload)),
    z=zload,
    vx=np.zeros(len(zload)),
    vy=np.zeros(len(zload)),
    vz=vzload,
)
# Create trace particle
tracked_ions = wp.Species(type=wp.Argon, charge_state=+1, name="Tracer")
tracker = TraceParticle(
    js=tracked_ions.js, x=0.0, y=0.0, z=0.0, vx=0.0, vy=0.0, vz=beam.vbeam,
)

# Potential particle plots. Cannot get window to stay fixed when plotting
# potential contours the contours move off the plot even though limts are
# held fixed.
wp.setup()
wp.winon()


def beamplots():
    wp.window(0)
    wp.ppzx(titles=False, color="red", msize=3)
    wp.pfzx(titles=False)
    wp.limits(wp.w3d.zmminglobal, wp.w3d.zmmaxglobal, 0.0, 0.55 * mm)
    wp.ptitles("Particles and Potential Contour", "z [m]", "x [m]")
    wp.refresh()
    wp.fma()


# Collect data from the mesh and initialize useful variables.
z = wp.w3d.zmesh
x = wp.w3d.xmesh
y = wp.w3d.ymesh
steps = 1
time = np.zeros(steps)
Ez_array = np.zeros((steps, len(z)))
Ez0 = wp.getselfe(comp="z")[0, 0, :]
phi0 = wp.getphi()[0, 0, :]

# Save arrays
np.save("potential_arrays", phi0)
np.save("field_arrays", Ez0)
np.save("gap_centers", gap_centers)
np.save("zmesh", z)
np.save("xmesh", x)
np.save("ymesh", y)

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
ax.plot(z / mm, phi0 / kV)
ax.set_xlabel("z [mm]")
ax.set_ylabel("Potential [kV]")
ymin, ymax = ax.get_ylim()
for i, cent in enumerate(gap_centers):
    left = cent - gap_width / 2 - length / 2
    right = cent + gap_width / 2 + length / 2
    ax.axvline(x=left / mm, c="gray", lw=0.7)
    ax.axvline(x=right / mm, c="gray", lw=0.7)
    ax.axvspan(left / mm, right / mm, color="grey", alpha=0.5)
ax.axhline(y=0, c="k", lw=1)
plt.savefig("potential", dpi=400)

fig, ax = plt.subplots()
ax.plot(z / mm, Ez0 / E_DC)
ax.set_xlabel("z [mm]")
ax.set_ylabel(r"Normed On-axis E-field $E(x=0, y=0, z)/E_{DC}$ [V/m]")
ymin, ymax = ax.get_ylim()
for i, cent in enumerate(gap_centers):
    left = cent - gap_width / 2 - length / 2
    right = cent + gap_width / 2 + length / 2
    ax.axvline(x=left / mm, c="gray", lw=0.7)
    ax.axvline(x=right / mm, c="gray", lw=0.7)
    ax.axvspan(left / mm, right / mm, color="grey", alpha=0.5)
ax.axhline(y=0, c="k", lw=1)
plt.savefig("Efield", dpi=400)
plt.show()


# Warp plotting for verification that mesh and conductors were created properly.
warpplots = False
if warpplots:
    wp.setup()
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

stop
for i in range(steps):
    Ez = wp.getselfe(comp="z")[0, 0, :]
    Ez_array[i, :] = Ez
    time[i] = wp.top.time
    wp.step()

np.save("Ez_gap_field_151", Ez_array)
np.save("zmesh", z)
np.save(f"time_{steps}", time)

fig, ax = plt.subplots()
ax.axhline(y=1, c="r", lw=1, label="Average DC Field")
ax.plot(
    z / mm, Ez_array[0, :] / E_DC, c="k", label=f"Time: {time_array[0]/ns:.2f} [ns]"
)
ax.set_xlabel("z [mm]")
ax.set_ylabel(fr"On-axis Electric field $E_z(r=0, z,t)/E_{{dc}}$ [kV/mm]")
ax.axvline(x=-zcenter / mm, c="gray", lw=1)
ax.axvline(x=zcenter / mm, c="gray", lw=1)
ax.legend()
plt.show()
