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
        xsize=cell_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent,
        ycent=ycent + (cell_width / 2 + ravg) / 2,
    )
    l_bot_prong = wp.Box(
        xsize=cell_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent,
        ycent=ycent - (cell_width / 2 + ravg) / 2,
    )
    l_rside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=cell_width,
        zsize=length,
        voltage=left_volt,
        zcent=cent - width / 2 - length / 2,
        xcent=xcent + (cell_width / 2 + ravg) / 2,
        ycent=ycent,
    )
    l_lside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=cell_width,
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
        xsize=cell_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent,
        ycent=ycent + (cell_width / 2 + ravg) / 2,
    )
    r_bot_prong = wp.Box(
        xsize=cell_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent,
        ycent=ycent - (cell_width / 2 + ravg) / 2,
    )
    r_rside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=cell_width,
        zsize=length,
        voltage=right_volt,
        zcent=cent + width / 2 + length / 2,
        xcent=xcent + (cell_width / 2 + ravg) / 2,
        ycent=ycent,
    )
    r_lside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=cell_width,
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

beam.vthz = 0.5 * beam.vbeam * beam.emit / wp.sqrt(beam.a0 * beam.b0)

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
wp.w3d.xmmin = -2 * mm
wp.w3d.xmmax = 2 * mm
wp.w3d.nx = 185

wp.w3d.ymmin = -2 * mm
wp.w3d.ymmax = 2 * mm
wp.w3d.ny = 185

# use gap positioning to find limits on zmesh. Add some spacing at end points.
# Use enough zpoint to resolve the wafers. In this case, resolve with 2 points.
wp.w3d.zmmin = -2 * mm
# wp.w3d.zmmax = gap_centers[-1] + fcup_dist
wp.w3d.zmmax = 2 * mm
# Set resolution to be 20um giving 35 points to resolve plates and 100 pts in gap
wp.w3d.nz = 100

# Add boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.dirichlet


wp.w3d.l4symtry = True
solver = wp.MRBlock3D()
wp.registersolver(solver)

box = wp.Box(
    xsize=2 * mm,
    ysize=2 * mm,
    zsize=2 * mm,
    voltage=7 * kV,
    zcent=0.0 * mm,
    xcent=0.0 * mm,
    ycent=0.0 * mm,
)
cylinder = wp.ZCylinder(
    radius=0.5 * mm,
    length=2 * mm,
    voltage=7 * kV,
    xcent=0.0 * mm,
    ycent=0.0 * mm,
)
conductor = box - cylinder
wp.installconductors(conductor)
wp.package("w3d")
wp.generate()

wp.setup()
wp.pfzx(fill=1, filled=1)
wp.fma()
z = wp.w3d.zmesh
dz = z[1] - z[0]
zcent = np.where((z >= 0.0 - dz) & (z <= 0.0 + dz))[0][0]

wp.pfxy(iz=zcent, fill=1, filled=1)
wp.fma()
