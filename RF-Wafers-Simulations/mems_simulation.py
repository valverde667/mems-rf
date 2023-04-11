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

start = time.time()

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


def beta(E, mass, q=1, nonrel=True):
    """Velocity of a particle with energy E."""
    if nonrel:
        sign = np.sign(E)
        beta = np.sqrt(2 * abs(E) / mass)
        beta *= sign
    else:
        gamma = (E + mass) / mass
        beta = np.sqrt(1 - 1 / gamma / gamma)

    return beta


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
    """Create a single wafer

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


class Mems_ESQ_SolidCyl:
    """
    Creates ESQ conductor based on MEMS design.

    The ESQ (at the moment) is created using solid cylindrical rods. The rods
    have a radius R and can be chopped, i.e. half-rod. The aperture radius rp
    remains fixed at 0.55 mm. Surrounding the ESQ is a rectangular conductor
    that makes up the cell. This cell is 3 mm x 3 mm in x and y. The transverse
    thickness is set to be 0.2 mm.
    Longitudinal (z-direction) the ESQs have length lq while the cell is given
    a z-length of copper_zlen. On each cell is a pair of prongs that connect
    to the ESQs. The first cell has two prongs in +/- y that connect to the
    top and bottom rods. The second cell has prongs in +/- x that connect to the
    left and right prongs. These sets of prongs, along with the cell, have
    opposing biases.

    Attributes
    ----------
    zc : float
        Center of electrode. The extent of the electrode is the half-total
        length in the positive and negative direction of zc.
    id : str
        An ID string for the conductor created. Setting all these to be
        identical will ensure the addition and subtraction of conductors is
        done correctly.
    V_left : float
        The voltage setting on the first cell that will also be the bias of the
        top and bottom rods.
    V_right : float
        The voltage setting on the second cell that will also be teh biase of
        the left and right rods.
    xc : float
        The center location of the rods in x.
    yc : float
        The center location of the rods in y.
    rp : float
        Aperture radius.
    lq : float
        Phycial length (longitudinally) of the ESQ rods.
    R : float
        Radius of the ESQ rods.
    copper_zlen : float
        Longitudinal length of the surrounding metallic structure.
    copper_xysize : float
        The size in x and y of the surrounding square conductor.
    prong_short_dim : float
        The non-connecting size of the prong connecting to the ESQ.
    prong_long_dim : float
        The connecting size of the prong connecting to the ESQ.

    Methods
    -------
    set_geometry
        Initializes geometrical settings for the ESQ wafer. This needs to be
        called each time the class is created in order to set the values.
    create_outer_box
        Creates the surrounding square conducting material along with the prongs
        that connect to the ESQ rods.
    create_rods
        Creates the ESQ rods.
    generate
        Combines the square conductor and the rods together to form the ESQ
        structure. This and teh set_geometry method should be the only calls
        done.
    """

    def __init__(self, zc, V_left, V_right, xc=0.0, yc=0.0, chop=False):

        self.zc = zc
        self.V_left = V_left
        self.V_right = V_right
        self.xc = 0.0
        self.yc = 0.0
        self.rp = None

        self.lq = None
        self.R = None
        self.chop = chop

        # Surrounding square dimensions.
        self.copper_zlen = None
        self.cell_xysize = None

        # Prong length that connects rods to surrounding conductor
        self.prong_short_dim = 0.2 * mm
        self.prong_long_dim = 0.2 * mm
        pos_xycent = None
        neg_xycent = None

    def set_geometry(
        self,
        rp=0.55 * mm,
        R=0.8 * mm,
        lq=0.695 * mm,
        copper_zlen=35 * um,
        cell_xysize=3.0 * mm,
    ):
        """Initializes the variable geometrical settings for the ESQ."""

        self.rp = rp
        self.R = R
        self.lq = lq
        self.copper_zlen = copper_zlen
        self.cell_xysize = cell_xysize

    def create_outer_box(self):
        """Create surrounding square conductors.

        The ESQs are surrounded by a square conductor. Because of the opposing
        polarities the surrounding square takes one voltage on one side of the
        wafer and the opposing bias on the other side. This function creates
        the +/- polarity.
        """

        l_zc = self.zc - self.lq / 2.0 - self.copper_zlen
        r_zc = self.zc + self.lq / 2.0 + self.copper_zlen

        size = self.cell_xysize

        l_box_out = wp.Box(
            xsize=size,
            ysize=size,
            zsize=self.copper_zlen,
            zcent=l_zc,
            xcent=self.xc,
            ycent=self.yc,
            voltage=self.V_left,
        )
        l_box_in = wp.Box(
            xsize=size - 0.1 * mm,
            ysize=size - 0.1 * mm,
            zsize=self.copper_zlen,
            zcent=l_zc,
            xcent=self.xc,
            ycent=self.yc,
            voltage=self.V_left,
            condid=l_box_out.condid,
        )

        # Create connecting prongs for top and bottom ESQ
        top_prong = wp.Box(
            xsize=self.prong_short_dim,
            ysize=self.prong_long_dim,
            zsize=self.copper_zlen,
            voltage=self.V_left,
            zcent=l_zc,
            xcent=0.0,
            ycent=size / 2 - 0.1 * mm - self.prong_long_dim / 2 + 0.05 * mm,
            condid=l_box_out.condid,
        )
        bot_prong = wp.Box(
            xsize=self.prong_short_dim,
            ysize=self.prong_long_dim,
            zsize=self.copper_zlen,
            voltage=self.V_left,
            zcent=l_zc,
            xcent=0.0,
            ycent=-size / 2 + 0.1 * mm + self.prong_long_dim / 2 - 0.05 * mm,
            condid=l_box_out.condid,
        )

        l_this_box = l_box_out - l_box_in
        l_box = l_this_box + top_prong + bot_prong

        r_box_out = wp.Box(
            xsize=size,
            ysize=size,
            zsize=self.copper_zlen,
            zcent=r_zc,
            xcent=self.xc,
            ycent=self.yc,
            voltage=self.V_right,
            condid=l_box_out.condid,
        )
        r_box_in = wp.Box(
            xsize=size - 0.1 * mm,
            ysize=size - 0.1 * mm,
            zsize=self.copper_zlen,
            zcent=r_zc,
            xcent=self.xc,
            ycent=self.yc,
            voltage=self.V_right,
            condid=l_box_out.condid,
        )

        r_prong = wp.Box(
            xsize=self.prong_long_dim,
            ysize=self.prong_short_dim,
            zsize=self.copper_zlen,
            voltage=self.V_right,
            zcent=r_zc,
            xcent=size / 2 - 0.1 * mm - self.prong_long_dim / 2 + 0.05 * mm,
            ycent=0.0,
            condid=l_box_out.condid,
        )
        l_prong = wp.Box(
            xsize=self.prong_long_dim,
            ysize=self.prong_short_dim,
            zsize=self.copper_zlen,
            voltage=self.V_right,
            zcent=r_zc,
            xcent=-size / 2 + 0.1 * mm + self.prong_long_dim / 2 - 0.05 * mm,
            ycent=0.0,
            condid=l_box_out.condid,
        )
        r_this_box = r_box_out - r_box_in
        r_box = r_this_box + r_prong + l_prong

        box = l_box + r_box

        return box

    def create_rods(self):
        """Create the biased rods"""

        size = self.cell_xysize
        pos_cent = size / 2.0 - 0.1 * mm - self.prong_long_dim - self.R + 0.07 * mm
        neg_cent = -size / 2.0 + 0.1 * mm + self.prong_long_dim + self.R - 0.07 * mm
        self.pos_xycent = pos_cent
        self.neg_xycent = neg_cent

        # assert cent - self.R >= self.rp, "Rods cross into aperture."
        # assert cent + self.R <= 2.5 * mm, "Rod places rods outside of cell."
        # assert 2. * pow(self.R + self.rp, 2) > 4. * pow(self.R, 2), "Rods touching."

        if self.chop == False:

            # Create top and bottom rods
            top = wp.ZCylinder(
                voltage=self.V_left,
                xcent=0.0,
                ycent=pos_cent,
                zcent=self.zc,
                radius=self.R,
                length=self.lq + 4 * self.copper_zlen,
            )
            bot = wp.ZCylinder(
                voltage=self.V_left,
                xcent=0.0,
                ycent=neg_cent,
                zcent=self.zc,
                radius=self.R,
                length=self.lq + 4 * self.copper_zlen,
            )

            # Create left and right rods
            left = wp.ZCylinder(
                voltage=self.V_right,
                xcent=neg_cent,
                ycent=0.0,
                zcent=self.zc,
                radius=self.R,
                length=self.lq + 4 * self.copper_zlen,
            )
            right = wp.ZCylinder(
                voltage=self.V_right,
                xcent=pos_cent,
                ycent=0.0,
                zcent=self.zc,
                radius=self.R,
                length=self.lq + 4 * self.copper_zlen,
            )

            conductor = top + bot + left + right

        else:

            l_zc = self.zc - self.lq / 2.0 - self.copper_zlen
            r_zc = self.zc + self.lq / 2.0 + self.copper_zlen
            # Chop the rounds and then recenter to adjust for chop
            chop_box_out = wp.Box(
                xsize=10 * mm,
                ysize=10 * mm,
                zsize=10 * mm,
                zcent=l_zc,
                xcent=self.xc,
                ycent=self.yc,
            )
            chop_box_in = wp.Box(
                xsize=2.6 * mm,
                ysize=2.6 * mm,
                zsize=10 * mm,
                zcent=l_zc,
                xcent=self.xc,
                ycent=self.yc,
            )
            # Create top and bottom rods
            top = wp.ZCylinder(
                voltage=self.V_left,
                xcent=0.0,
                ycent=pos_cent + self.R,
                zcent=self.zc,
                radius=self.R,
                length=self.lq + 4 * self.copper_zlen,
            )
            bot = wp.ZCylinder(
                voltage=self.V_left,
                xcent=0.0,
                ycent=neg_cent - self.R,
                zcent=self.zc,
                radius=self.R,
                length=self.lq + 4 * self.copper_zlen,
            )

            # Create left and right rods
            left = wp.ZCylinder(
                voltage=self.V_right,
                xcent=neg_cent - self.R,
                ycent=0.0,
                zcent=self.zc,
                radius=self.R,
                length=self.lq + 4 * self.copper_zlen,
            )
            right = wp.ZCylinder(
                voltage=self.V_right,
                xcent=pos_cent + self.R,
                ycent=0.0,
                zcent=self.zc,
                radius=self.R,
                length=self.lq + 4 * self.copper_zlen,
            )

            chop_box = chop_box_out - chop_box_in
            top = top - chop_box
            top.ycent = top.ycent + self.R
            bot = bot - chop_box
            bot.ycent = bot.ycent - self.R
            left = left - chop_box
            left.xcent = left.xcent - self.R
            right = right - chop_box
            right.xcent = right.xcent + self.R

            conductor = top + bot + left + right

        return conductor

    def generate(self):
        """Combine four electrodes to form ESQ."""
        # Create four poles
        square_conds = self.create_outer_box()
        rods = self.create_rods()

        conductor = square_conds + rods

        return conductor


class Data_Ext:
    """Extract the data from the lab windows in top and plot

    The lab windows are given their own index on creation. They can be kept
    in a list and then cycled through to grap the various data. This class
    will cycle through the list, grab the data, store it, and will have further
    capabilities to plot, save, etc.
    """

    def __init__(self, lab_windows, zcrossings, beam, trace_particle):
        self.lws = lab_windows
        self.zcs = zcrossings
        self.beam = beam
        self.trace_particle = trace_particle
        self.data_lw = {}
        self.data_zcross = {}
        self.save_path = os.getcwd()

        # Create lab window names and initialize empty dictionary
        nlw = len(self.lws)
        lw_names = [f"ilw{i+1}" for i in range(nlw)]
        for i in range(nlw):
            key = lw_names[i]
            this_dict = {key: {}}
            self.data_lw.update(this_dict)

        nzc = len(self.zcs)
        zc_names = [f"zc{i+1}" for i in range(nzc)]
        for i in range(nzc):
            key = zc_names[i]
            this_dict = {key: {}}
            self.data_zcross.update(this_dict)

        # List of keys for extracting data. This is used to make the dictionary
        # names and not to navigate through the lab window data features. If
        # a new key is added, be sure to modify the grab_data function to grap.
        self.data_lw_keys = [
            "I",
            "xrms",
            "yrms",
            "xprms",
            "yprms",
            "vxrms",
            "vyrms",
            "vzrms",
            "emitx",
            "emity",
            "emitx_n",
            "emity_n",
            "time",
        ]

        # Create a dictionary of values that will hold scale factors and names
        # for scaling the respective data and naming the plots
        self.scale_factors = {
            "I": (uA, "Current (uA)"),
            "xrms": (mm, "xrms (mm)"),
            "yrms": (mm, "yrms (mm)"),
            "xprms": (mrad, "xprms (mrad)"),
            "yprms": (mrad, "yprms (mrad)"),
            "vxrms": (um / ns, "vxrms (um/ns)"),
            "vyrms": (um / ns, "vyrms (um/ns)"),
            "vzrms": (um / ns, "vzrms (mm/ns)"),
            "emitx": (mm * mrad, "emitx (mm-mrad)"),
            "emity": (mm * mrad, "emity (mm-mrad)"),
            "emitx_n": (mm * mrad, "emitx_n (mm-mrad)"),
            "emity_n": (mm * mrad, "emity_n (mm-mrad)"),
            "time": (ns, "time (ns)"),
        }

    def grab_data(self):
        """Iterate through lab windows and extract data"""

        # iterate through lab window data and assign to dictionary entry.
        for i, key in enumerate(self.data_lw.keys()):
            this_lw = self.lws[i]
            this_I = wp.top.currlw[: wp.top.ilabwn[this_lw, 0], this_lw, 0]
            this_xrms = wp.top.xrmslw[: wp.top.ilabwn[this_lw, 0], this_lw, 0]
            this_yrms = wp.top.yrmslw[: wp.top.ilabwn[this_lw, 0], this_lw, 0]
            this_xprms = wp.top.xprmslw[: wp.top.ilabwn[this_lw, 0], this_lw, 0]
            this_yprms = wp.top.yprmslw[: wp.top.ilabwn[this_lw, 0], this_lw, 0]
            this_vxrms = wp.top.vxrmslw[: wp.top.ilabwn[this_lw, 0], this_lw, 0]
            this_vyrms = wp.top.vyrmslw[: wp.top.ilabwn[this_lw, 0], this_lw, 0]
            this_vzrms = wp.top.vzrmslw[: wp.top.ilabwn[this_lw, 0], this_lw, 0]
            this_emitx = wp.top.epsxlw[: wp.top.ilabwn[this_lw, 0], this_lw, 0]
            this_emity = wp.top.epsylw[: wp.top.ilabwn[this_lw, 0], this_lw, 0]
            this_emitx_n = wp.top.epsnxlw[: wp.top.ilabwn[this_lw, 0], this_lw, 0]
            this_emity_n = wp.top.epsnylw[: wp.top.ilabwn[this_lw, 0], this_lw, 0]
            this_time = wp.top.timelw[: wp.top.ilabwn[this_lw, 0], this_lw, 0]

            # Collect into single list
            vals = [
                this_I,
                this_xrms,
                this_yrms,
                this_xprms,
                this_yprms,
                this_vxrms,
                this_vyrms,
                this_vzrms,
                this_emitx,
                this_emity,
                this_emitx_n,
                this_emity_n,
                this_time,
            ]

            # Populate dictionary entry for this lab window
            this_dict_lw = dict(zip(self.data_lw_keys, vals))
            self.data_lw[key] = this_dict_lw

    def plot_hist(self, data, bins, scalex):
        """Plot binned data with fractional counts"""
        counts, edges = np.histogram(data, bins=bin)
        N = np.sum(counts)
        fig, ax = plt.subplots()
        ax.bar(
            edges[:-1] / scalex,
            counts[:] / N,
            width=np.diff(edges[:] / scalex),
            edgecolor="black",
            lw=1,
        )
        ax.set_ylabel("Fractional Counts")

        return (fig, ax)

    def plot_data(self, save_path=""):
        # Check for new save path
        if len(save_path) != 0:
            self.save_path = save_path
        else:
            # Create file name with datetime format
            now = datetime.datetime.now()
            fmt = "%Y-%m-%d_%H:%M:%S"
            fname_prefix = datetime.datetime.strftime(now, fmt)
            path = os.join(sefl.save_path, fname_prefix)

        # Loop through lab windows and export plots to pdf.


def calc_zESQ(zgaps, zFcup, d=3.0 * mm, lq=0.695 * mm):
    """Function to calculate ESQ doublet positions with separation d and voltage Vq

    The ESQs will be arranged as a doublet with edge-to-edge separation distance
    given by d. The current lattice configuration dictates that an ESQ can be
    inserted in the field-free region in the second gap (after every two RF
    gaps). The last ESQ doublet is placed in between the Fcup and edge ESQ edge.
    Note that in order for the ESQs to a length d of fringe field between one anotther
    the inter-spacing must be 2*d.

    The voltage can be given as a float value in which case all ESQs have the
    same voltage with alternating sign. Or, an array of voltages can be given
    specifying the voltage for each ESQ.
    """
    Ng = len(zgaps)
    # Loop through the gap positions and place ESQs. The RF gaps will always come
    # pairs and the field free region comes after the second ESQ or every odd
    # element in hte zgaps list.
    esq_pos = []
    for i in range(Ng - 2):
        if i % 2 != 0:
            # Calculate the center of the field free region.
            zc = (zgaps[i] + zgaps[i + 1]) / 2.0
            z1 = zc - d - lq / 2.0
            z2 = zc + d + lq / 2.0
            esq_pos.append(z1)
            esq_pos.append(z2)

    # Do the final position between the last gap and the Fcup
    zc = (zgaps[-1] + zFcup) / 2.0
    z1 = zc - d - lq / 2.0
    z2 = zc + d + lq / 2.0
    esq_pos.append(z1)
    esq_pos.append(z2)

    return np.array(esq_pos)


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
Ng = 4
Fcup_dist = 10 * mm

# Operating parameters
freq = 13.6 * MHz
hrf = SC.c / freq
period = 1.0 / freq
emittingRadius = 0.25 * mm
aperture = 0.55 * mm

# TODO: the time shift was calculated by hand and added here. This should be
#       fixed to be modular with user settings.
# Shift to include more beam so more closely resembles continuous injection.
rf_volt = lambda time: Vg * np.cos(twopi * freq * (time - 1.165 * period) + np.pi)

# Beam Paramters
init_E = 7.0 * keV
Tb = 0.1  # eV
div_angle = 3.78 * mrad
init_I = 10 * uA
Np_injected = 0  # initialize injection counter
Np_max = int(1e5)

# Sepcify the design phase.
dsgn_phase = -np.pi / 3.0

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
    this_beta = beta(E_s, mass_eV)
    this_cent = this_beta * SC.c / 2 / freq
    cent_offset = (phi_s[i] - phi_s[i - 1]) * this_beta * SC.c / freq / twopi
    if i < 1:
        gap_dist[i] = (phi_s[i] + np.pi) * this_beta * SC.c / twopi / freq
    else:
        gap_dist[i] = this_cent + cent_offset

    dsgn_Egain = Vg * np.cos(phi_s[i])
    E_s += dsgn_Egain

gap_centers = np.array(gap_dist).cumsum()
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

# Max and min here will determine length of lab window.
wp.w3d.zmmin = -18 * mm
wp.w3d.zmmax = 38 * mm
wp.w3d.nx = 40
wp.w3d.ny = 40
wp.w3d.nz = 250
lab_center = (wp.w3d.zmmax + wp.w3d.zmmin) / 2.0
dz = (wp.w3d.zmmax - wp.w3d.zmmin) / wp.w3d.nz

# Set boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.periodic

wp.top.pbound0 = wp.absorb
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
wp.top.zinject = -9 * mm

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
        vthx=vth / 4,
        vthy=vth / 4,
        vthz=vth / 4,
        vzmean=beam.vbeam,
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

# Loop through gap_centers and place diagnostics at center point between gaps.
for i in range(Ng - 1):
    zloc = (gap_centers[i + 1] + gap_centers[i]) / 2.0
    ilws.append(wp.addlabwindow(zloc))
    zdiagns.append(ZCrossingParticles(zz=zloc, laccumulate=1))

ilws.append(wp.addlabwindow(gap_centers[-1] + Fcup_dist))
zdiagns.append(ZCrossingParticles(zz=gap_centers[-1] + Fcup_dist, laccumulate=1))

g1 = GridCrossingDiags(zmmin=0.0, zmmax=wp.w3d.zmmax - 5 * mm, lmoving_frame=True)
g2 = GridCrossingDiags(
    zmmin=zdiagns[-1].getzz() - 10 * dz,
    zmmax=zdiagns[-1].getzz() - 5 * dz,
    starttime=385 * ns,
)

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
tracked_ions = wp.Species(type=wp.Argon, charge_state=1, name="Track", color=wp.red)
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
        this_lcond = create_wafer(zl, voltage=0.0)
        this_rcond = create_wafer(zr, voltage=rf_volt)
    else:
        this_lcond = create_wafer(zl, voltage=rf_volt)
        this_rcond = create_wafer(zr, voltage=0.0)

    conductors.append(this_lcond)
    conductors.append(this_rcond)

# Create matching section consisting of four quadrupoles
# TODO: Devise better placement scheme rather than hardcode.
start_match = wp.top.zinject + 1.5 * mm
match_pos = np.array([-7.1525, -3.9575, -0.7625, 2.4325,]) * mm
for i, pos in enumerate(match_pos):
    this_zc = pos
    if i % 2 == 0:
        this_Vq = -Vq

    this_ESQ = Mems_ESQ_SolidCyl(this_zc, this_Vq, -this_Vq, chop=True)
    this_ESQ.set_geometry(rp=aperture, R=1.3 * aperture, lq=lq)
    this_cond = this_ESQ.generate()
    conductors.append(this_cond)

for cond in conductors:
    wp.installconductors(cond)

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

# Calculate ESQ center positions and then install.
esq_pos = calc_zESQ(gap_centers, gap_centers[-1] + Fcup_dist, lq=lq)

# Loop through ESQ positions and place ESQs with alternating bias
Vq_list = np.ones(shape=len(esq_pos))
Vq_list[::1] *= -Vq  # Alternate signs in list
for i, pos in enumerate(esq_pos):
    this_ESQ = Mems_ESQ_SolidCyl(pos, Vq_list[i], -Vq_list[i], chop=True)
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
while tracker.getz()[-1] < 0.5 * (wp.w3d.zmmax + wp.w3d.zmmin):
    if wp.top.it % 5 == 0:
        wp.window(2)
        plotbeam(lplt_tracker=True)
        wp.fma()

    wp.step(1)

# Turn off injection once grid starts moving
wp.top.inject = 0
wp.uninstalluserinjection(injection)

for i in range(Ng - 2):
    wp.top.dt = 0.7 * dz / tracker.getvz()[-1]
    while tracker.getz()[-1] < gap_centers[i + 2]:
        wp.top.vbeamfrm = tracker.getvz()[-1]
        if wp.top.it % 5 == 0:
            wp.window(2)
            plotbeam(lplt_tracker=True)
            wp.fma()

        wp.step(1)

wp.top.vbeamfrm = tracker.getvz()[-1]
wp.top.dt = 0.7 * dz / tracker.getvz()[-1]
while tracker.getz()[-1] < zdiagns[-1].getzz():
    if wp.top.it % 5 == 0:
        wp.window(2)
        plotbeam(lplt_tracker=True)
        wp.fma()

    wp.step(1)

tracker_fin_time = tracker.gett()[-1]
final_time = tracker_fin_time + 1 * period
wp.top.vbeamfrm = 0.0
while wp.top.time < final_time:
    if wp.top.it % 5 == 0:
        wp.window(2)
        plotbeam(lplt_tracker=True)
        wp.fma()

    wp.step(1)

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

Data = Data_Ext(ilws, zdiagns, beam, tracker)
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


print(f"Elapsed time: {(time.time() - start) / 60:.2f} (min)")
