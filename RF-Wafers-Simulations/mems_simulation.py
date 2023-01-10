import warpoptions

"""
python3 single-species-simulation.py --esq_voltage=500 --fraction=.8 --speciesMass=20 --init_E=15e3
"""

#   mass of the ions being accelerated
warpoptions.parser.add_argument(
    "--species_mass", dest="speciesMass", type=int, default="40"
)

#   special cgm name - for mass output / scripting
warpoptions.parser.add_argument("--name", dest="name", type=str, default="multions")

#   special cgm path - for mass output / scripting
warpoptions.parser.add_argument("--path", dest="path", type=str, default="")


#   Volt ratio for ESQs @ToDo Zhihao : is this correct?
warpoptions.parser.add_argument(
    "--volt_ratio", dest="volt_ratio", type=float, default="1.04"
)

#   enables some additional code if True
warpoptions.parser.add_argument("--autorun", dest="autorun", type=bool, default=False)

# sets wp.steps(#)
warpoptions.parser.add_argument("--plotsteps", dest="plotsteps", type=int, default=20)

# changes simulation to a "cb-beam" simulation
warpoptions.parser.add_argument("--cb", dest="cb_framewidth", type=float, default=0)

# enables a Z-Crossing location, saving particles that are crossing the given z-value
warpoptions.parser.add_argument("--zcrossing", dest="zcrossing", type=float, default=0)

# set maximal running time, this disables other means of ending the simulation
warpoptions.parser.add_argument("--runtime", dest="runtime", type=float, default=0)

warpoptions.parser.add_argument(
    "--beamdelay", dest="beamdelay", type=float, default=0.0
)
warpoptions.parser.add_argument("--storebeam", dest="storebeam", default="[]")
warpoptions.parser.add_argument("--loadbeam", dest="loadbeam", type=str, default="")
warpoptions.parser.add_argument("--beamnumber", dest="beamnumber", type=int, default=-1)

# --Python packages
import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as SC
import time
import json
import os
import pdb

# --Import third-party packages
import warp as wp
from warp.particles.extpart import ZCrossingParticles
from warp.particles.singleparticle import TraceParticle


# --Import custom packages
import geometry
from geometry import RF_stack, ESQ_doublet

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

# Utility definitions
name = warpoptions.options.name
beamnumber = warpoptions.options.beamnumber

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

# Utility Functions
def initjson(fp=f"{basepath}{thisrunID}.json"):
    if not os.path.isfile(fp):
        print(f"Saving new Json")
        with open(fp, "w") as writefile:
            json.dump({}, writefile, sort_keys=True, indent=1)


def readjson(fp=f"{basepath}{thisrunID}.json"):
    initjson(fp)
    with open(fp, "r") as readfile:
        data = json.load(readfile)
    return data


def writejson(key, value, fp=f"{basepath}{thisrunID}.json"):
    print(f"Writing Data to json {fp}")
    # print("WRITING DATA")
    # print(f" KEY {key} \n VALUE {value}")
    writedata = readjson(fp)
    writedata[key] = value
    with open(fp, "w") as writefile:
        json.dump(writedata, writefile, sort_keys=True, indent=1)


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
    """ Create a single wafer

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

    def __init__(self, zc, id, V_left, V_right, xc=0.0, yc=0.0, chop=False):

        self.zc = zc
        self.id = id
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
            condid=self.id,
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
                condid=self.id,
            )
            bot = wp.ZCylinder(
                voltage=self.V_left,
                xcent=0.0,
                ycent=neg_cent,
                zcent=self.zc,
                radius=self.R,
                length=self.lq + 4 * self.copper_zlen,
                condid=self.id,
            )

            # Create left and right rods
            left = wp.ZCylinder(
                voltage=self.V_right,
                xcent=neg_cent,
                ycent=0.0,
                zcent=self.zc,
                radius=self.R,
                length=self.lq + 4 * self.copper_zlen,
                condid=self.id,
            )
            right = wp.ZCylinder(
                voltage=self.V_right,
                xcent=pos_cent,
                ycent=0.0,
                zcent=self.zc,
                radius=self.R,
                length=self.lq + 4 * self.copper_zlen,
                condid=self.id,
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
                condid=self.id,
            )
            bot = wp.ZCylinder(
                voltage=self.V_left,
                xcent=0.0,
                ycent=neg_cent - self.R,
                zcent=self.zc,
                radius=self.R,
                length=self.lq + 4 * self.copper_zlen,
                condid=self.id,
            )

            # Create left and right rods
            left = wp.ZCylinder(
                voltage=self.V_right,
                xcent=neg_cent - self.R,
                ycent=0.0,
                zcent=self.zc,
                radius=self.R,
                length=self.lq + 4 * self.copper_zlen,
                condid=self.id,
            )
            right = wp.ZCylinder(
                voltage=self.V_right,
                xcent=pos_cent + self.R,
                ycent=0.0,
                zcent=self.zc,
                radius=self.R,
                length=self.lq + 4 * self.copper_zlen,
                condid=self.id,
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


# ------------------------------------------------------------------------------
#    Script inputs
# Parameter inputs for running the script. Initially were set as command line
# arguments. Setting to a designated section for better organization and
# overview. Eventually this will be stripped and turn into an input file.
# ------------------------------------------------------------------------------

# Specify conductor characteristics
lq = 0.696 * mm
Vq = 0.2 * kV
gap_width = 2 * mm
Vg = 5 * kV
Vq = 0.1 * kV
Ng = 2

# Operating paramters
freq = 13.6 * MHz
period = 1.0 / freq
emittingRadius = 0.25 * mm
aperture = 0.55 * mm
rf_volt = lambda time: Vg * np.cos(2.0 * np.pi * freq * time)

# Beam Paramters
init_E = 7.0 * keV
Tb = 0.1 * eV
init_emit = 3 * mm * mrad
divergenceAngle = 5e-3
init_I = 10 * uA
Np_injected = 0  # initialize injection counter

# ------------------------------------------------------------------------------
#    Mesh setup
# Specify mesh sizing and time stepping for simulation.
# TODO: - set zmmin to be half a wavelength where the trakcer will be placed.
#       The first gap should then be placed commensurate with the tracker and
#       phaseing.
# ------------------------------------------------------------------------------
# Specify  simulation mesh
wp.w3d.xmmax = 1.5 * mm
wp.w3d.xmmin = -wp.w3d.xmmax
wp.w3d.ymmax = wp.w3d.xmmax
wp.w3d.ymmin = -wp.w3d.ymmax

framewidth = 10 * mm
wp.w3d.zmmin = -3.4 * mm
wp.w3d.zmmax = 22 * mm
wp.w3d.nx = 40
wp.w3d.ny = 40
wp.w3d.nz = 200
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
# Create Species
beam = wp.Species(type=wp.Argon, charge_state=1, name="Ar+", color=wp.blue)
beam.a0 = emittingRadius
beam.b0 = emittingRadius
beam.emit = init_emit
beam.ap0 = 0.0
beam.bp0 = 0.0
beam.ibeam = init_I
beam.vbeam = 0.0
beam.ekin = init_E
vth = np.sqrt(2 * Tb * wp.jperev / beam.mass)

# keep track of when the particles are born
wp.top.ssnpid = wp.nextpid()
wp.top.tbirthpid = wp.nextpid()

wp.top.ibeam_s = init_I
wp.top.ekin_s = init_E
wp.derivqty()
wp.top.dt = 0.7 * dz / beam.vbeam
inj_dz = beam.vbeam * wp.top.dt

# Create injection scheme. A uniform cylinder will be injected with each time
# step.
def injection():
    """A uniform injection to be called each time step.
    The injection is done each time step and the width of injection is
    calculated above using vz*dt.
    """
    global Np_injected
    Np_inject = 275
    Np_injected += Np_inject

    beam.add_uniform_cylinder(
        np=Np_inject,
        rmax=emittingRadius,
        zmin=0.0,
        zmax=inj_dz,
        vthx=vth,
        vthy=vth,
        vthz=vth,
        vzmean=beam.vbeam,
    )


wp.installuserinjection(injection)

# ------------------------------------------------------------------------------
#    Mesh setup
# Specify mesh sizing and time stepping for simulation.
# ------------------------------------------------------------------------------
# Specify  simulation mesh
wp.w3d.xmmax = 3 / 2 * mm
wp.w3d.xmmin = -wp.w3d.xmmax
wp.w3d.ymmax = wp.w3d.xmmax
wp.w3d.ymmin = -wp.w3d.ymmax

framewidth = 10 * mm
wp.w3d.zmmin = -3.4 * mm
wp.w3d.zmmax = 22 * mm
wp.w3d.nx = 40
wp.w3d.ny = 40
wp.w3d.nz = 200
dz = (wp.w3d.zmmax - wp.w3d.zmmin) / wp.w3d.nz
dt = 0.2 * ns
wp.top.dt = dt

# Set boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.periodic

wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb
wp.top.prwall = 0.55 * mm

# Setup Histories and moment calculations
wp.top.lspeciesmoments = True
wp.top.itlabwn = 1
wp.top.nhist = 1  # Save history data every N time step
wp.top.itmomnts = wp.top.nhist
wp.top.lhpnumz = True
wp.top.lhcurrz = True
wp.top.lhrrmsz = True
wp.top.lhxrmsz = True
wp.top.lhyrmsz = True
wp.top.lhepsnxz = True
wp.top.lhepsnyz = True
wp.top.lhvzrmsz = True
wp.top.lsavelostpart = True

# Set up fieldsolver
wp.w3d.l4symtry = False
solver = wp.MRBlock3D()
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
    js=tracked_ions.js, x=0.0, y=0.0, z=-3.38 * mm, vx=0.0, vy=0.0, vz=beam.vbeam
)

# Recalculate fields and conductor information every time step. Needed for
# oscillating fields in a moving frame.
solver.gridmode = 0

positionArray = np.array([6.76049119, 15.61205193]) * mm - 3.38 * mm

for i, pa in enumerate(positionArray):
    print(f"Unit {i} placed at {pa}")

conductors = []
for i, pos in enumerate(positionArray):
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

for cond in conductors:
    wp.installconductors(cond)

aperture_wall = wp.ZCylinderOut(
    radius=aperture, zlower=-wp.top.largepos, zupper=wp.top.largepos
)
scraper = wp.ParticleScraper(aperture_wall, lcollectlpdata=True)


# lESQ = Mems_ESQ_SolidCyl(15.5 * mm, "1", Vq, -Vq, chop=True)
# lESQ.set_geometry(rp=aperture, R=0.68 * aperture, lq=lq)

# rESQ = Mems_ESQ_SolidCyl(19.5 * mm, "2", -Vq, Vq, chop=True)
# rESQ.set_geometry(rp=aperture, R=0.68 * aperture, lq=lq)

# wp.installconductor(lESQ.generate())
# wp.installconductor(rESQ.generate())

# Recalculate the fields
wp.fieldsol(-1)

# Make a circle to show the beam pipe on warp plots in xy.
R = 0.5 * mm  # beam radius
t = np.linspace(0, 2 * np.pi, 100)
X = R * np.sin(t)
Y = R * np.cos(t)

scraper = wp.ParticleScraper(conductors, lcollectlpdata=True)

# Create cgm windows for plotting
wp.winon(winnum=2, suffix="pzx", xon=False)
wp.winon(winnum=3, suffix="pxy", xon=False)

# Create a lab window for the collecting diagnostic data at the end of the run.
# Create zparticle diagnostic. The function gchange is needed to allocate
# arrays for the windo moments. Lastly, create variables for the species index.
zdiagn = ZCrossingParticles(zz=max(z) - 5 * solver.dz, laccumulate=1)

# Create vertical line for diagnostic visual
pltdiagn_x = np.ones(3) * zdiagn.zz
pltdiagn_y = np.linspace(-wp.largepos, wp.largepos, 3)

# Inject particles for a singple period.
while wp.top.time < 1 * period:
    wp.window(2)
    wp.pfzx(
        fill=1,
        filled=1,
        plotselfe=1,
        comp="z",
        contours=50,
        cmin=-1.25 * Vg / gap_width,
        cmax=1.25 * Vg / gap_width,
        titlet="Ez, Ar+(Blue) and Track(Red)",
    )
    beam.ppzx(color=wp.blue, msize=2, titles=0)
    wp.plg(pltdiagn_y, pltdiagn_x, width=3, color=wp.magenta)
    wp.plp(tracker.getx()[-1], tracker.getz()[-1], color=wp.red, msize=3)
    wp.limits(z.min(), z.max(), x.min(), x.max())
    wp.fma()

    wp.window(3)
    beam.ppxy(color=wp.blue, msize=2, titlet="Particles Ar+(Blue) and N2+(Red) in XY")
    wp.limits(x.min(), x.max(), y.min(), y.max())
    wp.plg(Y, X, type="dash")
    wp.titlet = "Particles Ar+(Blue) and N2+(Red) in XY"
    wp.fma()

    wp.step(1)

# Turn off injection and then advance particles for 2 period later than current
# time of tracker particle.
wp.top.inject = 0
wp.uninstalluserinjection(injection)
while tracker.getz()[-1] < z[zdiagn.getiz()]:
    if wp.top.it % 5 == 0:
        wp.window(2)
        wp.pfzx(
            fill=1,
            filled=1,
            plotselfe=1,
            comp="z",
            contours=50,
            cmin=-1.25 * Vg / gap_width,
            cmax=1.25 * Vg / gap_width,
            titlet="Ez, Ar+(Blue) and Track(Red)",
        )
        beam.ppzx(color=wp.blue, msize=2, titles=0)
        wp.plg(pltdiagn_y, pltdiagn_x, width=3, color=wp.magenta)
        wp.plp(tracker.getx()[-1], tracker.getz()[-1], color=wp.red, msize=3)
        wp.limits(z.min(), z.max(), x.min(), x.max())
        wp.fma()

        wp.window(3)
        beam.ppxy(
            color=wp.blue, msize=2, titlet="Particles Ar+(Blue) and N2+(Red) in XY"
        )
        wp.limits(x.min(), x.max(), y.min(), y.max())
        wp.plg(Y, X, type="dash")
        wp.titlet = "Particles Ar+(Blue) and N2+(Red) in XY"
        wp.fma()

    wp.step(1)

tracker_fin_time = tracker.gett()[-1]
final_time = tracker_fin_time + 2 * period
while wp.top.time < final_time:
    if wp.top.it % 5 == 0:
        wp.window(2)
        wp.pfzx(
            fill=1,
            filled=1,
            plotselfe=1,
            comp="z",
            contours=50,
            cmin=-1.25 * Vg / gap_width,
            cmax=1.25 * Vg / gap_width,
            titlet="Ez, Ar+(Blue) and Track(Red)",
        )
        beam.ppzx(color=wp.blue, msize=2, titles=0)
        wp.plg(pltdiagn_y, pltdiagn_x, width=3, color=wp.magenta)
        wp.plp(tracker.getx()[-1], tracker.getz()[-1], color=wp.red, msize=3)
        wp.limits(z.min(), z.max(), x.min(), x.max())
        wp.fma()

        wp.window(3)
        beam.ppxy(
            color=wp.blue, msize=2, titlet="Particles Ar+(Blue) and N2+(Red) in XY"
        )
        wp.limits(x.min(), x.max(), y.min(), y.max())
        wp.plg(Y, X, type="dash")
        wp.titlet = "Particles Ar+(Blue) and N2+(Red) in XY"
        wp.fma()

    wp.step(1)

# ------------------------------------------------------------------------------
#    Simulation Diagnostics
# Here the final diagnostics are computed/extracted. The diagnostic plots are
# made.
# ------------------------------------------------------------------------------
Np_delivered = zdiagn.getvz().shape[0]
Efin = beam.mass * pow(zdiagn.getvz(), 2) / 2.0 / wp.jperev
tfin = zdiagn.gett()
frac_delivered = Np_delivered / Np_injected
frac_lost = abs(Np_delivered - Np_injected) / Np_injected

fig, ax = plt.subplots()
ax.set_title("Final Energy Distribution")
ax.set_xlabel("Kinetic Energy (keV)")
ax.set_ylabel("Number of Particles")
ax.hist(Efin / keV, bins=100, edgecolor="k")
plt.show()

fig, ax = plt.subplots()
ax.set_title("Final Time Distribution")
ax.set_xlabel(fr"$\Delta t$ (ns), $t_s$ = {tracker.gett()[-1]/ns:.3f} ns")
ax.set_ylabel("Number of Particles")
ax.hist((tfin - tracker.gett()[-1]) / ns, bins=100, edgecolor="k")
plt.show()
