import warpoptions

"""
python3 single-species-simulation.py --esq_voltage=500 --fraction=.8 --speciesMass=20 --ekininit=15e3
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
mm = 1e-3
um = 1e-6
nm = 1e-9
kV = 1e3
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
L_bunch = 1 * ns
lq = 0.696 * mm
Vq = 0.2 * kV
Units = 2
Vmax = 5 * kV
Vesq = 0.1 * kV
V_arrival = 1.0
ekininit = 7 * keV
freq = 13.6 * MHz
period = 1.0 / freq
emittingRadius = 0.25 * mm
aperture = 0.55 * mm
divergenceAngle = 5e-3
ibeaminit = 10 * uA
beamdelay = 0.0
Np_injected = 0

storebeam = warpoptions.options.storebeam
loadbeam = warpoptions.options.loadbeam

first_gapzc = 5 * mm  # First gap center

rf_volt = lambda time: Vmax * np.cos(2.0 * np.pi * freq * time)
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

# Create Species
beam = wp.Species(type=wp.Argon, charge_state=1, name="Ar+", color=wp.blue)
beam.a0 = emittingRadius
beam.b0 = emittingRadius
beam.emit = 3e-6
beam.ap0 = 0.0
beam.bp0 = 0.0
beam.ibeam = ibeaminit
beam.vbeam = 0.0
beam.ekin = ekininit

# keep track of when the particles are born
wp.top.ssnpid = wp.nextpid()
wp.top.tbirthpid = wp.nextpid()

# Set Injection Parameters for injector and beam
wp.top.ns = 1  # numper of species
wp.top.inject = 1  # Constant current injection
wp.top.npinject = 10
wp.top.ainject = emittingRadius
wp.top.binject = emittingRadius
wp.top.apinject = 0.0
wp.top.bpinject = 0.0
wp.top.vinject = 1.0  # source voltage

wp.top.ibeam_s = ibeaminit
wp.top.ekin_s = ekininit
wp.derivqty()
beam.vthz = 700.0


def injection():
    global Np_injected
    Np_inject = np.random.randint(low=75, high=125)
    Np_injected += Np_inject

    beam.add_uniform_cylinder(
        np=Np_inject,
        rmax=emittingRadius,
        zmin=0.0,
        zmax=3.677707206124987e-05,
        vzmean=beam.vbeam,
    )


wp.installuserinjection(injection)

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

tracked_ions = wp.Species(type=wp.Argon, charge_state=1, name="Track", color=wp.red)
tracker = TraceParticle(
    js=tracked_ions.js, x=0.0, y=0.0, z=-3.38 * mm, vx=0.0, vy=0.0, vz=beam.vbeam
)

solver.gridmode = 0  # Temporary fix for fields to oscillate in time.
x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh
ESQs = []
RFs = []
ID_ESQ = 100
ID_RF = 201
ID_target = 1


def rrms():
    x_dis = beam.getx()
    y_dis = beam.gety()

    xrms = np.sqrt(np.mean(x_dis ** 2))
    yrms = np.sqrt(np.mean(y_dis ** 2))
    rrms = np.sqrt(np.mean(x_dis ** 2 + y_dis ** 2))

    print(f" XRMS: {xrms} \n YRMS: {yrms} \n RRMS: {rrms}")

    return rrms


positionArray = np.array([6.76049119, 15.61205193]) * mm - 3.38 * mm

### Functions for automated wafer position by batch running
markedpositions = []
markedpositionsenergies = []

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


lESQ = Mems_ESQ_SolidCyl(15.5 * mm, "1", Vq, -Vq, chop=True)
lESQ.set_geometry(rp=aperture, R=0.68 * aperture, lq=lq)

rESQ = Mems_ESQ_SolidCyl(19.5 * mm, "2", -Vq, Vq, chop=True)
rESQ.set_geometry(rp=aperture, R=0.68 * aperture, lq=lq)

wp.installconductor(lESQ.generate())
wp.installconductor(rESQ.generate())

# Recalculate the fields
wp.fieldsol(-1)

zc_pos = True


def savezcrossing():
    if zc_pos:
        zc_data = {
            "x": zc.getx().tolist(),
            "y": zc.gety().tolist(),
            "z": zc_pos,
            "vx": zc.getvx().tolist(),
            "vy": zc.getvy().tolist(),
            "vz": zc.getvz().tolist(),
            "t": zc.gett().tolist(),
        }
        writejson("zcrossing", zc_data)
        zc_start_data = {
            "x": zc_start.getx().tolist(),
            "y": zc_start.gety().tolist(),
            "z": zc_start_position,
            "vx": zc_start.getvx().tolist(),
            "vy": zc_start.getvy().tolist(),
            "vz": zc_start.getvz().tolist(),
            "t": zc_start.gett().tolist(),
        }
        writejson("zcrossing_start", zc_data)
        print("STORED Z CROSSING")


zmid = 0.5 * (z.max() + z.min())

# Make a circle to show the beam pipe on warp plots in xy.
R = 0.5 * mm  # beam radius
t = np.linspace(0, 2 * np.pi, 100)
X = R * np.sin(t)
Y = R * np.cos(t)
deltaKE = 10e3
time_time = []
numsel = []
KE_select = []
KE_select_Max = []  # modified 4/16
RMS = []

Particle_Counts_Above_E = []  # modified 4/17,
# will list how much Particles have higher E than avg KE at that moment
beamwidth = []
energy_time = []
starting_particles = []
Z = [0]
scraper = wp.ParticleScraper(conductors, lcollectlpdata=True)


def plotf(axes, component, new_page=True):
    if axes not in ["xy", "zx", "zy"]:
        print("error!!!! wrong axes input!!")
        return

    if component not in ["x", "y", "z", "E"]:
        print("Error! Wrong component declared!!!")
        return

    if axes == "xy":
        plotfunc = wp.pfxy
    elif axes == "zy":
        plotfunc = wp.pfzy
    elif axes == "zx":
        plotfunc = wp.pfzx

    plotfunc(
        fill=1,
        filled=1,
        plotselfe=True,
        comp=component,
        titles=0,
        cmin=-1.2 * Vmax / geometry.gapGNDRF,
        cmax=1.2 * Vmax / geometry.gapGNDRF,
    )  # Vmax/geometry.RF_gap

    if component == "E":
        wp.ptitles(axes, "plot of magnitude of field")
    elif component == "x":
        wp.ptitles(axes, "plot of E_x component of field")
    elif component == "y":
        wp.ptitles(axes, " plot of E_y component of field")
    elif component == "z":
        wp.ptitles(axes, "plot of E_z component of field")

    if new_page:
        wp.fma()


if warpoptions.options.loadbeam == "":  # workaround
    wp.step(1)  # This is needed, so that beam exists

# Create cgm windows for plotting
wp.winon(winnum=2, suffix="pzx", xon=False)
wp.winon(winnum=3, suffix="pxy", xon=False)
# wp.winon(winnum=4, suffix="stats", xon=False)

# Calculate various control values to dictate when the simulation ends
velo = np.sqrt(2 * ekininit * beam.charge / beam.mass)
length = positionArray[-1] + 25 * mm
tmax = length / velo  # this is used for the maximum time for timesteps
zrunmax = length  # this is used for the maximum distance for timesteps
scale_maxEz = 1.25
app_maxEz = scale_maxEz * Vmax / geometry.gapGNDRF
if warpoptions.options.runtime:
    tmax = warpoptions.options.runtime

# Create a lab window for the collecting diagnostic data at the end of the run.
# Create zparticle diagnostic. The function gchange is needed to allocate
# arrays for the windo moments. Lastly, create variables for the species index.
zdiagn = ZCrossingParticles(zz=max(z) - 5 * solver.dz, laccumulate=1)
selectind = 0

# Grab number of particles injected.
hnpinj = wp.top.hnpinject[: wp.top.jhist + 1, :]
hnpselected = sum(hnpinj[:, 0])

# Creat array for holding number of particles that cross diagnostics
npdiagn_select = []
vz_select = []
tdiagn_select = []

# Create vertical line for diagnostic visual
pltdiagn_x = np.ones(3) * zdiagn.zz
pltdiagn_y = np.linspace(-wp.largepos, wp.largepos, 3)

# Inject particles for a singple period.
while wp.top.time < 1 * period:
    # Check whether diagnostic arrays are empty
    if zdiagn.getn(selectind) != 0:
        npdiagn_select.append(zdiagn.getn(selectind))
        vz_select.append(zdiagn.getvz(selectind).mean())
        tdiagn_select.append(zdiagn.gett(selectind).mean())

    wp.window(2)
    wp.pfzx(
        fill=1,
        filled=1,
        plotselfe=1,
        comp="z",
        contours=50,
        cmin=-app_maxEz,
        cmax=app_maxEz,
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
tracker_fin_time = tracker.gett()[-1]
# final_time = tracker_fin_time + .1 * period
final_time = 1.5 * period
while wp.top.time < final_time:
    # Check whether diagnostic arrays are empty
    if zdiagn.getn(selectind) != 0:
        npdiagn_select.append(zdiagn.getn(selectind))
        vz_select.append(zdiagn.getvz(selectind).mean())
        tdiagn_select.append(zdiagn.gett(selectind).mean())

    wp.window(2)
    wp.pfzx(
        fill=1,
        filled=1,
        plotselfe=1,
        comp="z",
        contours=50,
        cmin=-app_maxEz,
        cmax=app_maxEz,
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

### END of Simulation
# Grab number of particles injected.
hnpinj = wp.top.hnpinject[: wp.top.jhist + 1, :]
hnpselected = sum(hnpinj[:, 0])
print("Number {} injected: {}".format(beam.name, hnpselected))
npdiagn_select = np.array(npdiagn_select)
vz_select = np.array(vz_select)
tdiagn_select = np.array(tdiagn_select)

# Calculate KE and current statistics
keselect = beam.mass * pow(vz_select, 2) / 2 / wp.jperev

currselect = beam.charge * vz_select * npdiagn_select

# Calculate end of simulation KE for all particles. This will entail grabbing
# values from the lost particle histories.
inslost = wp.top.inslost  # Starting index for each species in the lost arrays
uzlost = wp.top.uzplost  # Vz array for lost particle velocities
Nuz = np.hstack((beam.getvz(), uzlost[inslost[0] : inslost[-1]]))
Nke = beam.mass * pow(Nuz, 2) / 2 / wp.jperev

# Plot statistics. Find limits for axes.
KEmax_limit = max(keselect)
tmin_limit = min(tdiagn_select)
tmax_limit = max(tdiagn_select)
currmax_limit = max(currselect)

# Create plots for kinetic energy, current, and particle counts.
fig, ax = plt.subplots(nrows=3, ncols=1)
keplt = ax[0]
currplt = ax[1]
currplt.sharex(keplt)
kehist = ax[2]

# Make KE plots
keplt.plot(tdiagn_select / ns, keselect / wp.kV, c="b")
keplt.set_xlim(tmin_limit / ns, tmax_limit / ns)
keplt.set_ylim(ekininit / wp.kV, KEmax_limit / wp.kV)
keplt.set_ylabel("Avg KE in z [KeV]")

# Make Current Plots
currplt.plot(tdiagn_select / ns, currselect / 1e-6, c="b")
currplt.set_xlim(tmin_limit / ns, tmax_limit / ns)
currplt.set_ylim(0, currmax_limit / 1e-6)
currplt.set_ylabel(r" Avg Current [$\mu$A]")
currplt.set_xlabel("Time [ns]")

# Make histogram of particle energies for each species
kehist.hist(
    Nke / wp.kV, bins=100, color="b", alpha=0.7, edgecolor="k", linewidth=1, label="Ar+"
)
kehist.set_xlabel("End Energy [KeV]")
kehist.set_ylabel("Number of Particles")
kehist.legend()
plt.tight_layout()
plt.savefig("stats", dpi=300)
plt.show()

###### Final Plots
### Frame, Particle count vs Time Plot
wp.plg(numsel, time_time, color=wp.blue)
wp.ptitles("Particle Count vs Time", "Time (s)", "Number of Particles")
wp.fma()  # fourth to last frame in cgm file
# plot lost particles with respect to Z
wp.plg(beam.lostpars, wp.top.zplmesh + wp.top.zbeam)
wp.ptitles("Particles Lost vs Z", "Z", "Number of Particles Lost")
wp.fma()
### Frame, surviving particles plot:
for i in range(len(numsel)):
    p = max(numsel)
    starting_particles.append(p)
# fraction of surviving particles
f_survive = [i / j for i, j in zip(numsel, starting_particles)]
# want the particles that just make it through the last RF, need position of RF.
# This way we can see how many particles made it through the last important
# component of the accelerator

wp.plg(f_survive, time_time, color=wp.green)
wp.ptitles(
    "Fraction of Surviving Particles vs Time",
    "Time (s)",
    "Fraction of Surviving Particles",
)
wp.fma()
### Frame, rms envelope plot
wp.hpxrms(color=wp.red, titles=0)
wp.hpyrms(color=wp.blue, titles=0)
wp.hprrms(color=wp.green, titles=0)
wp.ptitles("X(red), Y(blue), R(green)", "Time [s]", "X/Y/R [m]", "")
wp.fma()
### Frame, rms envelope plot
wp.pzenvx(color=wp.red, titles=0)
wp.pzenvy(color=wp.blue, titles=0)
wp.ptitles("X(red), Y(blue)", "Z [m]", "X/Y [m]", "")
wp.fma()
### Frame, vx and vy plot
wp.hpvxbar(color=wp.red, titles=0)
wp.hpvybar(color=wp.blue, titles=0)
wp.ptitles("X(red), Y(blue), R(green)", "Time [s]", "X/Y/R [m]", "")
wp.fma()
### Frame, Kinetic Energy at certain Z value
wp.plg(KE_select, time_time, color=wp.blue)
wp.limits(0, 70e-9, 0, 30e3)  # limits(xmin,xmax,ymin,ymax)
wp.ptitles("Kinetic Energy vs Time")
wp.fma()
# ### Frame, rms and kinetic energy vs time
# fig, ax1 = plt.subplots()

# color = "tab:red"
# ax1.set_xlabel("time [s]")
# ax1.set_ylabel("RMS", color=color)
# ax1.plot(time_time, RMS, color=color)
# ax1.tick_params(axis="y", labelcolor=color)

# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

# color = "tab:blue"
# ax2.set_ylabel("Kinetic Energy [eV]", color=color)
# ax2.plot(time_time, KE_select, color=color)
# ax2.tick_params(axis="y", labelcolor=color)

# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()
# plt.savefig("RMS & E vs Time.png")
### Frame, maximal kinetic energy at certain Z value
wp.plg(KE_select_Max, time_time, color=wp.blue)
wp.limits(0, 70e-9, 0, 30e3)  # limits(xmin,xmax,ymin,ymax)
wp.ptitles(" Maximal Kinetic Energy vs Time")
wp.fma()
# kinetic energy plot
wp.plg(KE_select, time_time, color=wp.blue)
wp.ptitles("kinetic energy vs time")
wp.fma()
### kinetic energy plot
wp.plg(KE_select_Max, time_time, color=wp.red)
wp.ptitles("Max kinetic energy vs time")
wp.fma()

wp.plg(Particle_Counts_Above_E, KE_select, color=wp.blue)
wp.ptitles("Particle Count(t) vs Energy(t)")  # modified 4/16
wp.fma()
### Frame, showing ---
KE = beam.getke()
plotmin = np.min(KE) - 1
plotmax = np.max(KE) + 1
plotE = np.linspace(plotmin, plotmax, 20)
listcount = []
for e in plotE:
    elementcount = 0
    for k in KE:
        if k > e:
            elementcount += 1

    listcount.append(elementcount)
wp.plg(listcount, plotE, color=wp.red)
wp.ptitles("Number of Particles above E vs E after last gap ")
C, edges = np.histogram(KE, bins=len(plotE), range=(plotmin, plotmax))
wp.plg(C, plotE)
wp.fma()
#####

### Data storage
# save history information, so that we can plot all cells in one plot

t = np.trim_zeros(wp.top.thist, "b")
hepsny = beam.hepsny[0]
hepsnz = beam.hepsnz[0]
hep6d = beam.hepsx[0] * beam.hepsy[0] * beam.hepsz[0]
hekinz = 1e-6 * 0.5 * wp.top.aion * wp.amu * beam.hvzbar[0] ** 2 / wp.jperev
u = beam.hvxbar[0] ** 2 + beam.hvybar[0] ** 2 + beam.hvzbar[0] ** 2
hekin = 1e-6 * 0.5 * wp.top.aion * wp.amu * u / wp.jperev
hxrms = beam.hxrms[0]
hyrms = beam.hyrms[0]
hrrms = beam.hrrms[0]

hpnum = beam.hpnum[0]

print("debug", t.shape, hepsny.shape)
out = np.stack((t, hepsny, hepsnz, hep6d, hekinz, hekin, hxrms, hyrms, hrrms, hpnum))

rt = (time.time() - start) / 60
print(f"RUNTIME OF THIS SIMULATION: {rt:.0f} minutes")
if warpoptions.options.autorun:
    writejson("runtimeminutes", rt)

### END BELOW HERE IS CODE THAT MIGHT BE USEFUL LATER
# Optional plots:
"""#plot history of scraped particles plot for conductors
wp.plg(conductors.get_energy_histogram)
wp.fma()

wp.plg(conductors.plot_energy_histogram)
wp.fma()

wp.plg(conductors.get_current_history)
wp.fma()

wp.plg(conductors.plot_current_history)
wp.fma()"""

"""wp.plg(t, x, color=wp.green)
wp.ptitles("Spread of survived particles in the x direction")
wp.fma() #last frame -1 in file
"""
