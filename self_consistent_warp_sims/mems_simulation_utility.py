import numpy as np
import scipy.constants as SC
import os
import glob
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import warp as wp

# useful constants
cm = 1e-2
mm = 1e-3
um = 1e-6
ns = 1e-9
mrad = 1e-3
kV = 1e3
keV = 1e3
uA = 1e-6
twopi = 2 * np.pi
Ar_mass_eV = 37.21132474


def create_save_path(dir_name="sim_outputs", prefix="sim"):
    """Sets up the directory to save outputs from main script.

    Directory will be defaulted to sim_outputs/sim### where sim### holds the individual
    outputs from runs starting at sim000 and incrementing.
    This function will ensure the directories are setup first and then will return
    the path.
    """

    # Check that the directory is setup.
    base_dir = os.path.join(os.getcwd(), "sim_outputs")
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # With the base directory set up, search for directories within using the
    # sim### template.
    counter = 0
    while True:
        dir_name = f"{prefix}{counter:03d}"
        dir_path = os.path.join(base_dir, dir_name)

        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_name}")
            return dir_path

        counter += 1


def move_cgms(target, match="*.cgm*"):
    """Move all files in current directory to target that have match in name.

    This is mainly to help move the cgm files to the save path created above.
    This was easier than figuring out how to do that in Warp's ecosystem.
    """
    # Find all files that fit match condition
    source_dir = os.getcwd()
    matching_files = glob.glob(os.path.join(source_dir, match))

    for file_path in matching_files:
        file_name = os.path.basename(file_path)
        target_path = os.path.join(target, file_name)

        # Move files to target directory
        shutil.move(file_path, target_path)
    print(f"Moved cgm files to {target}")


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


def calc_velocity(E, mass):
    """Calculate non-relativistic velocity of particle with energy E and mass

    Energy and mass are assumed to be given in mks units (joules and kg).
    """
    v = np.sqrt(2.0 * E / mass)
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
    )
    bot_prong = wp.Box(
        xsize=prong_width,
        ysize=cell_width / 2 - ravg,
        zsize=length,
        zcent=cent,
        xcent=xcent,
        ycent=ycent - (cell_width / 2 + ravg) / 2,
        voltage=voltage,
    )
    rside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=prong_width,
        zsize=length,
        zcent=cent,
        xcent=xcent + (cell_width / 2 + ravg) / 2,
        ycent=ycent,
        voltage=voltage,
    )
    lside_prong = wp.Box(
        xsize=cell_width / 2 - ravg,
        ysize=prong_width,
        zsize=length,
        zcent=cent,
        xcent=xcent - (cell_width / 2 + ravg) / 2,
        ycent=ycent,
        voltage=voltage,
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


def calc_gap_centers(E_s, mass, phi_s, gap_mode, dsgn_freq, dsgn_gap_volt):
    """Calculate Gap centers based on energy gain and phaseing.

    When the gaps are operating at the same phase then the distance is
    beta*lambda / 2. However, if the gap phases are different then there is
    additional distance that the particle must cover to arrive at this new phase.

    The initial phasing is assumed to start with the first gap having a minimum
    E-field (maximally negative). If this is not desired, the offset should be
    calculated outside this function.

    Parameters:
    -----------
    E_s: float
        Initial beam energy in units of (eV).

    mass: float
        Mass of the ion in question in unites of (eV).

    phi_s: list or array
        The synchronous phases of arrival for each gap. Assumed to be rising in
        magnitude with the max phase being <= 0. In units of (radians).

    gap_mode: list or array
        Integer values that will give the n-value for 2npi if additional spacing is
        needed.

    dsgn_freq: float
        Operating frequency of the gaps. Assumed constant througout.

    dsgn_gap_volt: float
        Operating voltage of gaps. Assumed constant throughout.

    Returns:
    --------
    gap_centers: array
        Cumulative sum of the gap distances (center-to-center) for the lattice.
    """

    # Initialize arrays and any useful constants.
    gap_dist = np.zeros(len(phi_s))
    h = SC.c / dsgn_freq

    # Loop through number of gaps and assign the center-to-center distance.
    # Update the energy at end.
    for i in range(len(phi_s)):
        this_beta = beta(E_s, mass)
        this_cent = this_beta * h / 2.0
        shift = this_beta * h * gap_mode[i]
        cent_offset = (phi_s[i] - phi_s[i - 1]) * this_beta * h / twopi
        if i < 1:
            gap_dist[i] = (phi_s[i] + np.pi) * this_beta * h / twopi + shift
        else:
            gap_dist[i] = this_cent + cent_offset + shift

        dsgn_Egain = dsgn_gap_volt * np.cos(phi_s[i])
        E_s += dsgn_Egain

    # gap locations from zero are given by cumulative sum of center-to-center
    # distances.
    return gap_dist.cumsum()


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


def calc_zmatch_sect(lq, d, Nq=4):
    """Calculate the z-centers to place quadrupoles"""
    zcents = np.empty(Nq)
    for i in range(Nq):
        this_zcent = d + 2 * i * d + lq * i + lq / 2
        zcents[i] = this_zcent

    return zcents


def calc_phase_shift(freq, distance, vbeam):
    """Calculate the phase shift needed to maintain RF resonance"""
    phase = twopi * freq * distance / vbeam

    return phase


def make_dist_plot(
    xdata,
    ydata,
    xlabel="",
    ylabel="",
    auto_clip=True,
    xclip=(None, None),
    yclip=(None, None),
    levels=30,
    bins=50,
    xref=None,
    yref=None,
    weight=None,
    dx_bin=None,
    dy_bin=None,
):
    """Quick Function to make a joint plot for the distribution of x and y data.

    The function uses seaborns JointGrid to create a KDE plot on the main figure and
    histograms of the xdata and ydata on the margins.

    """
    if auto_clip:
        cut = 0
    with sns.axes_style("darkgrid"):
        g = sns.JointGrid(x=xdata, y=ydata, marginal_ticks=True, height=6, ratio=2)
        if auto_clip:
            g.plot_joint(sns.kdeplot, levels=levels, fill=True, cmap="flare", cut=0)
        else:
            g.plot_joint(
                sns.kdeplot, fill=True, levels=levels, cmap="flare", clip=(xclip, yclip)
            )
        sns.histplot(
            x=xdata,
            bins=bins,
            edgecolor="k",
            lw=0.5,
            alpha=0.7,
            stat="count",
            weights=weight,
            binwidth=dx_bin,
            ax=g.ax_marg_x,
        )
        sns.histplot(
            y=ydata,
            bins=bins,
            edgecolor="k",
            lw=0.5,
            alpha=0.7,
            stat="count",
            weights=weight,
            binwidth=dy_bin,
            ax=g.ax_marg_y,
        )
        if xref != None:
            g.refline(x=xref)
        if yref != None:
            g.refline(y=yref)

        g.set_axis_labels(
            xlabel=r"Relative Time Difference $\Delta t / \tau_{rf}$",
            ylabel=rf"Kinetic Energy $\mathcal{{E}}$ (keV)",
        )
        g.ax_marg_x.set_ylabel(r"Counts/$N_p$")
        g.ax_marg_y.set_xlabel(r"Counts/$N_p$")

    plt.tight_layout()

    return g


def zdiagnostics(
    diagnostic_list,
    trace_particle,
    mass,
    file_path,
    file_name="Zcross_diagnostics.pdf",
    mask_E=None,
    mask_t=None,
    mask_rperp=None,
    mask_vperp=None,
):
    """Loop through z-diagnostics and compute/plot various metrics

    The function takes in a list of diagnostic objects that are assumed to be
    the ZCrossingParticles objects defined in Warp.
    For each diagnostic a windowing a selection can be done to grab particles
    based on energy, time, position, etc.
    Masking is done relative to design or trace particle. The masks are assumed
    to be symmetric and a fraction of the design particle. Thus, the maskE will
    selected particles with mask = (-mask_E, +mask_E).

    """

    # Initialize pdf to save plots to
    pdf = PdfPages(f"{file_name}")

    # Begin looping through diagnostics
    for i, diag in enumerate(diagnostic_list):
        masks = []
        # Finder tracker index to grab tracker values at this locations
        tracker_ind = np.argmin(abs(tracker.getz() - diag.getzz()))

        # Convert the vz values to energy
        this_E = mass * pow(diagn.getvz() / SC.c, 2) / 2.0
        this_Es = mass * pow(tracker.getvz()[tracker_ind] / SC.c, 2) / 2.0

        # Work through mask options. Make a mask for each then combine at end.
        if mask_E:
            this_mask_E = abs(this_E - this_Es) < mask_E
            masks.append(this_mask_E)

        if mask_t:
            this_mask_t = abs(diag.gett() - tracker.gett()[tracker_ind]) < mask_t
            masks.append(this_mask_t)

        if mask_rperp:
            this_rperp = np.sqrt(pow(diagn.getx(), 2) + pow(diagn.gety(), 2))
            this_mask_rperp = this_rperp < mask_r_perp
            masks.append(this_mask_rperp)

        if mask_vperp:
            this_vperp = np.sqrt(pow(diagn.getvx(), 2) + pow(diagn.getvy(), 2))
            this_mask_vperp = this_vperp < mask_vperp
            masks.append(this_mask_vperp)

        # Combine all masks if any were created. If not, create a mask of all true
        if len(masks) > 0:
            mask = np.logical_and.reduce(masks)
        else:
            mask = np.full(len(this_E), True, dtype=bool)

        # Grab masked particle coordinates
        this_x = diag.getx()[mask]
        this_y = diag.gety()[mask]
        this_z = diag.getz()[mask]
        this_vx = diag.getvx()[mask]
        this_vy = diag.getvy()[mask]
        this_vz = diag.getvz()[mask]
        this_E = 0.5 * mass * pow(diagn.getvz() / SC.c, 2)
