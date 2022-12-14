# This script is a testing bed for the creating and viewing the ESQ geometries.
# The real geometry has one side of the wafer with metal connecting to the
# positive rods. The other side of the wafer has metal connecting to the
# negative rods.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as mpl
from mpl_toolkits.mplot3d import axes3d
import scipy.integrate as integrate
import os
import math
import csv
import pdb
import sys

import warp as wp

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

# Useful constants
kV = wp.kV
mm = wp.mm
um = 1e-6


class ESQ_SolidCyl:
    """
    Creates an ESQ object comprised of four solid cylinders extenind in z.

    Attributes
    ----------
    radius : float
        raidus of cylindrical ectrode.
    zc : float
        Center of electrode. The extent of the electrode is the half-total
        length in the positive and negative direction of zc.
    length : float
        Length of electrode.

    Methods
    -------
    pole(voltage, xcent, ycent)
        Creates the individual electrode using Warp's ZCylinder
    generate(voltage, xcent, ycent, data=False)
        Combines four poles to create esq object.
    """

    def __init__(self, radius=0.5 * mm, zc=2.2 * mm, length=0.695 * mm):

        self.radius = radius
        self.zc = zc
        self.length = length

    def pole(self, voltage, xcent, ycent):
        """Create individual electrode for ESQ

        Parameters
        ----------
        voltage : float
            Voltage of condctor.
        xcent : float
            Center of electrode in x
        ycent : float
            Center of electrode in y

        Returns
        -------
        conductor : Warp object
            The return is a cylinder extending in z with length "length"
            centered at zc extending to (zc - length/2, zc + length/2) with
            voltage "voltage."
        """

        conductor = wp.ZCylinder(
            voltage=voltage,
            xcent=xcent,
            ycent=ycent,
            zcent=self.zc,
            radius=self.radius,
            length=self.length,
        )
        return conductor

    def generate(self, voltage, xcent, ycent, data=False):
        """Combine four electrodes to form ESQ.

        Note that in the xy-plane the voltage for the top/bottom electrode is
        set to +.
        """
        # Create four poles
        top = self.pole(voltage=voltage, xcent=0, ycent=ycent)
        bottom = self.pole(voltage=voltage, xcent=0, ycent=-ycent)
        left = self.pole(voltage=-voltage, xcent=-xcent, ycent=0)
        right = self.pole(voltage=-voltage, xcent=xcent, ycent=0)

        # Combine poles into single ESQ
        conductor = top + bottom + left + right

        return conductor


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


if __name__ == "__main__":

    # ------------------------------------------------------------------------------
    #                    Logical Flags
    # Logical flags for controlling various routines within the script. All flags
    # prefixed with l_.
    # ------------------------------------------------------------------------------
    l_warpplots = True
    wp.setup()
    # ------------------------------------------------------------------------------
    #                     Create and load mesh and conductors
    # ------------------------------------------------------------------------------
    # Set paraemeeters for conductors
    separation = 0 * mm
    Nesq = 1

    zc = 0 * mm
    aperture = 0.55 * mm
    pole_rad = 0.415 * mm
    ESQ_length = 0.7 * mm
    xycent = aperture + pole_rad

    # Creat mesh using conductor geometries (above) to keep resolution consistent
    wp.w3d.xmmin = -1.5 * mm
    wp.w3d.xmmax = 1.5 * mm
    wp.w3d.nx = 150

    wp.w3d.ymmin = -1.5 * mm
    wp.w3d.ymmax = 1.5 * mm
    wp.w3d.ny = 150

    # Calculate nz to get about designed dz
    wp.w3d.zmmin = -1.5 * mm
    wp.w3d.zmmax = 1.5 * mm
    wp.w3d.nz = 400

    # Add boundary conditions
    wp.w3d.bound0 = wp.dirichlet
    wp.w3d.boundnz = wp.dirichlet
    wp.w3d.boundxy = wp.periodic
    wp.f3d.mgtol = 1e-6

    solver = wp.MRBlock3D()
    wp.registersolver(solver)

    ESQ = Mems_ESQ_SolidCyl(0.0, "1", 0.1 * kV, -0.1 * kV, chop=True)
    ESQ.set_geometry(rp=aperture, R=0.68 * aperture, lq=ESQ_length)
    wp.installconductor(ESQ.generate())
    wp.generate()

    z = wp.w3d.zmesh
    x, y = wp.w3d.xmesh, wp.w3d.ymesh
    xzero_ind = np.argmin(abs(x))
    dz = wp.w3d.dz
    Ex = wp.getselfe(comp="x")

    # Warp plotting for verification that mesh and conductors were created properly.
    warpplots = True
    if warpplots:
        wp.pfzx(fill=1, filled=1)
        wp.fma()
        wp.pfzy(fill=1, filled=1)
        wp.fma()

        # Plot xy where the surrounding wall starts
        val = ESQ.lq / 2.0 + 35 * um

        plate_cent_ind = np.argmin(abs(z + ESQ.lq / 2.0))
        zind = plate_cent_ind
        wp.pfxy(iz=zind, fill=1, filled=1)
        wp.fma()

        # plot the left-side wafer
        plate_cent_ind = np.argmin(abs(z - val))
        zind = plate_cent_ind
        wp.pfxy(iz=zind, fill=1, filled=1)
        wp.fma()

    dExdx = (Ex[xzero_ind + 1, xzero_ind, :] - Ex[xzero_ind, xzero_ind, :]) / wp.w3d.dx

    fig, ax = plt.subplots()
    ax.plot(z / mm, dExdx / kV * mm * mm)
    ax.axvline(
        x=-(ESQ.zc + ESQ.lq / 2.0 + 2 * ESQ.copper_zlen) / mm, c="k", ls="--", lw=1
    )
    ax.axvline(
        x=(ESQ.zc + ESQ.lq / 2.0 + 2 * ESQ.copper_zlen) / mm, c="k", ls="--", lw=1
    )
    ax.axhline(y=0, c="k", ls="--", lw=1)

    plt.show()
