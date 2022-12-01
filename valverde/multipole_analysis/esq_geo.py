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

wp.setup()

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


class mems_ESQ_SolidCyl:
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

    def __init__(self, zc, id, V_left, V_right, xc=0.0, yc=0.0):

        self.zc = zc
        self.id = id
        self.V_left = V_left
        self.V_right = V_right
        self.xc = 0.0
        self.yc = 0.0
        self.rp = None

        self.lq = None
        self.R = None

        # Surrounding square dimensions.
        self.copper_thickness = None
        self.cell_xysize = None

        # Prong length that connects rods to surrounding conductor
        self.prong_short_dim = 0.2 * mm
        self.prong_long_dim = 0.4 * mm

    def set_geometry(
        self,
        rp=0.55 * mm,
        R=0.8 * mm,
        lq=0.695 * mm,
        copper_thickness=35 * um,
        cell_xysize=3.0 * mm,
    ):
        self.rp = rp
        self.R = R
        self.lq = lq
        self.copper_thickness = copper_thickness
        self.cell_xysize = cell_xysize

    def create_outer_box(self):
        """Create surrounding square conductors.

        The ESQs are surrounding by a square conductor. Because of the opposing
        polarities the surrounding square takes one voltage on one side of the
        wafer and the opposing bias on the other side. This function creates
        the +/- polarity """

        l_zc = self.zc - self.lq / 2.0 - self.copper_thickness
        r_zc = self.zc + self.lq / 2.0 + self.copper_thickness

        size = self.cell_xysize

        l_box_out = wp.Box(
            xsize=size,
            ysize=size,
            zsize=self.copper_thickness,
            zcent=l_zc,
            xcent=self.xc,
            ycent=self.yc,
            voltage=self.V_left,
            condid=self.id,
        )
        l_box_in = wp.Box(
            xsize=size - 0.0002,
            ysize=size - 0.0002,
            zsize=self.copper_thickness,
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
            zsize=self.copper_thickness,
            voltage=self.V_left,
            zcent=l_zc,
            xcent=0.0,
            ycent=size / 2 - self.prong_long_dim / 2,
            condid=l_box_out.condid,
        )
        bot_prong = wp.Box(
            xsize=self.prong_short_dim,
            ysize=self.prong_long_dim,
            zsize=self.copper_thickness,
            voltage=self.V_left,
            zcent=l_zc,
            xcent=0.0,
            ycent=-(size / 2 - self.prong_long_dim / 2),
            condid=l_box_out.condid,
        )

        l_this_box = l_box_out - l_box_in
        l_box = l_this_box + top_prong + bot_prong

        r_box_out = wp.Box(
            xsize=size,
            ysize=size,
            zsize=self.copper_thickness,
            zcent=r_zc,
            xcent=self.xc,
            ycent=self.yc,
            voltage=self.V_right,
            condid=l_box_out.condid,
        )
        r_box_in = wp.Box(
            xsize=size - 0.0002,
            ysize=size - 0.0002,
            zsize=self.copper_thickness,
            zcent=r_zc,
            xcent=self.xc,
            ycent=self.yc,
            voltage=self.V_right,
            condid=l_box_out.condid,
        )

        r_prong = wp.Box(
            xsize=self.prong_long_dim,
            ysize=self.prong_short_dim,
            zsize=self.copper_thickness,
            voltage=self.V_right,
            zcent=r_zc,
            xcent=size / 2 - self.prong_long_dim / 2,
            ycent=0.0,
            condid=l_box_out.condid,
        )
        l_prong = wp.Box(
            xsize=self.prong_long_dim,
            ysize=self.prong_short_dim,
            zsize=self.copper_thickness,
            voltage=self.V_right,
            zcent=r_zc,
            xcent=-(size / 2 - self.prong_long_dim / 2),
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

        # Create top and bottom rods
        top = wp.ZCylinder(
            voltage=self.V_left,
            xcent=0.0,
            ycent=size / 2.0 - self.prong_long_dim - self.R / 2,
            zcent=self.zc,
            radius=self.R,
            length=self.lq + 4 * self.copper_thickness,
            condid=self.id,
        )
        bot = wp.ZCylinder(
            voltage=self.V_left,
            xcent=0.0,
            ycent=-(size / 2.0 - self.prong_long_dim - self.R / 2),
            zcent=self.zc,
            radius=self.R,
            length=self.lq + 4 * self.copper_thickness,
            condid=self.id,
        )

        # Create left and right rods
        left = wp.ZCylinder(
            voltage=self.V_right,
            xcent=-(size / 2.0 - self.prong_long_dim - self.R / 2),
            ycent=0.0,
            zcent=self.zc,
            radius=self.R,
            length=self.lq + 4 * self.copper_thickness,
            condid=self.id,
        )
        right = wp.ZCylinder(
            voltage=self.V_right,
            xcent=size / 2.0 - self.prong_long_dim - self.R / 2,
            ycent=0.0,
            zcent=self.zc,
            radius=self.R,
            length=self.lq + 4 * self.copper_thickness,
            condid=self.id,
        )

        conductor = top + bot + left + right

        return conductor

    def generate(self):
        """Combine four electrodes to form ESQ.

        Note that in the xy-plane the voltage for the top/bottom electrode is
        set to +.
        """
        # Create four poles
        square_conds = self.create_outer_box()
        rods = self.create_rods()

        conductor = square_conds + rods

        return conductor


# ------------------------------------------------------------------------------
#                    Logical Flags
# Logical flags for controlling various routines within the script. All flags
# prefixed with l_.
# ------------------------------------------------------------------------------
l_warpplots = True
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
wp.w3d.zmmin = -1.3 * mm
wp.w3d.zmmax = 1.3 * mm
design_dz = 20 * um
wp.w3d.nz = 400

# Add boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.periodic
wp.f3d.mgtol = 1e-8

solver = wp.MRBlock3D()
wp.registersolver(solver)


ESQ = mems_ESQ_SolidCyl(0.0, "1", 0.1 * kV, -0.1 * kV)
ESQ.set_geometry(rp=aperture, R=pole_rad, lq=ESQ_length)
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
    x=-(ESQ.zc + ESQ.lq / 2.0 + 2 * ESQ.copper_thickness) / mm, c="k", ls="--", lw=1
)
ax.axvline(
    x=(ESQ.zc + ESQ.lq / 2.0 + 2 * ESQ.copper_thickness) / mm, c="k", ls="--", lw=1
)

plt.show()
