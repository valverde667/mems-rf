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

    def __init__(
        self,
        zc,
        id,
        V_left,
        V_right,
        xc=0.0,
        yc=0.0,
        lq=0.695 * mm,
        R=0.8 * mm,
        rp=0.55 * mm,
    ):

        self.zc = zc
        self.id = id
        self.V_left = V_left
        self.V_right = V_right
        self.xc = 0.0
        self.yc = 0.0
        self.lq = lq
        self.R = R
        self.rp = rp

        # Surrounding square dimensions.
        self.copper_thickness = None
        self.cell_width = None

    def create_outer_box(self, cell_width=1 * mm, copper_thickness=35 * um):
        """Create surrounding square conductors.

        The ESQs are surrounding by a square conductor. Because of the opposing
        polarities the surrounding square takes one voltage on one side of the
        wafer and the opposing bias on the other side. This function creates
        the +/- polarity """

        self.cell_width = cell_width
        self.copper_thickness = copper_thickness
        l_zc = self.zc - self.lq / 2.0 + copper_thickness
        r_zc = self.zc + self.lq / 2.0 - copper_thickness

        l_box_out = wp.Box(
            xsize=cell_width,
            ysize=cell_width,
            zsize=copper_thickness,
            zcent=l_zc,
            xcent=self.xc,
            ycent=self.yc,
            voltage=self.V_left,
            condid=self.id,
        )
        l_box_in = wp.Box(
            xsize=cell_width - 0.0002,
            ysize=cell_width - 0.0002,
            zsize=copper_thickness,
            zcent=l_zc,
            xcent=self.xc,
            ycent=self.yc,
            voltage=self.V_left,
            condid=l_box_out.condid,
        )
        l_box = l_box_out - l_box_in

        r_box_out = wp.Box(
            xsize=cell_width,
            ysize=cell_width,
            zsize=copper_thickness,
            zcent=r_zc,
            xcent=self.xc,
            ycent=self.yc,
            voltage=self.V_right,
            condid=l_box_out.condid,
        )
        r_box_in = wp.Box(
            xsize=cell_width - 0.0002,
            ysize=cell_width - 0.0002,
            zsize=copper_thickness,
            zcent=r_zc,
            xcent=self.xc,
            ycent=self.yc,
            voltage=self.V_right,
            condid=l_box_out.condid,
        )
        r_box = r_box_out - r_box_in

        box = l_box + r_box

        return box

    def generate(self, data=False):
        """Combine four electrodes to form ESQ.

        Note that in the xy-plane the voltage for the top/bottom electrode is
        set to +.
        """
        # Create four poles
        square_conds = self.create_outer_box()

        conductor = square_conds

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
rod_fraction = 0.5

zc = 0 * mm
wallvoltage = 0 * kV
aperture = 0.55 * mm
pole_rad = aperture * 1.5
ESQ_length = 0.695 * mm
xycent = aperture + pole_rad
walllength = 0.1 * mm
wallzcent = ESQ_length + 1.0 * mm + walllength / 2

# Creat mesh using conductor geometries (above) to keep resolution consistent
wp.w3d.xmmin = -aperture - pole_rad
wp.w3d.xmmax = aperture + pole_rad
design_dx = 5 * um
calc_nx = (wp.w3d.xmmax - wp.w3d.xmmin) / design_dx
wp.w3d.nx = 200

wp.w3d.ymmin = -aperture - pole_rad
wp.w3d.ymmax = aperture + pole_rad
wp.w3d.ny = wp.w3d.nx

# Calculate nz to get about designed dz
wp.w3d.zmmin = -0.4 * mm
wp.w3d.zmmax = 0.4 * mm
wp.w3d.nz = 200

# Add boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.periodic
wp.f3d.mgtol = 1e-8

solver = wp.MRBlock3D()
wp.registersolver(solver)


ESQ = mems_ESQ_SolidCyl(0.0, "1", 0.5 * kV, -0.5 * kV)
wp.installconductor(ESQ.generate())
wp.generate()

z = wp.w3d.zmesh
dz = wp.w3d.dz

# Warp plotting for verification that mesh and conductors were created properly.
warpplots = True
if warpplots:
    wp.pfzx(fill=1, filled=1)
    wp.fma()
    wp.pfzy(fill=1, filled=1)
    wp.fma()

    # plot the left-side wafer
    xy_cent = ESQ.zc - ESQ.lq / 2.0
    plate_cent_ind = np.argmin(abs(z - ESQ.lq / 2.0))
    zind = plate_cent_ind
    wp.pfxy(iz=zind, fill=1, filled=1)
    wp.fma()
