# Script for various modeling and characterizing of ESQ conductors. As of right,
# the effective length and multipole moments are calculated using one of the
# conductor classes; either a solid or hollow cylindrical rod. The doublet system
# is loaded onto a mesh in isolation with no RF-acceleration gap field. However,
# the grounded conducting planes are placed on the mesh. This helps with viewing
# visualization and is somewhat illustrative of the real system.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy.integrate as integrate
import os
import math
import csv
import pdb
import sys

# Create argument parser for scaling. Must be done before importing Warp
import warpoptions

warpoptions.parser.add_argument("--scale_pole", default=False, type=float)
warpoptions.parser.add_argument("--scale_length", default=False, type=float)
inputs = warpoptions.parser.parse_args()
if inputs.scale_pole != False:
    scale_pole_rad = inputs.scale_pole
else:
    scale_pole_rad = 8 / 7
if inputs.scale_length != False:
    scale_Lesq = inputs.scale_length
else:
    scale_Lesq = 36.0

import warp as wp

# Save string for convenience
savepath = "/Users/nickvalverde/Desktop/ESQ_files/"

# Useful constants
kV = wp.kV
mm = wp.mm
um = 1e-6
print(f"--Using Pole Scale Factor of {scale_pole_rad}")
print(f"--Using ESQ Length Scale Factor of {scale_Lesq}")

# ------------------------------------------------------------------------------
#                     User Defined function
# Section createst the conductor classes for loading onto the mesh as well as
# some utility functions to be used.
# ------------------------------------------------------------------------------
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


class ESQ_ShellCyl:
    """Creates ESQ object comprised of thin-shell cylinders

    Attributes
    ----------
    radius : float
        outer radius of cylindrical ectrode.
    thickness : float
        thickness of shell i.e. rout - rin = thickness
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
        self, radius=0.75 * mm, thickness=0.1 * mm, zc=2.2 * mm, length=1.59 * mm
    ):

        self.radius = radius
        self.thickness = thickness
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

        # Create outer conductor
        outconductor = wp.ZCylinder(
            voltage=voltage,
            xcent=xcent,
            ycent=ycent,
            zcent=self.zc,
            radius=self.radius,
            length=self.length,
        )

        # Create inner conductor
        rin = self.radius - self.thickness
        inconductor = wp.ZCylinder(
            voltage=voltage,
            xcent=xcent,
            ycent=ycent,
            zcent=self.zc,
            radius=rin,
            length=self.length,
        )

        # Create final conductor
        conductor = outconductor - inconductor

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


class Wall:
    """Creates a solid cylinder with a hole bored through along z.

    Attributes
    ----------
    rextent : float
        Extent of conductor in r.
    zextent : float
        Length of conductor in z

    Methods
    -------
    generate(apperture, voltage, zcenter)
         Creates a solid cylinder with a hole.
    """

    def __init__(self, rextent=100 * wp.w3d.xmmax, zextent=0.1 * mm):
        self.rextent = rextent
        self.zextent = zextent

    def generate(self, apperture, voltage, zcenter):
        """Creates Warp conductor

        Parameters
        ----------
        apperture : float
            This will be inner radius of apperture (radius of hole).
        voltage : float
            Voltage of conductor.
        zcenter : float
            Where center of conductor is places.

        Returns
        -------
        condcutor : Warp conductor
            Returns a solid cylinder with a an apperture hole bored through.
        """
        wall = wp.ZCylinder(
            voltage=voltage, zcent=zcenter, length=self.zextent, radius=wp.w3d.xmmax
        )
        hole = wp.ZCylinder(
            voltage=voltage, zcent=zcenter, length=self.zextent, radius=apperture
        )
        # conductor = wp.ZAnnulus(
        #     rmin=apperture,
        #     voltage=voltage,
        #     zcent=zcenter,
        #     rmax=self.rextent,
        #     length=self.zextent,
        # )
        conductor = wall - hole

        return conductor


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
        diff = np.sqrt((mesh[index] ** 2 - value ** 2) ** 2)
        difference.append(diff)

    # Smallest element will be the index closest to value in indices
    i = np.argmin(difference)
    index = indices[i]

    return index


def efflength(gradient, dl):
    """Calculate effective quadrupole length

    Function calculates the effective length of a quadrupole by integrating the
    the gradient array using step size dl. G* is used by taking the max value
    of the gradient. The integral is then divided by this value giving an
    effective length.

    Parameters
    ----------
    gradient : ndarray
        1D-array of field gradient
    dl : flaot
        integration step size

    Returns
    -------
    ell : float
        calculated effective length
    """

    # Evaluate G*
    Gstar = max(gradient)
    # Evaluate integral of gradient
    integral = integrate.simps(gradient, dx=dl)

    # Evaluate effective length
    ell = integral / Gstar

    return ell


def interp2d_area(x_interp, y_interp, xmesh, ymesh, grid_data):
    """Interpolation routine that uses area weighting

    Routine will find the nearest grid points in xmesh and ymesh that corresponding
    to the points that are to be interpolated for x_interp and y_interp. The
    values at these points are given in grid_vals. The function will then return
    the interpolated values.

    Paramters
    ---------
    x_interp: ndarray
        Array of values to perform the interpolation at in x.

    y_interp: ndarray
        Array of values to perform the interpolation at in y.

    xmesh: ndarray
        The array holding the gridded x-values.

    ymesh: ndarray
        The array holding the gridded y-values.

    grid_data: ndarray
        This is the size (nx, ny) matrix holding the values for each (x,y) coordinate
        on the grid. In other words, this holds the value for some 2D function
        f(x,y) on the gridded values created by xmesh and ymesh.

    Returns
    -------
    interp_data: ndarray
        Array of equal size to x_interp and y_interp holding the interpolated
        data values for each coordinate pair in (x_interp, y_interp)
    """
    # Create a zero padding. If the interp value is exactly the grid value this
    # helps to treat the subtraction and approximately 0 and not numerical noise.
    numerical_padding = np.random.random() * 1e-12

    # Initialize geometrical values and interpolated array
    dx = xmesh[1] - xmesh[0]
    dy = ymesh[1] - ymesh[0]
    dA = dx * dy
    interp_data = np.zeros(len(x_interp))

    # Loop through interpolation points, find grid points, and interpolate.
    for i, (xm, ym) in enumerate(zip(x_interp, y_interp)):
        xm += numerical_padding
        ym += numerical_padding

        # Find grid points that enclose the xm-ym coordinates in use
        lowx = xm - dx
        highx = xm + dx

        lowy = ym - dy
        highy = ym + dy

        mask_x = (xmesh >= lowx) & (xmesh <= highx)
        mask_y = (ymesh >= lowy) & (xmesh <= highy)

        left, right = xmesh[mask_x][0], xmesh[mask_x][-1]
        bottom, top = ymesh[mask_y][0], ymesh[mask_y][-1]

        # Record indices for the gridpoints in use.
        x_indices = np.where(mask_x)[0]
        ix_left = x_indices[0]
        ix_right = x_indices[-1]

        y_indices = np.where(mask_y)[0]
        iy_bottom = y_indices[0]
        iy_top = y_indices[-1]

        # Calculate Areas and weight grid data
        A1 = (xm - left) * (ym - bottom)
        A2 = (right - xm) * (ym - bottom)
        A3 = (xm - left) * (top - ym)
        A4 = (right - xm) * (top - ym)

        q1m = grid_data[ix_left, iy_bottom] * A4 / dA
        q2m = grid_data[ix_right, iy_bottom] * A3 / dA
        q3m = grid_data[ix_left, iy_top] * A2 / dA
        q4m = grid_data[ix_right, iy_top] * A1 / dA

        qm = q1m + q2m + q3m + q4m
        interp_data[i] = qm

    return interp_data


# ------------------------------------------------------------------------------
#                     Create and load mesh and conductors
# ------------------------------------------------------------------------------
# Set paraemeeters for conductors
voltage = 0.3 * kV
separation = 0 * mm
Nesq = 1

zc = 0 * mm
wallvoltage = 0 * kV
aperture = 0.55 * mm
pole_rad = aperture * scale_pole_rad
ESQ_length = aperture * scale_Lesq
xycent = aperture + pole_rad
walllength = 0.1 * mm
wallzcent = ESQ_length / 2 + 1.0 * mm + walllength / 2
ddd
# Creat mesh using conductor geometries (above) to keep resolution consistent
wp.w3d.xmmin = -xycent - pole_rad * (1 + 0.10)
wp.w3d.xmmax = xycent + pole_rad * (1 + 0.10)
wp.w3d.nx = 300

wp.w3d.ymmin = -xycent - pole_rad * (1 + 0.10)
wp.w3d.ymmax = xycent + pole_rad * (1 + 0.10)
wp.w3d.ny = 300

# Calculate nz to get about designed dz
wp.w3d.zmmin = -ESQ_length
wp.w3d.zmmax = ESQ_length
design_dz = 5 * um
calc_nz = (wp.w3d.zmmax - wp.w3d.zmmin) / design_dz
wp.w3d.nz = 650
print(int(calc_nz))

# Add boundary conditions
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.neumann
wp.f3d.mgtol = 1e-8

wp.w3d.l4symtry = False
solver = wp.MRBlock3D()
wp.registersolver(solver)

# Create left and right quads
leftconductor = ESQ_SolidCyl(zc=-zc, radius=pole_rad, length=ESQ_length)
leftquad = leftconductor.generate(
    voltage=voltage, xcent=xycent, ycent=xycent, data=False
)
# rightconductor = ESQ_SolidCyl(zc=zc, radius=pole_rad, length=ESQ_length)
# rightquad = rightconductor.generate(
#     voltage=-voltage, xcent=xycent, ycent=xycent, data=False
# )

leftwall = Wall().generate(apperture=aperture, voltage=wallvoltage, zcenter=-wallzcent)
rightwall = Wall().generate(apperture=aperture, voltage=wallvoltage, zcenter=wallzcent)

# Install Conductors and generate mesh
wp.installconductor(leftquad)
# wp.installconductor(rightquad)
# wp.installconductor(leftwall)
# wp.installconductor(rightwall)

wp.generate()

# ------------------------------------------------------------------------------
#                     Calculate effective length
# ------------------------------------------------------------------------------
# Rename meshes and find indicesfor the mesh z-center and z-center of right quad
x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh
zzeroindex = getindex(z, 0.0, wp.w3d.dz)
zcenterindex = getindex(z, zc, wp.w3d.dz)
xzeroindex = getindex(x, 0.0, wp.w3d.dx)
yzeroindex = getindex(y, 0.0, wp.w3d.dy)

# Create Warp plots. Useful for quick-checking
warpplots = False
if warpplots:
    wp.setup()
    leftquad.drawzx(filled=True)
    rightwall.drawzx(filled=True)
    leftwall.drawzx(filled=True)
    wp.fma()

    leftquad.drawxy(filled=True)
    wp.fma()

    wp.pfxy(iz=zcenterindex, fill=1, filled=1)
    wp.fma()

    wp.pfzx(fill=1, filled=1)
    wp.fma()

    wp.pfxy(
        plotselfe=1, plotphi=0, comp="x", fill=1, filled=1, contours=50, iz=zcenterindex
    )
    wp.fma()

    wp.pfxy(
        plotselfe=1, plotphi=0, comp="y", fill=1, filled=1, contours=50, iz=zcenterindex
    )
    wp.fma()

# Grab Fields
phi = wp.getphi()
phixy = wp.getphi()[:, :, zcenterindex]
Ex = wp.getselfe(comp="x")
Ey = wp.getselfe(comp="y")
gradex = Ex[xzeroindex + 1, yzeroindex, :] / wp.w3d.dx

make_effective_length_plots = False
if make_effective_length_plots:
    # Create plot of Ex gradient
    fig, ax = plt.subplots()
    ax.set_xlabel("z [mm]")
    ax.set_ylabel(r"$E_x(dx, 0, z)$/dx [kV mm$^{-2}$]")
    ax.set_title(r"$E_x$ Gradient One Grid-cell Off-axis vs z")
    ax.scatter(z / mm, gradex / kV / 1e6, s=1.2)
    ax.axhline(y=0, c="k", lw=0.5)
    ax.axvline(x=0, c="k", lw=0.5)

    # add ESQ markers to plot
    esq1left = -zc - ESQ_length / 2
    esq1right = -zc + ESQ_length / 2
    esq2left = zc - ESQ_length / 2
    esq2right = zc + ESQ_length / 2
    ax.axvline(x=esq1left / mm, c="b", lw=0.8, ls="--", label="First ESQ")
    ax.axvline(x=esq1right / mm, c="b", lw=0.8, ls="--")
    ax.axvline(x=esq2left / mm, c="r", lw=0.8, ls="--", label="Second ESQ")
    ax.axvline(x=esq2right / mm, c="r", lw=0.8, ls="--")
    ax.axvline(
        x=(wallzcent - walllength / 2) / mm, c="grey", lw=0.8, ls="--", label="Wall"
    )
    ax.axvline(x=-(wallzcent - walllength / 2) / mm, c="grey", lw=0.8, ls="--")
    ax.axvline(x=(wallzcent + walllength / 2) / mm, c="grey", lw=0.8, ls="--")
    ax.axvline(x=-(wallzcent + walllength / 2) / mm, c="grey", lw=0.8, ls="--")
    plt.legend()
    plt.savefig(savepath + "full-mesh.pdf", dpi=400)
    plt.show()

# Plot and calculate effective length
# Integrate over right esq. Note, this gradient is negative.
dEdx = abs(gradex[:])
ell = efflength(dEdx, wp.w3d.dz)
print("Effective Length = ", ell / mm)

if make_effective_length_plots:
    fig, ax = plt.subplots()
    ax.set_title(
        f"Integrand For Effective Length {ell/mm:.4f} mm, zc = {zc/mm :.4f} mm, n = {Nesq}, Lq = {ESQ_length/mm:.4f} mm",
        fontsize="small",
    )
    ax.set_ylabel(r"$|E(x=dx,y=0,z)$/dx| [kV mm$^{-2}$]")
    ax.set_xlabel("z [mm]")
    ax.scatter(z / mm, dEdx / kV / 1000 / 1000, s=0.5)
    # Annotate
    ax.axhline(y=0, lw=0.5, c="k")
    ax.axvline(x=esq2left / mm, c="r", lw=0.8, ls="--", label="ESQ Edges")
    ax.axvline(x=esq2right / mm, c="r", lw=0.8, ls="--")
    ax.axvline(
        x=(wallzcent - walllength / 2) / mm, c="grey", lw=0.8, ls="--", label="Wall"
    )
    ax.axvline(x=(wallzcent + walllength / 2) / mm, c="grey", lw=0.8, ls="--")
    ax.legend()
    plt.savefig(savepath + "integrand.pdf", dpi=400)
    plt.show()

# ------------------------------------------------------------------------------
#                          Multipole Analysis
# This section will do the multipole analysis.
# The x and y component of the electric field (Ex and Ey) are give on the full
# 3D mesh. The analytic treatment of the multipole is given on the x-y plane
# and is usualy seen as a function of r and theta E(r, theta). The 3D grid is
# not a problem here since the analysis can be done for each plane at each grid
# point of z. However, this is computationally expensive, and instead the field
# compoenents are marginalized in z by integrating over the effective length of
# one quad and dividing by this effective length.
# ------------------------------------------------------------------------------
# Find fields in the region from -ell/2 to ell/2
eff_index_left = getindex(z, 0 * mm, wp.w3d.dz)
eff_index_right = getindex(z, z.max(), wp.w3d.dz)
Ex_comp = Ex.copy()[:, :, :]
Ey_comp = Ey.copy()[:, :, :]
nx, ny, nz = Ex_comp.shape

# Reshape the fields to nx*ny by nz. This will give a column of vectors, where
# each vector is the field along z at a given x,y coordinate.
Ex_comp = Ex_comp.reshape(int(nx * ny), nz)
Ey_comp = Ey_comp.reshape(int(nx * ny), nz)
np.save("Ex_comp", Ex_comp)
np.save("Ey_comp", Ey_comp)

integrated_Ex = integrate.simpson(Ex_comp, dx=wp.w3d.dz) / ell
integrated_Ey = integrate.simpson(Ey_comp, dx=wp.w3d.dz) / ell

make_transField_plots = False
if make_transField_plots:
    fig, ax = plt.subplots()
    ax.set_title(r"$E_x(x,y,z=zcent)$")
    X, Y = np.meshgrid(x, y, indexing="ij")
    contourx = ax.contourf(
        X / mm, Y / mm, Ex[:, :, zcenterindex], levels=50, cmap="viridis"
    )
    ax.contour(
        X / mm,
        Y / mm,
        Ex[:, :, zcenterindex],
        levels=50,
        linewidths=0.1,
        linestyles="solid",
        colors="k",
    )
    fig.colorbar(contourx, ax=ax)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    plt.savefig("/Users/nickvalverde/Desktop/Ex_original.pdf", dpi=400)
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title(r" $E_y(x,y,z=zcent)$")
    contourx = ax.contourf(
        X / mm, Y / mm, Ey[:, :, zcenterindex], levels=50, cmap="viridis"
    )
    ax.contour(
        X / mm,
        Y / mm,
        Ey[:, :, zcenterindex],
        levels=50,
        linewidths=0.1,
        linestyles="solid",
        colors="k",
    )
    fig.colorbar(contourx, ax=ax)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    plt.savefig("/Users/nickvalverde/Desktop/Ey_original.pdf", dpi=400)
    plt.show()

    # fig, ax = plt.subplots()
    # ax.set_title(r"Integrated $E_x(x,y)$")
    # contourx = ax.contourf(
    #     X / mm, Y / mm, integrated_Ex.reshape(nx, ny), levels=50, cmap="viridis"
    # )
    # ax.contour(
    #     X / mm,
    #     Y / mm,
    #     integrated_Ex.reshape(nx, ny),
    #     levels=50,
    #     linewidths=0.1,
    #     linestyles="solid",
    #     colors="k",
    # )
    # fig.colorbar(contourx, ax=ax)
    # ax.set_xlabel("x [mm]")
    # ax.set_ylabel("y [mm]")
    # plt.savefig("/Users/nickvalverde/Desktop/x_transfields.pdf", dpi=400)
    # plt.show()
    #
    # fig, ax = plt.subplots()
    # ax.set_title(r"Integrated $E_y(x,y)$")
    # contoury = ax.contourf(
    #     X / mm, Y / mm, integrated_Ey.reshape(nx, ny), levels=50, cmap="viridis"
    # )
    # ax.contour(
    #     X / mm,
    #     Y / mm,
    #     integrated_Ey.reshape(nx, ny),
    #     levels=50,
    #     linewidths=0.1,
    #     linestyles="solid",
    #     colors="k",
    # )
    # fig.colorbar(contoury, ax=ax)
    # ax.set_xlabel("x [mm]")
    # ax.set_ylabel("y [mm]")
    # plt.savefig("/Users/nickvalverde/Desktop/y_transfields.pdf", dpi=400)
    # plt.show()


# ------------------------------------------------------------------------------
#                     Testing area for interpolation
# Exact geometrical fields can be specified and used to test calculated values.
# Dipole field to be fixed due to division by zero error.
# ------------------------------------------------------------------------------
Exfun = lambda x, y: pow(x, 3) - 3 * x * pow(y, 2)
Eyfun = lambda x, y: -3 * pow(x, 2) * y + pow(y, 3)
xtest = np.linspace(-0.8, 0.8, 500) * mm
ytest = np.linspace(-0.8, 0.8, 500) * mm
X, Y = np.meshgrid(xtest, ytest)
Extest = Exfun(X, Y)
Eytest = Eyfun(X, Y)
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------

# Set up paramters for interpolation
interp_R = aperture - 2 * wp.w3d.dx
interp_np = math.ceil(np.sqrt(2) * np.pi * wp.w3d.nx * interp_R / aperture)
print(f"Np = {interp_np}")
interp_theta = np.linspace(0, 2 * np.pi, interp_np)
interp_x = interp_R * np.cos(interp_theta)
interp_y = interp_R * np.sin(interp_theta)

interp_Ex = np.zeros(interp_np)
interp_Ey = np.zeros(interp_np)

# Perform interpolation using Warp's getgrid. The algorithm is written in
# Fortran and so the indices array-lengths for the grid data need to be deducted
# one unit for proper indexing in Fortran. The interpolated arrays for the
# field data (interp_Ex, interp_Ey) are changed in place
wp.top.getgrid2d(
    len(interp_x),
    interp_x,
    interp_y,
    interp_Ex,
    len(x) - 1,
    len(y) - 1,
    integrated_Ex.reshape(nx, ny),
    x.min(),
    x.max(),
    y.min(),
    y.max(),
)

wp.top.getgrid2d(
    len(interp_x),
    interp_x,
    interp_y,
    interp_Ey,
    len(x) - 1,
    len(y) - 1,
    integrated_Ey.reshape(nx, ny),
    x.min(),
    x.max(),
    y.min(),
    y.max(),
)
# Uncomment this portion to run the test cases
# wp.top.getgrid2d(
#     len(interp_x),
#     interp_x,
#     interp_y,
#     interp_Ex,
#     len(xtest) - 1,
#     len(ytest) - 1,
#     Extest,
#     xtest.min(),
#     xtest.max(),
#     ytest.min(),
#     ytest.max(),
# )
# wp.top.getgrid2d(
#     len(interp_x),
#     interp_x,
#     interp_y,
#     interp_Ex,
#     len(xtest) - 1,
#     len(ytest) - 1,
#     Eytest,
#     xtest.min(),
#     xtest.max(),
#     ytest.min(),
#     ytest.max(),
# )

# ------------------------------------------------------------------------------
#                    Calculate multipole coefficients
# ------------------------------------------------------------------------------
# Evaluate the coefficients a_n and b_n for Ex and Ey.

make_3d_integrand_plot = False
if make_3d_integrand_plot:
    # Make contour polot of integrated z values for Ex
    theta3d = np.linspace(0, 2 * np.pi, int(2 * 4))
    # x3d = np.zeros(theta3d)
    # y3d = np.zeros(theta3d)
    dtheta = interp_theta[1] - interp_theta[0]
    # for i,angle in enumerate(theta3d):
    #     # find value in x and y
    #     index = getindex(angle, interp_theta, dtheta)
    #     x3d[i] = interp_x[index]
    #

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection="3d")
    ax.set_title(r"Value of Integration $\int E_x(x,y,z)dz$", fontsize="small")
    ax.plot3D(interp_x, interp_y, np.zeros(len(interp_x)), "gray")
    ax.scatter3D(interp_x, interp_y, interp_Ex, c=interp_Ex, cmap="Greens")
    # for (xi,yi,Ei) in zip(interp_x, interp_y, interp_Ex):
    #     ax.plot([xi,xi], [yi,yi], [0,Ei], 'k--')
    ax.set_xlabel(r"$x = R\cos(\theta)$ [mm]")
    ax.set_ylabel(r"$y = R\sin(\theta)$ [mm]")
    ax.set_zlabel(r"$\bar{E}_x(r, \theta)$ [V/m]")
    plt.tight_layout()
    plt.savefig(savepath + "z_integration_visual.pdf", dpi=400)
    plt.show()

n_order = 14
nterms = np.array([i for i in range(1, n_order + 1)])
dtheta = interp_theta[1] - interp_theta[0]

Excoeff_array = np.zeros((2, len(nterms)))
Eycoeff_array = np.zeros((2, len(nterms)))
R = interp_R / aperture

for i in range(1, len(nterms)):
    n = nterms[i]

    # Treat n=0 coefficient separately since coeff is different
    if n == 1:
        coeff = 1 / 2 / np.pi
        Ax_integrand = interp_Ex
        Bx_integrand = 0
        Ay_integrand = 0
        By_integrand = interp_Ey

        Ax = coeff * integrate.simpson(Ax_integrand, dx=dtheta)
        Bx = 0
        Ay = 0
        By = coeff * integrate.simpson(By_integrand, dx=dtheta)

        Excoeff_array[:, i] = Ax, Bx
        Eycoeff_array[:, i] = Ay, By

    coeff = pow(1 / R, n - 1) / np.pi
    Ax_integrand = interp_Ex * np.cos((n - 1) * interp_theta)
    Bx_integrand = interp_Ex * np.sin((n - 1) * interp_theta)
    Ay_integrand = -1.0 * interp_Ey * np.sin((n - 1) * interp_theta)
    By_integrand = interp_Ey * np.cos((n - 1) * interp_theta)

    Ax = coeff * integrate.simpson(Ax_integrand, dx=dtheta)
    Bx = coeff * integrate.simpson(Bx_integrand, dx=dtheta)
    Ay = coeff * integrate.simpson(Ay_integrand, dx=dtheta)
    By = coeff * integrate.simpson(By_integrand, dx=dtheta)

    Excoeff_array[:, i] = Ax, Bx
    Eycoeff_array[:, i] = Ay, By

# ------------------------------------------------------------------------------
#                           Make plots of coefficient data
# Visualize coefficient magnitudes with a 3D bar plot
# The n-terms will be plotted using 3D bars, where there extent in z is their
# contribution to the sum. To different coefficients are put on the same plot
# with a separation given by the y3 argument. This controls where they are put
# on the xy-plane. The dimensions of the box are given by dx,dy and dz. The
# dimensions dx and dy are decided best onf aesthtic where dz will be give the
# height or contribution to the sum.
# ------------------------------------------------------------------------------
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(211, projection="3d")

y3 = np.ones(len(nterms))
z3 = np.zeros(len(nterms))

# Set width of bars. These settings are for plot aesthetics and not significant
xbar_width = np.ones(len(nterms)) / 4
ybar_width = np.ones(len(nterms)) / 2

# Take squared-coefficients from Ex
An = Excoeff_array[0, :].copy()
Bn = Excoeff_array[1, :].copy()

# Use maximum multipole value for normalization
norm = np.max(np.sqrt(pow(An, 2) + pow(Bn, 2)))
nmax_index = nterms[np.argmax(np.sqrt(pow(An, 2) + pow(Bn, 2)))]
An_norm = np.max(abs(An))
Bn_norm = np.max(abs(Bn))

# Store data in a dataframe and append to csv file. If csv file already exists
# the column headers are ignored. If not, the file is created with headers.
filename = "multipole_data.csv"
file_exists = filename in os.listdir(savepath)
df = pd.DataFrame()
df["init"] = [np.nan]
df["n-max"] = nmax_index
df["R_pole/R_aper"] = scale_pole_rad
df["L_esq/R_aper"] = scale_Lesq
df["separation[mm]"] = separation
df["n-interp"] = interp_np
for i in range(len(nterms)):
    # Loop through n-poles and create column header
    df[f"Norm A{i+1}"] = An[i] / An_norm
    df[f"Norm B{i+1}"] = Bn[i] / An_norm
for i in range(len(nterms)):
    df[f"A{i+1}"] = An[i]
    df[f"B{i+1}"] = Bn[i]
df["dx[mm]"] = wp.w3d.dx / mm
df["dy[mm]"] = wp.w3d.dy / mm
df["dz[mm]"] = wp.w3d.dz / mm
df["mesh_zext[mm]"] = (wp.w3d.zmmax - wp.w3d.zmmin) / mm
df.drop("init", axis=1, inplace=True)

with open(os.path.join(savepath, filename), "a") as f:
    df.to_csv(f, header=not (file_exists), index=False)

# Print out numerical information for coefficients
print(f"--Scale Fraction {scale_pole_rad}")
print(f"--Max order n = {nterms[nmax_index]}:")
print("--Normalized-squared coefficients (A,B)")
print(f"### Coeff. Values Squared Normalized by Maximum Coeff. ###")

for i, n in enumerate(nterms):
    print(f"--n={n}")
    print(f"(An, Bn): ({An[i]:.5E},  {Bn[i]:.5E})")
    print(
        f"Noramalized by max n-term (An, Bn): ({An[i]/An_norm:.5E}, {Bn[i]/Bn_norm:.5E})"
    )
    print("")

do_multple_barplots = False
if do_multple_barplots:
    # Plot An, Bn and An+Bn on bar plot where height represents fraction of Max pole
    ax.bar3d(nterms, 1 * y3, z3, xbar_width, ybar_width, An / norm, color="b")
    ax.bar3d(nterms, 3 * y3, z3, xbar_width, ybar_width, Bn / norm, color="g")
    ax.bar3d(nterms, 6 * y3, z3, xbar_width, ybar_width, (An + Bn) / norm, color="k")

    ax.set_title(
        fr"Normalized Squared-Multipole Coefficients for $E(x,y)$", fontsize="x-small",
    )
    ax.set_xlabel("n", fontsize="small")
    ax.set_ylabel("")
    ax.set_zlabel(r"Fraction of $\max[A_n^2 + B_n^2]$", fontsize="small")
    ax.set_yticks([])

    # Create legend labels using a proxy. Needed for 3D bargraph
    blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="b")
    green_proxy = plt.Rectangle((0, 0), 1, 1, fc="g")
    black_proxy = plt.Rectangle((0, 0), 1, 1, fc="k")
    ax.legend(
        [blue_proxy, green_proxy, black_proxy],
        [r"$A_n^2$", r"$B_n^2$", r"$A_n^2 + B_n^2$"],
        fontsize="x-small",
    )
    plt.tight_layout()
    plt.savefig(savepath + "multipole_coeffs.pdf", dpi=400)
    plt.show()

    # Make plot taking out maximum contribution for 'zoomed in' look
    maskAn = An < An_norm
    maskBn = Bn < Bn_norm
    mask_sum = (An + Bn) < (An_norm + Bn_norm)
    An_masked = An[maskAn]
    Bn_masked = Bn[maskBn]
    sum_masked = (An + Bn)[mask_sum]
    n_maskedA = nterms[maskAn]
    n_maskedB = nterms[maskBn]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(211, projection="3d")

    y3 = np.ones(len(An_masked))
    z3 = np.zeros(len(An_masked))

    # Set width of bars. These settings are for plot aesthetics and not significant
    xbar_width = np.ones(len(n_maskedA)) / 4
    ybar_width = np.ones(len(n_maskedA)) / 2

    # Plot An, Bn and An+Bn on bar plot where height represents fraction of Max pole
    ax.bar3d(n_maskedA, 1 * y3, z3, xbar_width, ybar_width, An_masked / norm, color="b")
    ax.bar3d(n_maskedB, 3 * y3, z3, xbar_width, ybar_width, Bn_masked / norm, color="g")
    ax.bar3d(
        nterms[mask_sum],
        6 * y3,
        z3,
        xbar_width,
        ybar_width,
        sum_masked / norm,
        color="k",
    )

    ax.set_title(
        fr"Normalized Squared-Multipole Coefficients (Dominant Term Removed)",
        fontsize="x-small",
    )
    ax.set_xlabel("n", fontsize="small")
    ax.set_ylabel("")
    ax.set_zlabel(r"Fraction of $\max[A_n^2 + B_n^2]$", fontsize="small")
    ax.set_yticks([])

    # Create legend labels using a proxy. Needed for 3D bargraph
    blue_proxy = plt.Rectangle((0, 0), 1, 1, fc="b")
    green_proxy = plt.Rectangle((0, 0), 1, 1, fc="g")
    black_proxy = plt.Rectangle((0, 0), 1, 1, fc="k")
    ax.legend(
        [blue_proxy, green_proxy],
        [r"$A_n^2$", r"$B_n^2$", r"$A_n^2 + B_n^2$"],
        fontsize="x-small",
    )
    plt.tight_layout()
    plt.savefig(savepath + "zoomed_multipole_coeffs.pdf", dpi=400)
    plt.show()
