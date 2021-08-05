"""Model ESQ doublet in Warp and pull information. Plotting the ESQ gradient
along with field strength is done. Calculating effective length of quadrupole.
Also, different quadrupole designs are used: solid cylinder rods (ESQ), hollow
cylindrical shell rods (ESQGrant), and Timo's design. """

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import pdb

# import conductors as cond
import sys
import warp as wp

# Save string for convenience
savepath = "/Users/nickvalverde/Desktop/ESQ_files/"

# Useful constants
kV = wp.kV
mm = wp.mm
um = 1e-6

# Create mesh
wp.w3d.xmmin = -0.8 * mm
wp.w3d.xmmax = 0.8 * mm
wp.w3d.nx = 200

wp.w3d.ymmin = -0.8 * mm
wp.w3d.ymmax = 0.8 * mm
wp.w3d.ny = 202

wp.w3d.zmmin = -4 * mm
wp.w3d.zmmax = 4 * mm
wp.w3d.nz = 350

# # Timo-Create mesh
# wp.w3d.xmmin = -1.5 * mm
# wp.w3d.xmmax = 1.5 * mm
# wp.w3d.nx = 150
#
# wp.w3d.ymmin = -1.5 * mm
# wp.w3d.ymmax = 1.5 * mm
# wp.w3d.ny = 150
#
# wp.w3d.zmmin = -4 * mm
# wp.w3d.zmmax = 4 * mm
# wp.w3d.nz = 700

# Add boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.periodic

wp.w3d.l4symtry = False
solver = wp.MRBlock3D()
wp.registersolver(solver)


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


# Set paraemeeters for conductors
voltage = 0.3 * kV
separation = 1.8 * mm
length_multiplier = 1
length = length_multiplier * 0.695 * mm

# length = cond.wafer_thickness + 2 * cond.copper_thickness #Timo esq
zc = separation / 2 + length / 2
wallvoltage = 0 * kV
aperture = 0.55 * mm
pole_rad = 0.50 * mm
xycent = aperture + pole_rad
walllength = 0.1 * mm
wallzcent = separation / 2 + length + separation + walllength / 2

# Create left and right quads
leftconductor = ESQ_SolidCyl(zc=-zc, length=length)
leftquad = leftconductor.generate(
    voltage=voltage, xcent=xycent, ycent=xycent, data=True
)
rightconductor = ESQ_SolidCyl(zc=zc, length=length)
rightquad = rightconductor.generate(
    voltage=-voltage, xcent=xycent, ycent=xycent, data=True
)

# # For Timo Conductors
# leftquad = cond.ESQ(position=-zc, invertPolarity=-1, voltage=voltage)
# rightquad = cond.ESQ(position=zc, invertPolarity=1, voltage=voltage)
# Create left and right grounded walls
leftwall = Wall().generate(apperture=aperture, voltage=wallvoltage, zcenter=-wallzcent)
rightwall = Wall().generate(apperture=aperture, voltage=wallvoltage, zcenter=wallzcent)

# Install Conductors and generate mesh
wp.installconductor(leftquad)
wp.installconductor(rightquad)
wp.installconductor(leftwall)
wp.installconductor(rightwall)

# Multipole settings
wp.w3d.nmom = 10
wp.w3d.nzmom = wp.w3d.nz

wp.generate()


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


# Rename meshes and find indices
x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh
zzeroindex = getindex(z, 0.0, wp.w3d.dz)
zcenterindex = getindex(z, zc, wp.w3d.dz)

# Create Warp plots. Useful for quick-checking
warpplots = True
if warpplots:
    wp.setup()
    leftquad.drawzx(filled=True)
    rightquad.drawzx(filled=True)
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
gradex = Ex[1, 0, :] / wp.w3d.dx

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
    esq1left = -zc - length / 2
    esq1right = -zc + length / 2
    esq2left = zc - length / 2
    esq2right = zc + length / 2
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
dEdx = abs(gradex[zzeroindex:])
ell = efflength(dEdx, wp.w3d.dz)
print("Effective Length = ", ell / mm)

if make_effective_length_plots:
    fig, ax = plt.subplots()
    ax.set_title(
        f"Integrand For Effective Length {ell/mm:.4f} mm, zc = {zc/mm :.4f} mm, n = {length_multiplier}, Lq = {length/mm:.4f} mm",
        fontsize="small",
    )
    ax.set_ylabel(r"$|E(x=dx,y=0,z)$/dx| [kV mm$^{-2}$]")
    ax.set_xlabel("z [mm]")
    ax.scatter(z[zzeroindex:] / mm, dEdx / kV / 1000 / 1000, s=0.5)
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
eff_index_left = getindex(z, zc - ell / 2, wp.w3d.dz)
eff_index_right = getindex(z, zc + ell / 2, wp.w3d.dz)
Ex_comp = Ex.copy()[:, :, eff_index_left : eff_index_right + 1]
Ey_comp = Ey.copy()[:, :, eff_index_left : eff_index_right + 1]
nx, ny, nz = Ex_comp.shape

# Find the index corresponding to the effective length of the quadrupole from
# above
eff_index_left = getindex(z, zc - ell / 2, wp.w3d.dz)
eff_index_right = getindex(z, zc + ell / 2, wp.w3d.dz)

# Reshape the fields to nx*ny by nz. This will give a column of vectors, where
# each vector is the field along z at a given x,y coordinate.
Ex_comp = Ex_comp.reshape(int(nx * ny), nz)
Ey_comp = Ey_comp.reshape(int(nx * ny), nz)

integrated_Ex = integrate.simpson(Ex_comp, dx=wp.w3d.dz) / ell
integrated_Ey = integrate.simpson(Ey_comp, dx=wp.w3d.dz) / ell

make_transField_plots = True
if make_transField_plots:
    fig, ax = plt.subplots()
    ax.set_title(r"$E_x(x,y,z=zc)$")
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
    ax.set_title(r" $E_x(x,y,z=zc)$")
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

    fig, ax = plt.subplots()
    ax.set_title(r"Integrated $E_x(x,y)$")
    contourx = ax.contourf(
        X / mm, Y / mm, integrated_Ex.reshape(nx, ny), levels=50, cmap="viridis"
    )
    ax.contour(
        X / mm,
        Y / mm,
        integrated_Ex.reshape(nx, ny),
        levels=50,
        linewidths=0.1,
        linestyles="solid",
        colors="k",
    )
    fig.colorbar(contourx, ax=ax)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    plt.savefig("/Users/nickvalverde/Desktop/x_transfields.pdf", dpi=400)
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title(r"Integrated $E_y(x,y)$")
    contoury = ax.contourf(
        X / mm, Y / mm, integrated_Ey.reshape(nx, ny), levels=50, cmap="viridis"
    )
    ax.contour(
        X / mm,
        Y / mm,
        integrated_Ey.reshape(nx, ny),
        levels=50,
        linewidths=0.1,
        linestyles="solid",
        colors="k",
    )
    fig.colorbar(contoury, ax=ax)
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    plt.savefig("/Users/nickvalverde/Desktop/y_transfields.pdf", dpi=400)
    plt.show()

# Set up paramters for interpolation
interp_R = 0.90 * aperture
interp_np = 50
interp_theta = np.linspace(0, 2 * np.pi, interp_np)

interp_x = interp_R * np.cos(interp_theta)
interp_y = interp_R * np.cos(interp_theta)

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
    len(wp.w3d.xmesh) - 1,
    len(wp.w3d.ymesh) - 1,
    integrated_Ex.reshape(nx, ny),
    wp.w3d.xmmin,
    wp.w3d.xmmax,
    wp.w3d.ymmin,
    wp.w3d.ymmax,
)

wp.top.getgrid2d(
    len(interp_x),
    interp_x,
    interp_y,
    interp_Ey,
    len(wp.w3d.xmesh) - 1,
    len(wp.w3d.ymesh) - 1,
    integrated_Ey.reshape(nx, ny),
    wp.w3d.xmmin,
    wp.w3d.xmmax,
    wp.w3d.ymmin,
    wp.w3d.ymmax,
)

# Evaluate the coefficients a_n and b_n for Ex and Ey.
n_order = 4
nterms = np.array([i for i in range(n_order)])
dtheta = interp_theta[1] - interp_theta[0]

Excoeff_array = np.zeros((2, len(nterms)))
Eycoeff_array = np.zeros((2, len(nterms)))
R = interp_R / aperture
pdb.set_trace()
for i in range(1, len(nterms)):
    # Define coefficient that comes from Fourier Analysis
    n = nterms[i]

    coeff = 2 * pow(1 / R, n) / np.pi
    Ax_integrand = interp_Ex.reverse() * np.cos(n * interp_theta)
    Bx_integrand = interp_Ex * np.sin(n * interp_theta)
    Ay_integrand = -1.0 * interp_Ey * np.sin(n * interp_theta)
    By_integrand = interp_Ey * np.cos(n * interp_theta)

    Ax = coeff * integrate.simpson(Ax_integrand, dx=dtheta)
    Bx = coeff * integrate.simpson(Bx_integrand, dx=dtheta)
    Ay = coeff * integrate.simpson(Ay_integrand, dx=dtheta)
    By = coeff * integrate.simpson(By_integrand, dx=dtheta)

    Excoeff_array[:, i] = Ax, Bx
    Eycoeff_array[:, i] = Ay, By

# Define coefficient that comes from Fourier Analysis
n = nterms[i]

coeff = 1 / 2 / np.pi
Ax_integrand = interp_Ex
Bx_integrand = 0
Ay_integrand = 0
By_integrand = interp_Ey

Ax = coeff * integrate.simpson(Ax_integrand, dx=dtheta)
Bx = 0
Ay = 0
By = coeff * integrate.simpson(By_integrand, dx=dtheta)

Excoeff_array[:, 0] = Ax, Bx
Eycoeff_array[:, 0] = Ay, By
