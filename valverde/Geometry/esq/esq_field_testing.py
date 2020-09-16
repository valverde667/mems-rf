"""Model ESQ doublet in Warp and pull information. Plotting the ESQ gradient
along with field strength is done. Calculating effective length of quadrupole.
Also, different quadrupole designs are used: solid cylinder rods (ESQ), hollow
cylindrical shell rods (ESQGrant), and Timo's design. """

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import sys

import warp as wp

# Useful constants
kV = wp.kV
mm = wp.mm
um = 1e-6

# Create mesh
wp.w3d.xmmin = -2.4 * mm
wp.w3d.xmmax = 2.4 * mm
wp.w3d.nx = 150

wp.w3d.ymmin = -2.4 * mm
wp.w3d.ymmax = 2.4 * mm
wp.w3d.ny = 150

wp.w3d.zmmin = -6 * mm
wp.w3d.zmmax = 6 * mm
wp.w3d.nz = 700

# Add boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.dirichlet

wp.w3d.l4symtry = True
solver = wp.MRBlock3D()
wp.registersolver(solver)


class ESQ:
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

    def __init__(self, radius=0.75 * mm, zc=2.2 * mm, length=1.59 * mm):

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


class ESQGrant:
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
voltage = 0.5 * kV
xycent = 1.3 * mm
separation = 2 * mm
length = 1.59 * mm
zc = separation / 2 + length / 2
wallvoltage = 0 * kV
aperture = 0.55 * mm
walllength = 0.1 * mm
wallzcent = separation / 2 + length + separation + walllength / 2

# Create left and right quads
leftconductor = ESQGrant(zc=-zc)
leftquad = leftconductor.generate(
    voltage=voltage, xcent=xycent, ycent=xycent, data=True
)
rightconductor = ESQGrant(zc=zc)
rightquad = rightconductor.generate(
    voltage=-voltage, xcent=xycent, ycent=xycent, data=True
)
# Create left and right grounded walls
leftwall = Wall().generate(apperture=aperture, voltage=wallvoltage, zcenter=-wallzcent)
rightwall = Wall().generate(apperture=aperture, voltage=wallvoltage, zcenter=wallzcent)

# Install Conductors and generate mesh
wp.installconductor(leftquad)
wp.installconductor(rightquad)
wp.installconductor(leftwall)
wp.installconductor(rightwall)
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
    else:
        pass

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
wp.setup()
leftquad.drawzx(filled=True)
rightquad.drawzx(filled=True)
rightwall.drawzx(filled=True)
leftwall.drawzx(filled=True)
wp.fma()
leftquad.drawxy(filled=True)
wp.fma()
wp.pfxy(iz=208, fill=1, filled=1)
wp.fma()
wp.pfzx(fill=1, filled=1)
wp.fma()

# Grab Fields
phi = wp.getphi()
phixy = wp.getphi()[:, :, zcenterindex]
Ex = wp.getselfe(comp="x")
gradex = Ex[1, 0, :] / wp.w3d.dx

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
ax.axvline(x=(wallzcent - walllength / 2) / mm, c="grey", lw=0.8, ls="--", label="Wall")
ax.axvline(x=-(wallzcent - walllength / 2) / mm, c="grey", lw=0.8, ls="--")
ax.axvline(x=(wallzcent + walllength / 2) / mm, c="grey", lw=0.8, ls="--")
ax.axvline(x=-(wallzcent + walllength / 2) / mm, c="grey", lw=0.8, ls="--")
plt.legend()
plt.show()

# Plot and calculate effective length
# Integrate over right esq. Note, this gradient is negative.
dEdx = abs(gradex[zzeroindex:])
ell = efflength(dEdx, wp.w3d.dz)
print("Effective Length = ", ell / mm)
# Plot integrand
fig, ax = plt.subplots()
ax.set_title("Integrand For Effective Length")
ax.set_ylabel(r"$|E(x=dx,y=0,z)$/dx| [kV mm$^{-2}$]")
ax.set_xlabel("z [mm]")
ax.scatter(z[zzeroindex:] / mm, dEdx / kV / 1000 / 1000, s=0.5)
# Annotate
ax.axhline(y=0, lw=0.5, c="k")
ax.axvline(x=esq2left / mm, c="r", lw=0.8, ls="--", label="ESQ Edges")
ax.axvline(x=esq2right / mm, c="r", lw=0.8, ls="--")
ax.axvline(x=(wallzcent - walllength / 2) / mm, c="grey", lw=0.8, ls="--", label="Wall")
ax.axvline(x=(wallzcent + walllength / 2) / mm, c="grey", lw=0.8, ls="--")
ax.legend()
plt.savefig("ell-integrand.svg", dpi=300)
plt.show()


# fig, ax = plt.subplots()
# ax.set_xlabel("x [mm]")
# ax.set_ylabel("y [mm]")
# X, Y = np.meshgrid(x, y)
# cont = ax.contour(X / mm, Y / mm, phixy, levels=50)
# cb = fig.colorbar(cont)
# plt.show()
