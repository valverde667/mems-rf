# Script for various modeling and characterizing of ESQ conductors. As of right,
# the effective length and multipole moments are calculated using one of the
# conductor classes; either a solid or hollow cylindrical rod. The doublet system
# is loaded onto a mesh in isolation with no RF-acceleration gap field. However,
# the grounded conducting planes are placed on the mesh. This helps with viewing
# visualization and is somewhat illustrative of the real system.
# The maximum voltage on the mesh is also located and stored.
# The script can be iterated using the iterate_esq.sh script. All the data
# is stored in a .csv file called multipole_data.csv. The path should be changed
# for any new user.

# Create argument parser for scaling. Must be done before importing Warp
import warpoptions

# Scale pole argument will set the radius of the ESQ rod in units of aperture
# radius
warpoptions.parser.add_argument("--scale_pole_rad", default=False, type=float)

# Scale length argument will set the length of the ESQ rod in units of aperture
# radius
warpoptions.parser.add_argument("--scale_length", default=False, type=float)

# Rod fraction controls where to chop rods using the simulation box. A fraction
# of 0 will place simulation box just before rod giving no rod. A fraction of 2
# will place the chop at 2R giving the full rod.
warpoptions.parser.add_argument("--rod_fraction", default=False, type=float)
warpoptions.parser.add_argument("--voltage", default=False, type=float)
inputs = warpoptions.parser.parse_args()
if inputs.scale_pole_rad != False:
    scale_pole_rad = inputs.scale_pole_rad
else:
    # Around optimum value found for isolated single quad.
    scale_pole_rad = 1.304
if inputs.scale_length != False:
    scale_Lesq = inputs.scale_length
else:
    # Actual length of 0.695 * mm
    scale_Lesq = 1.264
if inputs.rod_fraction != False:
    rod_fraction = inputs.rod_fraction
else:
    # Full rod plus spacing between rod end and mesh limit
    rod_fraction = 1.0

if inputs.voltage != False:
    voltage = inputs.voltage
else:
    voltage = 400.0

from imports import *
from esq_geo import Mems_ESQ_SolidCyl


# Save string for convenience
savepath = "/Users/nickvalverde/Desktop/ESQ_files/"

print(f"--Using Pole Scale Factor of {scale_pole_rad}")
print(f"--Using ESQ Length Scale Factor of {scale_Lesq}")
print(f"--Using Rod Fraction of {rod_fraction}")
print(f"--Using Voltage {voltage/kV:.2f} [kV]")


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

    def __init__(self, radius=0.5 * mm, zc=0.0 * mm, length=0.695 * mm):
        self.radius = radius
        self.zc = zc
        self.length = length

    def pole(self, voltage, xcent, ycent):
        """Create individual electrode for ESQ.

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

    def generate(self, voltage, xcent, ycent, data=False, chop=False):
        """Combine four electrodes to form ESQ.

        Note that in the xy-plane the voltage for the top/bottom electrode is
        set to +.
        """
        rp = 0.55 * mm
        c = rp + self.radius
        # Create four poles
        top = self.pole(voltage=voltage, xcent=0, ycent=c)
        bottom = self.pole(voltage=voltage, xcent=0, ycent=-c)
        left = self.pole(voltage=-voltage, xcent=-c, ycent=0)
        right = self.pole(voltage=-voltage, xcent=c, ycent=0)

        # Combine poles into single ESQ
        conductor = top + bottom + left + right

        if chop == True:
            # Create surrounding box to chop rods in half.
            chop_box_out = wp.Box(
                xsize=10 * mm,
                ysize=10 * mm,
                zsize=10 * mm,
                zcent=self.zc,
                xcent=0.0,
                ycent=0.0,
            )
            chop_box_in = wp.Box(
                xsize=2 * (rp + self.radius),
                ysize=2 * (rp + self.radius),
                zsize=10 * mm,
                zcent=self.zc,
                xcent=0.0,
                ycent=0.0,
            )
            box = chop_box_out - chop_box_in
            conductor = conductor - box

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

    """Grab index of specifed value within a mesh"""

    # Check if value is already in mesh
    if value in mesh:
        index = np.where(mesh == value)[0][0]

    else:
        index = np.argmin(abs(mesh - value))

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


def calc_zmatch_sect(lq, d, Nq=4):
    """Calculate the z-centers to place quadrupoles"""
    zcents = np.empty(Nq)
    for i in range(Nq):
        this_zcent = d + 2 * i * d + lq * i + lq / 2
        zcents[i] = this_zcent

    return zcents


# ------------------------------------------------------------------------------
#                    Beam Settings
# Beam specifications that are useful for calculating quantities like kappa
# ------------------------------------------------------------------------------
beam = wp.Species(type=wp.Argon, charge_state=+1)
mass = beam.mass
beam.ekin = 7e3

# ------------------------------------------------------------------------------
#                    Logical Flags
# Logical flags for controlling various routines within the script. All flags
# prefixed with l_.
# ------------------------------------------------------------------------------
l_warpplots = True
l_make_effective_length_plots = True
l_make_transField_plots = False
l_plot_breakdown = False
l_make_3d_integrand_plot = False
l_multple_barplots = False
# ------------------------------------------------------------------------------
#                     Create and load mesh and conductors
# ------------------------------------------------------------------------------
# Set parameters for conductors
separation = 0 * mm
Nesq = 1

zc = 0 * mm
wallvoltage = 0 * kV
aperture = 0.55 * mm
pole_rad = aperture * scale_pole_rad
ESQ_length = aperture * scale_Lesq
xycent = aperture + pole_rad
walllength = 0.1 * mm
wallzcent = ESQ_length + 1.0 * mm + walllength / 2

Nq = 4
d = 3 * mm
quad_zcs = calc_zmatch_sect(ESQ_length, d, Nq=Nq)

# Creat mesh using conductor geometries (above) to keep resolution consistent
wp.w3d.xmmax = 1.5 * mm
wp.w3d.xmmin = -wp.w3d.xmmax
design_dx = 10 * um
calc_nx = (wp.w3d.xmmax - wp.w3d.xmmin) / design_dx
wp.w3d.nx = int(calc_nx)

wp.w3d.ymmax = wp.w3d.xmmax
wp.w3d.ymmin = wp.w3d.xmmin
wp.w3d.ny = wp.w3d.nx

# Calculate nz to get about designed dz

# wp.w3d.zmmax = 3.3475 * mm
# wp.w3d.zmmin = -wp.w3d.zmmax
wp.w3d.zmmin = quad_zcs[0] - ESQ_length / 2 - d
wp.w3d.zmmax = quad_zcs[-1] + ESQ_length / 2 + d
design_dz = 65 * um
calc_nz = (wp.w3d.zmmax - wp.w3d.zmmin) / design_dz
wp.w3d.nz = int(calc_nz)

# Add boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.periodic
wp.f3d.mgtol = 1e-8
solver = wp.MRBlock3D()
wp.registersolver(solver)

# Create Quadrupole
for i, zcs in enumerate(quad_zcs):
    if i % 2 == 0:
        this_volt = voltage
    else:
        this_volt = -voltage

    this_quad = Mems_ESQ_SolidCyl(zcs, this_volt, -this_volt, chop=True)
    this_quad.set_geometry(rp=aperture, R=pole_rad, lq=ESQ_length)
    wp.installconductor(this_quad.generate())

wp.generate()
# ------------------------------------------------------------------------------
#                     Calculate effective length
# ------------------------------------------------------------------------------
# Rename meshes and find indicesfor the mesh z-center and z-center of right quad
x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh
zcenterindex = getindex(z, zc, wp.w3d.dz)
xzeroindex = getindex(x, 0.0, wp.w3d.dx)
yzeroindex = getindex(y, 0.0, wp.w3d.dy)

# Grab Fields
phi = wp.getphi()
phixy = phi[:, :, zcenterindex]
Ex = wp.getselfe(comp="x")
Ey = wp.getselfe(comp="y")
Ez = wp.getselfe(comp="z")
Emag = wp.getselfe(comp="E")
gradex = Ex[xzeroindex + 1, yzeroindex, :] / wp.w3d.dx

# iparts = [0]
# c = [(quad_zcs[i]/ mm + quad_zcs[i+1]/mm) / 2 for i in range(len(quad_zcs) - 1)]
# for cc in c:
#     iparts.append(np.argmin(abs(cc-z/mm)))
# iparts.append(len(z) -1)

# grad_array = []
# z_array = []
# for i in range(len(iparts)-1):
#     grad_array.append(gradex[iparts[i]:iparts[i+1]+1])
#     z_array.append(z[iparts[i]:iparts[i+1]+1])

# fig, ax = plt.subplots()
# colors = ['k', 'b', 'g', 'm']
# for i in range(len(colors)):
#     ax.plot(z_array[i]/mm, grad_array[i], c=colors[i])

# grad_array = np.array(grad_array)
# z_array = np.array(z_array)
# np.save("matching_section_gradient", grad_array)
np.save("gradient", gradex)
np.save("z", z)

# Plot and calculate effective length
dEdx = abs(gradex[:])
ell = efflength(dEdx, wp.w3d.dz)
print("Effective Length = ", ell / mm)
kappa = wp.echarge * gradex / 2 / beam.ekin / wp.jperev
stop
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

# Find max electric fields. To do this, the xy-plane for each grid point in z is
# examined and the maximum field found for each component.
Emaxs = np.zeros(len(z))
for i in range(len(z)):
    this_E = Emag[:, :, i]
    Emaxs[i] = np.max(this_E)

zmax_ind = np.argmax(Emaxs)
xmax_ind, ymax_ind = np.unravel_index(
    Emag[:, :, zmax_ind].argmax(), Emag[:, :, zmax_ind].shape
)
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
n_order = 14
nterms = np.array([i for i in range(1, n_order + 1)])
dtheta = interp_theta[1] - interp_theta[0]

Ancoeff_array = np.zeros(len(nterms))
Bncoeff_array = np.zeros(len(nterms))

R = interp_R / aperture

for i in range(1, len(nterms)):
    n = nterms[i]

    coeff = pow(1.0 / R, n - 1) / 2 / np.pi
    # Partition Ex and Ey parts of integral for An and Bn for clarity
    An_Ex_integrand = interp_Ex * np.cos((n - 1) * interp_theta)
    An_Ey_integrand = -interp_Ey * np.sin((n - 1) * interp_theta)
    An_integrand = An_Ex_integrand + An_Ey_integrand

    Bn_Ex_integrand = interp_Ex * np.sin((n - 1) * interp_theta)
    Bn_Ey_integrand = interp_Ey * np.cos((n - 1) * interp_theta)
    Bn_integrand = Bn_Ex_integrand + Bn_Ey_integrand

    An = coeff * integrate.trapezoid(An_integrand, dx=dtheta)
    Bn = coeff * integrate.trapezoid(Bn_integrand, dx=dtheta)
    Ancoeff_array[n - 1] = An
    Bncoeff_array[n - 1] = Bn

# Use maximum multipole value for normalization
norm = np.max(abs(Ancoeff_array) + abs(Bncoeff_array))
nmax_index = np.argmax(abs(Ancoeff_array) + abs(Bncoeff_array))
An_norm = np.max(abs(Ancoeff_array))
Bn_norm = np.max(abs(Bncoeff_array))

# ------------------------------------------------------------------------------
#                    Data Storage
# ------------------------------------------------------------------------------
# Store data in a dataframe and append to csv file. If csv file already exists
# the column headers are ignored. If not, the file is created with headers.
filename = "multipole_data.csv"
file_exists = filename in os.listdir(savepath)
df = pd.DataFrame()
df["init"] = [np.nan]
df["n-max"] = nmax_index + 1
df["R_rod/R_aper"] = scale_pole_rad
df["L_esq/R_aper"] = scale_Lesq
df["rod-fraction"] = rod_fraction
df["separation[mm]"] = separation
df["n-interp"] = interp_np
df["voltage"] = voltage
df["Emag"] = np.max(Emaxs)
df["xmax_ind"] = xmax_ind
df["ymax_ind"] = ymax_ind
df["zmax_ind"] = zmax_ind
for i in range(len(nterms)):
    # Loop through n-poles and create column header
    df[f"Norm A{i+1}"] = Ancoeff_array[i] / An_norm
    df[f"Norm B{i+1}"] = Bncoeff_array[i] / Bn_norm
for i in range(len(nterms)):
    df[f"A{i+1}"] = Ancoeff_array[i]
    df[f"B{i+1}"] = Bncoeff_array[i]
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
print("### Coeff. Values Squared Normalized by Maximum Coeff. ###")

for i, n in enumerate(nterms):
    print(f"####  n={n}  ####")
    print(f"(An, Bn): ({Ancoeff_array[i]:.5E},  {Bncoeff_array[i]:.5E})")
    print(f"Normed An: {Ancoeff_array[i]/An_norm:.5E}")
    print(f"Normed Bn: {Bncoeff_array[i]/Bn_norm:.5E}")
    print("")
# ------------------------------------------------------------------------------
#                          Plotting Section
# All plots and visualization should be put here.
# ------------------------------------------------------------------------------
# Create Warp plots. Useful for quick-checking
if l_warpplots:
    wp.setup()
    # leftquad.drawzx(filled=True)
    # rightwall.drawzx(filled=True)
    # leftwall.drawzx(filled=True)
    # wp.fma()

    # leftquad.drawxy(filled=True)
    # wp.fma()

    wp.pfxy(iz=zcenterindex, fill=1, filled=1)
    wp.fma()

    # find center of box
    # this_iz = getindex(
    #     z, leftquad.zc - leftquad.lq / 2.0 - leftquad.copper_zlen, wp.w3d.dz
    # )
    # wp.pfxy(iz=this_iz, fill=1, filled=1)
    # wp.fma()

    # this_iz = getindex(
    #     z, leftquad.zc + leftquad.lq / 2.0 + leftquad.copper_zlen, wp.w3d.dz
    # )
    # wp.pfxy(iz=this_iz, fill=1, filled=1)
    # wp.fma()

    wp.pfzx(fill=1, filled=1)
    wp.fma()

    wp.pfxy(
        plotselfe=1,
        plotphi=0,
        comp="x",
        fill=1,
        filled=1,
        contours=100,
        iz=zcenterindex,
    )
    wp.fma()

    wp.pfxy(
        plotselfe=1,
        plotphi=0,
        comp="y",
        fill=1,
        filled=1,
        contours=100,
        iz=zcenterindex,
    )
    wp.fma()

if l_make_effective_length_plots:
    # Create plot of Ex gradient
    fig, ax = plt.subplots()
    ax.set_xlabel("z (mm)")
    ax.set_ylabel(r"$E_x(dx, 0, z)$/dx (kV mm$^{-2}$)")
    ax.set_title(r"$E_x$ Gradient One Grid-cell Off-axis vs z")
    ax.scatter(z / mm, gradex / kV / 1e6, c="k", s=1.2)
    ax.axhline(y=0, c="k", lw=0.5)
    ax.axvline(x=0, c="k", lw=0.5)

    # add ESQ markers to plot
    esq1left = -zc - ESQ_length / 2
    esq1right = -zc + ESQ_length / 2
    # esq2left = zc - ESQ_length / 2
    # esq2right = zc + ESQ_length / 2
    ax.axvline(x=esq1left / mm, c="b", lw=0.8, ls="--", label="ESQ Ends")
    ax.axvline(x=esq1right / mm, c="b", lw=0.8, ls="--")
    ax.axvline(x=-ell / 2 / mm, c="g", lw=0.8, ls="--")
    ax.axvline(x=ell / 2 / mm, c="g", lw=0.8, ls="--")
    # ax.axvline(x=esq2left / mm, c="r", lw=0.8, ls="--", label="Second ESQ")
    # ax.axvline(x=esq2right / mm, c="r", lw=0.8, ls="--")
    # ax.axvline(
    #     x=(wallzcent - walllength / 2) / mm, c="grey", lw=0.8, ls="--", label="Wall"
    # )
    # ax.axvline(x=-(wallzcent - walllength / 2) / mm, c="grey", lw=0.8, ls="--")
    # ax.axvline(x=(wallzcent + walllength / 2) / mm, c="grey", lw=0.8, ls="--")
    # ax.axvline(x=-(wallzcent + walllength / 2) / mm, c="grey", lw=0.8, ls="--")
    plt.legend()
    plt.savefig(savepath + "full-mesh.svg", dpi=400)
    plt.show()

if l_make_effective_length_plots:
    fig, ax = plt.subplots()
    ax.set_title(
        f"Integrand For Effective Length {ell/mm:.4f} mm, zc = {zc/mm :.4f} mm, n = {Nesq}, Lq = {ESQ_length/mm:.4f} mm",
        fontsize="small",
    )
    ax.set_ylabel(r"$|E(x=dx,y=0,z)$/dx| (kV mm$^{-2}$)")
    ax.set_xlabel("z (mm)")
    ax.scatter(z / mm, dEdx / kV / 1000 / 1000, s=0.5)
    # Annotate
    ax.axhline(y=0, lw=0.5, c="k")
    ax.axvline(x=-ell / 2 / mm, c="r", lw=0.8, ls="--", label=" Effective ESQ Edges")
    ax.axvline(x=ell / 2 / mm, c="r", lw=0.8, ls="--")
    # ax.axvline(
    #     x=(wallzcent - walllength / 2) / mm, c="grey", lw=0.8, ls="--", label="Wall"
    # )
    # ax.axvline(x=(wallzcent + walllength / 2) / mm, c="grey", lw=0.8, ls="--")
    ax.legend()
    plt.savefig(savepath + "integrand.png", dpi=400)
    plt.show()

if l_make_transField_plots:
    fig, ax = plt.subplots()
    ax.set_title(r"$E_x(x,y,z=zcent)$")
    X, Y = np.meshgrid(x, y, indexing="ij")
    contourx = ax.contourf(
        X / mm, Y / mm, Ex[:, :, zcenterindex], levels=500, cmap="viridis"
    )
    ax.contour(
        X / mm,
        Y / mm,
        Ex[:, :, zcenterindex],
        levels=100,
        linewidths=0.1,
        linestyles="solid",
        colors="k",
    )
    fig.colorbar(contourx, ax=ax)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    plt.savefig("/Users/nickvalverde/Desktop/Ex_original.pdf", dpi=400)
    plt.show()

    fig, ax = plt.subplots()
    ax.set_title(r" $E_y(x,y,z=zcent)$")
    contourx = ax.contourf(
        X / mm, Y / mm, Ey[:, :, zcenterindex], levels=500, cmap="viridis"
    )
    ax.contour(
        X / mm,
        Y / mm,
        Ey[:, :, zcenterindex],
        levels=100,
        linewidths=0.1,
        linestyles="solid",
        colors="k",
    )
    fig.colorbar(contourx, ax=ax)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
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

if l_plot_breakdown:
    fig, ax = plt.subplots()
    X, Y = np.meshgrid(x, y)
    cp = ax.contourf(X / mm, Y / mm, Emag[:, :, zmax_ind] / 1e7, levels=50)
    # cp.cmap.set_under('w')
    # cp.set_clim(0.05)
    cbar = fig.colorbar(cp)
    cbar.set_label(r"$|E|/10^7$ (V/m)", rotation=90)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Location of 50%-Breakdown Limit (xy-plane)")
    ax.scatter(
        [x[xmax_ind] / mm],
        [y[ymax_ind] / mm],
        c="r",
        marker="x",
        s=100,
        label="Max Field",
    )
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.savefig("Exy.svg")

    fig, ax = plt.subplots()
    Z, X = np.meshgrid(z, x)
    cp = ax.contourf(Z / mm, X / mm, Emag[:, ymax_ind, :] / 1e7, levels=50)
    # cp.cmap.set_under('w')
    # cp.set_clim(0.05)
    cbar = fig.colorbar(cp)
    cbar.set_label(r"$|E|/10^7$ (V/m)", rotation=90)
    ax.set_xlabel("z (mm)")
    ax.set_ylabel("x (mm)")
    ax.set_title("Location of 50%-Breakdown Limit (xz-plane)")
    ax.scatter(
        [z[zmax_ind] / mm],
        [x[xmax_ind] / mm],
        c="r",
        marker="x",
        s=100,
        label="Max Field",
    )
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    plt.savefig("Exz.svg")
    plt.show()

if l_make_3d_integrand_plot:
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

if l_multple_barplots:
    # Plot An, Bn and An+Bn on bar plot where height represents fraction of Max pole
    ax.bar3d(nterms, 1 * y3, z3, xbar_width, ybar_width, An / norm, color="b")
    ax.bar3d(nterms, 3 * y3, z3, xbar_width, ybar_width, Bn / norm, color="g")
    ax.bar3d(nterms, 6 * y3, z3, xbar_width, ybar_width, (An + Bn) / norm, color="k")

    ax.set_title(
        rf"Normalized Squared-Multipole Coefficients for $E(x,y)$",
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
        rf"Normalized Squared-Multipole Coefficients (Dominant Term Removed)",
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


if interp:
    # Find index of closest two points
    dists = np.sqrt(pow(a6 - interp_val, 2))
    nearest_inds = dists.argsort()[:2]
    yvals = a6[nearest_inds]
    xvals = rod_fracs[nearest_inds]

    s = pd.Series([xvals[0], np.nan, xvals[1]], index=[yvals[0], interp_val, yvals[1]])
    interp_data = s.interpolate(method="index")
    val = interp_data.iloc[1]

    ax.axvline(x=val, c="k", lw=1, ls="--", label=rf"$R/r_p$ = {val:.3f}")
    ax.legend()
