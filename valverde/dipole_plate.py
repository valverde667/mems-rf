# Script to simulate the dipole deflector plate used in lab. The system will
# be centered on the dipole z-axis.
import numpy as np
import matplotlib.pyplot as plt
import warp as wp

# Define usefule constants
mm = 1e-3
kV = 1e3
keV = 1e3

# ------------------------------------------------------------------------------
#    User Functions
# ------------------------------------------------------------------------------
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


# ------------------------------------------------------------------------------
#    Geoemtry and Mesh Setup
# ------------------------------------------------------------------------------
# Define geometries in systems
drift_to_dipole = 0.0 * mm
# drift_to_dipole = 25. * mm
dipole_length = 50.0 * mm
dipole_gap_width = 11.0 * mm
dipole_width = 1.5 * mm
dipole_bias = 4 * kV
# drift_to_analyzer = 185. * mm
drift_to_analyzer = 0.0
analyzer_slit_cent = 37.0 * mm
lattice_slit_center = -2.5 * mm


# Create mesh centering the dipole on the z-axis.
wp.w3d.xmmin = -15.0 * mm
wp.w3d.xmmax = 15.0 * mm
wp.w3d.nx = 200

wp.w3d.ymmin = -15.0 * mm
wp.w3d.ymmax = 15.0 * mm
wp.w3d.ny = 200

wp.w3d.zmmin = -dipole_length / 2 - 25 * mm
wp.w3d.zmmax = dipole_length / 2 + 20 * mm
wp.w3d.nz = 400


wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.dirichlet

# wp.w3d.l2symtry = True
solver = wp.MRBlock3D()
wp.registersolver(solver)

# Create conducting objects and load onto the mesh

topplate = wp.Box(
    xsize=dipole_width,
    ysize=70.0 * mm,
    zsize=dipole_length,
    voltage=-dipole_bias,
    zcent=0.0,
    xcent=dipole_gap_width / 2.0,
    ycent=0.0,
)
botplate = wp.Box(
    xsize=dipole_width,
    ysize=70.0 * mm,
    zsize=dipole_length,
    voltage=+dipole_bias,
    zcent=0.0,
    xcent=-dipole_gap_width / 2.0,
    ycent=0.0,
)


wp.installconductor(topplate)
wp.installconductor(botplate)

wp.generate()

# ------------------------------------------------------------------------------
#    Analysis
# Here the electric field (or other fields) are grabbed and analyzed.
# ------------------------------------------------------------------------------
x, y, z = wp.w3d.xmesh, wp.w3d.ymesh, wp.w3d.zmesh

# grab index of center of entrance slit and dipole center
x1 = getindex(x, lattice_slit_center, wp.w3d.dx)
x0 = getindex(x, 0.0, wp.w3d.dx)
y0 = getindex(y, 0.0, wp.w3d.dy)

# Grab indices for various z positions such as start of dipole and end
ind_dipole_start = getindex(z, drift_to_dipole - dipole_length / 2, wp.w3d.dz)
ind_dipole_end = getindex(z, drift_to_dipole + dipole_length / 2, wp.w3d.dz)
ind_dipole_zcent = getindex(z, 0.0, wp.w3d.dz)

# Create Warp plots. Useful for quick-checking
warpplots = True
if warpplots:
    wp.setup()
    topplate.drawzx(filled=True)
    botplate.drawzx(filled=True)
    wp.fma()

    wp.pfzx(fill=1, filled=1)
    wp.fma()

    wp.pfxy(iz=ind_dipole_zcent, fill=1, filled=1)
    wp.fma()

# Plot on axis electric field
Eavg = 2 * dipole_bias / dipole_gap_width
Ex = wp.getselfe(comp="x")
Ez = wp.getselfe(comp="z")
phi = wp.getphi()
phiz_enter = phi[x1, y0, :]
Ex0 = Ex[x0, y0, :]
Ex1 = Ex[x1, y0, :]
fig, ax = plt.subplots()
ax.plot(z / mm, Ex0 / Eavg, label=f"x=0.0 mm")
ax.plot(z / mm, Ex1 / Eavg, label=f"x={lattice_slit_center/mm:.2f} mm")
ax.set_title(r"Electric Field at x-positions")
ax.set_ylabel(r"$E_x/E_{avg}$")
ax.set_xlabel(r"$z$ [mm]")
ax.axvline(x=z[ind_dipole_start] / mm, c="k", ls="--")
ax.axvline(x=z[ind_dipole_end] / mm, c="k", ls="--")
ax.legend()
plt.show()

fig, ax = plt.subplots()
ax.plot(z / mm, Ez[x1, y0, :] / Eavg, label=f"x={lattice_slit_center/mm:.2f} mm")
ax.set_title(r"Electric Field at x-positions")
ax.set_ylabel(r"$E_z/E_{avg}$")
ax.set_xlabel(r"$z$ [mm]")
ax.axvline(x=z[ind_dipole_start] / mm, c="k", ls="--")
ax.axvline(x=z[ind_dipole_end] / mm, c="k", ls="--")
ax.legend()
plt.show()


# fig,ax = plt.subplots()
# ax.set_title("Potential Along z at x=slit center (3mm)")
# ax.set_xlabel("z [mm]")
# ax.set_ylabel(r"Potential Normalized by Plate Bias")
# ax.plot(z / mm, phiz_enter/kV)
# ax.axvline(x=z[ind_dipole_start] / mm, c='k', ls='--')
# ax.axvline(x=z[ind_dipole_end] / mm, c='k', ls='--')
# plt.show()
