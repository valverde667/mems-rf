# Solve KV-envelope equations with given initial conditions and different
# lattice creations.

import numpy as np
import scipy.constants as SC
import matplotlib.pyplot as plt
import matplotlib as mpl
import streamlit as st
import pdb

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

# Define useful constants
mm = 1e-3
um = 1e-6
kV = 1e3
mrad = 1e-3
keV = 1e3
uA = 1e-6

# System and Geometry settings
lq = 0.695 * mm
d = 2.0 * mm
Vq = 0.6 * kV
Nq = 4
rp = 0.55 * mm
rsource = 0.25 * mm
G_hardedge = 2 * Vq / pow(rp, 2)
res = 10 * um

# Beam Settings
Q = 6.986e-5
emit = 1.336 * mm * mrad
init_E = 7 * keV
init_I = 10 * uA
div_angle = 3.78 * mrad * 0.0
Tb = 0.1  # eV


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


def setup_lattice(lq, d, Vq, Nq, G=G_hardedge, res=res):
    """Build hard-edge field gradient

    Function will setup mesh with hard edge gradient ESQ. The quads will have
    length lq and be seperated by a distance d. The lattice will be symmetric about
    0 and will be OFODO where O is the drift length d."""

    # Calculate length of lattice period.
    Lp = lq * Nq + d * (Nq + 1)
    Nz = int(Lp / res)
    z = np.linspace(0.0, Lp, Nz)

    # Find indices of lq centers and mask from center - lq/2 to center + lq/2
    masks = []
    for i in range(Nq):
        this_zc = (i + 1) * d + i * lq + lq / 2
        this_mask = (z >= this_zc - lq / 2) & (z <= this_zc + lq / 2)
        masks.append(this_mask)

    # Create gradient array
    gradz = np.zeros(z.shape[0])
    for i, mask in enumerate(masks):
        if i % 2 == 0:
            gradz[mask] = -G
        else:
            gradz[mask] = G

    # Shift z array to be symmetric
    z -= Lp / 2.0

    return (z, gradz)


class Lattice:
    def __init__(self):
        self.zmin = 0.0
        self.zmax = None
        self.centers = None
        self.Np = None
        self.dz = None
        self.z = None
        self.grad = None

        self.params = {"lq": None, "Vq": None, "rp": None, "Gstar": None, "Gmax": None}

    def calc_Vset(self, Gmax):
        """Calculate the necessary voltage to generate the max gradient used"""

        Vset = 1.557857e-10 * Gmax
        return Vset

    def hard_edge(self, lq, d, Vq, Nq, rp, res=10 * um):
        """Create a hard-edge model for the ESQs.
        The ESQ centers will be placed at centers and kappa calculated."""

        Lp = lq * Nq + d * (Nq + 1)
        self.Np = int(Lp / res)
        self.z = np.linspace(0.0, Lp, self.Np)
        self.zmax = self.z.max()

        self.Np = int((self.zmax - self.zmin) / res)
        self.z = np.linspace(self.zmin, self.zmax, self.Np)
        self.grad = np.zeros(self.z.shape[0])
        Gstar = abs(2.0 * Vq / pow(rp, 2))

        # Find indices of lq centers and mask from center - lq/2 to center + lq/2
        masks = []
        self.centers = np.zeros(Nq)
        for i in range(Nq):
            this_zc = (i + 1) * d + i * lq + lq / 2
            this_mask = (self.z >= this_zc - lq / 2) & (self.z <= this_zc + lq / 2)
            masks.append(this_mask)
            self.centers[i] = this_zc

        for i, mask in enumerate(masks):
            if i % 2 == 0:
                this_g = -Gstar
            else:
                this_g = Gstar

            self.grad[mask] = this_g

        # Update the paramters dictionary with values used.
        updated_params = [lq, Vq, rp, Gstar, None]

        for key, value in zip(self.params.keys(), updated_params):
            self.params[key] = value

        cent0 = zext / 2.0
        self.centers = np.array([cent0 * i for i in range(Nq + 1)])

    def user_input(self, file_string, Nq, scales, lq=0.695 * mm):
        """Create Nq matching section from extracted gradient.

        An isolated gradient is read in from the file_string = (path-s, path-grad)
        and loaded onto the mesh. The length of the ESQ should be provided and
        the separation distance calculated from the z-mesh provided by
        subtracting lq and then diving the resulting mesh length in half:
        i.e. d = (zmax - zmin - lq) / 2.
        Each ESQ is then placed at then placed with 2d interspacing:
            d-lq-2d-lq-2d-...-lq-d
        """

        zext, grad = np.load(file_string[0]), np.load(file_string[1])
        dz = zext[1] - zext[0]
        if zext.min() < -1e-9:
            # Shift z array to start at z=0
            zext += abs(zext.min())
            zext[0] = 0.0

        # Initialize the first arrays quadrupole with corresponding mesh.
        # Calculate the set voltage needed to create the scaled gradient and
        # record.
        Vset = np.zeros(Nq)
        self.z = zext.copy()
        self.grad = grad.copy() * scales[0]
        Gmax = grad.max()
        Vset = np.array([self.calc_Vset(Gmax * scale) * kV for scale in scales])

        # Loop through remaining number of quadrupoles after the first and build
        # up the corresponding mesh and gradient.
        for i in range(1, Nq):
            this_z = zext.copy() + self.z[-1] + dz
            this_grad = grad.copy() * scales[i]
            self.z = np.hstack((self.z, this_z))
            self.grad = np.hstack((self.grad, this_grad))

        # Update the paramters dictionary with values used.
        updated_params = [lq, Vset, None, None, grad.max()]

        for key, value in zip(self.params.keys(), updated_params):
            self.params[key] = value


user_input = True
if user_input:
    lattice = Lattice()
    file_names = ("iso_zgrad.npy", "iso_esq_grad.npy")
    scales = []
    for i in range(Nq):
        if i % 2 == 0:
            scales.append(1.0)
        else:
            scales.append(-1.0)

    scales = np.array(scales)
    scales *= (0.4, 0.5, 0.3, 0.2)

    lattice.user_input(file_names, Nq, scales=scales)
    z, gradz = lattice.z, lattice.grad

else:
    lattice = Lattice()
    lattice.hard_edge(lq, d, Vq, Nq, rp)
    z, gradz = lattice.z, lattice.grad


# Beam specifications
beam = wp.Species(type=wp.Argon, charge_state=1)
mass_eV = beam.mass * pow(SC.c, 2) / wp.jperev
beam.ekin = init_E
beam.ibeam = init_I
beam.a0 = rsource
beam.b0 = rsource
beam.ap0 = div_angle
beam.bp0 = div_angle
beam.ibeam = init_I
beam.vbeam = 0.0
beam.ekin = init_E
vth = np.sqrt(Tb * wp.jperev / beam.mass)
wp.derivqty()


# Solve KV equations
dz = z[1] - z[0]
kappa = wp.echarge * gradz / 2.0 / init_E / wp.jperev
ux_initial, uy_initial = rsource, rsource
vx_initial, vy_initial = div_angle, div_angle


soln_matrix = np.zeros(shape=(len(z), 4))
soln_matrix[0, :] = ux_initial, uy_initial, vx_initial, vy_initial

# Grab position and angle arrays from matrix
ux = soln_matrix[:, 0]
uy = soln_matrix[:, 1]
vx = soln_matrix[:, 2]
vy = soln_matrix[:, 3]

# Main loop to update equation. Loop through matrix and update entries.
for n in range(1, len(soln_matrix)):
    # Evaluate term present in both equations
    term = 2 * Q / (ux[n - 1] + uy[n - 1])

    # Evaluate terms for x and y
    term1x = pow(emit, 2) / pow(ux[n - 1], 3) - kappa[n - 1] * ux[n - 1]
    term1y = pow(emit, 2) / pow(uy[n - 1], 3) + kappa[n - 1] * uy[n - 1]

    # Update v_x and v_y first.
    vx[n] = (term + term1x) * dz + vx[n - 1]
    vy[n] = (term + term1y) * dz + vy[n - 1]

    # Use updated v to update u
    ux[n] = vx[n] * dz + ux[n - 1]
    uy[n] = vy[n] * dz + uy[n - 1]


# Plot results
fig, ax = plt.subplots()
ax.set_xlabel("z (mm)")
ax.set_ylabel(r"$r_x,\,r_y$ (mm)")
ax.plot(z / mm, ux / mm, c="k", label=r"$r_x$")
ax.plot(z / mm, uy / mm, c="b", label=r"$r_y$")
ax.axhline(y=rp / mm, ls="--", lw=1, c="r", label="Aperture")
ax.legend()
