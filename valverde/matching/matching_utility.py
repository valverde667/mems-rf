# Script contains functions and classes used in the transverse matching
# modeling. They were moved here to reduce clutter and also provide a place
# to focus on improving the functions and classes.
# Improvement suggestions are provided in the notes below.

import numpy as np
import scipy.constants as SC

# Define useful constants
mm = 1e-3
mrad = 1e-3
um = 1e-6
kV = 1e3
mrad = 1e-3
keV = 1e3
uA = 1e-6
MHz = 1e6
twopi = np.pi * 2

# ------------------------------------------------------------------------------
#   Note for Imporovement
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
#    Function and Class definitions
# Various functions and classes used in the script are defined here.
# ------------------------------------------------------------------------------
def create_combinations(s1, s2, s3, s4):
    """Utility function for creating combinations of the array elements in s1-s4."""
    combinations = np.array(list(itertools.product(s1, s2, s3, s4)))
    return combinations


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


def calc_gap_centers(E_s, mass, phi_s, dsgn_freq, dsgn_gap_volt):
    gap_dist = np.zeros(len(phi_s))
    for i in range(Ng):
        this_beta = beta(E_s, mass)
        this_cent = this_beta * SC.c / 2 / dsgn_freq
        cent_offset = (phi_s[i] - phi_s[i - 1]) * this_beta * SC.c / dsgn_freq / twopi
        if i < 1:
            gap_dist[i] = (phi_s[i] + np.pi) * this_beta * SC.c / twopi / dsgn_freq
        else:
            gap_dist[i] = this_cent + cent_offset

        dsgn_Egain = dsgn_gap_volt * np.cos(phi_s[i])
        E_s += dsgn_Egain
    return np.array(gap_dist).cumsum()


class Lattice:
    def __init__(self):
        self.zmin = 0.0
        self.zmax = None
        self.centers = None
        self.Np = None
        self.dz = None
        self.z = None
        self.grad = None
        self.gap_centers = None

        self.lattice_params = {
            "lq": None,
            "Vq": None,
            "rp": None,
            "Gstar": None,
            "Gmax": None,
        }

    def calc_Vset(self, Gmax):
        """Calculate the necessary voltage to generate the max gradient used"""

        Vset = 1.557857e-10 * Gmax
        return Vset

    def calc_Gmax(self, Volt):
        """Calculate max gradient needed to give desired voltage"""
        Gmax = Volt / 1.557857e-10
        return Gmax

    def hard_edge_match(
        self, lq, lq_eff, d, Vq, Nq, rp, scales, max_grad=2e9, res=10 * um
    ):
        """Create a hard-edge model for the ESQs.
        The ESQ centers will be placed at centers and kappa calculated. Each
        ESQ is given"""

        Lp = 2 * d * Nq + 4 * lq
        self.Np = int(Lp / res)
        self.z = np.linspace(0.0, Lp, self.Np)
        self.zmax = self.z.max()

        self.Np = int((self.zmax - self.zmin) / res)
        self.z = np.linspace(self.zmin, self.zmax, self.Np)
        self.grad = np.zeros(self.z.shape[0])

        Gstar = max_grad
        Vset = np.empty(Nq)

        # Find indices of lq centers and mask from center - lq/2 to center + lq/2
        masks = []
        self.centers = np.zeros(Nq)
        for i in range(Nq):
            this_zc = d + 2 * i * d + lq * i + lq / 2
            this_mask = (self.z >= this_zc - lq_eff / 2) & (
                self.z <= this_zc + lq_eff / 2
            )
            masks.append(this_mask)
            self.centers[i] = this_zc

        for i, mask in enumerate(masks):
            this_g = Gstar * scales[i]
            self.grad[mask] = this_g
            Vset[i] = self.calc_Vset(this_g)

        # Update the paramters dictionary with values used.
        updated_params = [lq, Vset, rp, Gstar, None]

        for key, value in zip(self.lattice_params.keys(), updated_params):
            self.lattice_params[key] = value

    def user_input_match(self, file_string, Nq, scales, lq=0.695 * mm):
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
        for key, value in zip(self.lattice_params.keys(), updated_params):
            self.lattice_params[key] = value

    def accel_lattice(
        self,
        gap_centers,
        file_string,
        scales,
        Lp,
        Nq=2,
        lq=0.695 * mm,
        g=2 * mm,
        res=25e-6,
    ):
        """Create the acceleration secttion of the lattice.

        At the moment, the acceleration portion only uses thin lens kicks.
        Thus, the gap centers are merlely used for ESQ placement. A field-free
        region exists after every unit (a unit is two RF wafers).
        A lattice period is comprised of one RF-unit starting at the RF acceelration
        plate (g/2 from the gap center) and extends to the beginning of the
        second acceleration plate.

        Note: this assumes one lattice period and only works for 2 ESQs.
        """

        iso_z, iso_grad = np.load(file_string[0]), np.load(file_string[1])
        zesq_extent = iso_z[-1] - iso_z[0]
        Gmax = np.max(iso_grad)
        g1, g2, g3 = gap_centers[:-1]
        Lp = g3 - g2 - g

        # Scale the two ESQs in place
        index = np.argmin(abs(iso_z))
        l_esq_grad = iso_grad[: index + 1]
        r_esq_grad = iso_grad[index + 1 :]
        l_esq_grad *= scales[0]
        r_esq_grad *= scales[1]
        Vsets = np.zeros(Nq)
        for i, s in enumerate(scales):
            if i % 2 == 0:
                Vsets[i] = self.calc_Vset(s * Gmax)
            else:
                Vsets[i] = -self.calc_Vset(s * Gmax)

        start = g1 + g / 2
        stop = 0
        interval = start - stop
        nsteps = int(interval / res)
        z = np.linspace(0, gap_centers[1] + g / 2, nsteps, endpoint=True)
        grad = np.zeros(len(z))

        # Attach field region
        zfield = iso_z.copy()
        zfield += z[-1] + res + iso_z.max()
        z = np.hstack((z, zfield))
        grad = np.hstack((grad, iso_grad))

        self.z = z
        self.grad = grad

        # Update the paramters dictionary with values used.
        updated_params = [lq, Vsets, None, None, Gmax]
        for key, value in zip(self.lattice_params.keys(), updated_params):
            self.lattice_params[key] = value


def solver(solve_matrix, dz, kappa, emit, Q):
    """Solve KV-envelope equations with Euler-cromer method.

    The solve_matrix input will be an Nx4 matrix where N is the number of steps
    to take in the solve and the four columns are the transverse position and
    angle in x and y. The solver will take a step dz and evaluate the equations
    with a fixed emittance and perveance Q. Kappa is assumed to be an array.
    """

    ux, uy = solve_matrix[:, 0], solve_matrix[:, 1]
    vx, vy = solve_matrix[:, 2], solve_matrix[:, 3]

    for n in range(1, solve_matrix.shape[0]):
        # Evaluate term present in both equations
        term = 2 * Q / (ux[n - 1] + uy[n - 1])

        # Evaluate terms for x and y
        term1x = pow(emit, 2) / pow(ux[n - 1], 3) + kappa[n - 1] * ux[n - 1]
        term1y = pow(emit, 2) / pow(uy[n - 1], 3) - kappa[n - 1] * uy[n - 1]

        # Update v_x and v_y first.
        vx[n] = (term + term1x) * dz + vx[n - 1]
        vy[n] = (term + term1y) * dz + vy[n - 1]

        # Use updated v to update u
        ux[n] = vx[n] * dz + ux[n - 1]
        uy[n] = vy[n] * dz + uy[n - 1]

    return solve_matrix


def calc_energy_gain(Vg, phi_s):
    return Vg * np.cos(phi_s)


def solver_with_accel(
    solve_matrix, dz, kappa, emit, Q, zmesh, gap_centers, Vg=7e3, phi_s=0.0, E=7e3
):
    """Solve KV-envelope equations with Euler-cromer method and acceleration kicks.

    The solve_matrix input will be an Nx4 matrix where N is the number of steps
    to take in the solve and the four columns are the transverse position and
    angle in x and y. The solver will take a step dz and evaluate the equations
    with a fixed emittance and perveance Q. Kappa is assumed to be an array.

    To incorporate acceleration, the zmesh and gap centers are provided. At the
    gap center a kick is applied to the angle in x and y. Additionally, the
    Q and emittance are adjusted following the scaling relationships for
    increased energy.
    """

    ux, uy = solve_matrix[:, 0], solve_matrix[:, 1]
    vx, vy = solve_matrix[:, 2], solve_matrix[:, 3]
    current_energy = E

    # Partition solve loop into chunks dealing with each part in the lattice.
    # First chunk is start-gap1. Then gap1-gap2 and gap2-Lp.
    gap1_ind = np.argmin(abs(gap_centers[0] - zmesh))
    gap2_ind = np.argmin(abs(gap_centers[1] - zmesh))

    # Do the first part of the advancement.
    for n in range(1, gap1_ind + 1):
        # Evaluate term present in both equations
        term = 2 * Q / (ux[n - 1] + uy[n - 1])

        # Evaluate terms for x and y
        term1x = pow(emit, 2) / pow(ux[n - 1], 3) + kappa[n - 1] * ux[n - 1]
        term1y = pow(emit, 2) / pow(uy[n - 1], 3) - kappa[n - 1] * uy[n - 1]

        # Update v_x and v_y first.
        vx[n] = (term + term1x) * dz + vx[n - 1]
        vy[n] = (term + term1y) * dz + vy[n - 1]

        # Use updated v to update u
        ux[n] = vx[n] * dz + ux[n - 1]
        uy[n] = vy[n] * dz + uy[n - 1]

    # save counter for next loop. Update angles, Q and emittance
    current_ind = n
    dE = calc_energy_gain(Vg, phi_s[0])
    rx_kick = vx[-1] / (1 + np.sqrt(dE / current_energy))
    ry_kick = vy[-1] / (1 + np.sqrt(dE / current_energy))
    Q = Q / pow(1 + dE / current_energy, 3.0 / 2.0)
    emit = emit / np.sqrt(1 + dE / current_energy)

    # update values
    current_energy += dE
    vx[current_ind] = rx_kick
    vy[current_ind] = ry_kick

    for n in range(current_ind + 1, gap2_ind + 1):
        # Evaluate term present in both equations
        term = 2 * Q / (ux[n - 1] + uy[n - 1])

        # Evaluate terms for x and y
        term1x = pow(emit, 2) / pow(ux[n - 1], 3) + kappa[n - 1] * ux[n - 1]
        term1y = pow(emit, 2) / pow(uy[n - 1], 3) - kappa[n - 1] * uy[n - 1]

        # Update v_x and v_y first.
        vx[n] = (term + term1x) * dz + vx[n - 1]
        vy[n] = (term + term1y) * dz + vy[n - 1]

        # Use updated v to update u
        ux[n] = vx[n] * dz + ux[n - 1]
        uy[n] = vy[n] * dz + uy[n - 1]

    # save counter for next loop. Update angles, Q and emittance
    current_ind = n
    dE = calc_energy_gain(Vg, phi_s[1])
    rx_kick = vx[-1] / (1 + np.sqrt(dE / current_energy))
    ry_kick = vy[-1] / (1 + np.sqrt(dE / current_energy))
    Q = Q / pow(1 + dE / current_energy, 3.0 / 2.0)
    emit = emit / np.sqrt(1 + dE / current_energy)

    # update values
    current_energy += dE
    vx[current_ind] = rx_kick
    vy[current_ind] = ry_kick

    for n in range(current_ind + 1, solve_matrix.shape[0]):
        # Evaluate term present in both equations
        term = 2 * Q / (ux[n - 1] + uy[n - 1])

        # Evaluate terms for x and y
        term1x = pow(emit, 2) / pow(ux[n - 1], 3) + kappa[n - 1] * ux[n - 1]
        term1y = pow(emit, 2) / pow(uy[n - 1], 3) - kappa[n - 1] * uy[n - 1]

        # Update v_x and v_y first.
        vx[n] = (term + term1x) * dz + vx[n - 1]
        vy[n] = (term + term1y) * dz + vy[n - 1]

        # Use updated v to update u
        ux[n] = vx[n] * dz + ux[n - 1]
        uy[n] = vy[n] * dz + uy[n - 1]

    return solve_matrix
