# Script contains functions and classes used in the transverse matching
# modeling. They were moved here to reduce clutter and also provide a place
# to focus on improving the functions and classes.
# Improvement suggestions are provided in the notes below.

import numpy as np
import scipy.constants as SC
import scipy.optimize as sciopt
import itertools
import pdb

import warp as wp


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
    Ng = len(phi_s)
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
        self, lq, lq_eff, d, Vq, Nq, rp, scales, max_grad=2e9, res=25 * um
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

    def acceleration_lattice(
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

        # Scale the two ESQs in place
        index = int(len(iso_z) / 2)
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

        start = g1 - g / 2
        stop = g2 + g / 2
        interval = stop - start
        nsteps = int(interval / res)
        z = np.linspace(0, interval, nsteps, endpoint=True)
        grad = np.zeros(len(z))

        # Check if the gradient field provided fills the space. It could happen
        # that the additional free space is less than the zextent of the gradient.
        # in this case, print out a warning and use the full zmesh to complete
        # the array. If there is more space, then the gradient field can be
        # woven in.
        drift_space = Lp - stop
        if drift_space <= zesq_extent + 6 * res:
            print(f"Fill factor: {stop/drift_space:3f}. Using zmesh provided.")
            # Attach field region
            zfield = iso_z.copy()
            zfield += z[-1] + res + iso_z.max()
            z = np.hstack((z, zfield))
            grad = np.hstack((grad, iso_grad))

        else:
            print(f"Fill factor: {stop/drift_space:3f}. Stitching in field.")
            # Treat left side of stitching
            lstart = stop + res
            lstop = lstart + drift_space / 2
            linterval = lstop - lstart
            lnsteps = int(linterval / res)

            lz = np.linspace(lstart, lstop, lnsteps, endpoint=True)
            lgrad = np.zeros(len(lz))

            # Include z from input file
            zfield = iso_z.copy()
            zfield += lz[-1] + res + iso_z.max()

            # Now treat right side
            rstart = zfield[-1] + res
            rstop = Lp
            rinterval = rstop - rstart
            rnsteps = int(rinterval / res)

            rz = np.linspace(rstart, rstop, rnsteps, endpoint=True)
            rgrad = np.zeros(len(rz))

            z = np.hstack((z, lz, zfield, rz))
            grad = np.hstack((grad, lgrad, iso_grad, rgrad))

        self.z = z
        self.grad = grad

        # Update the paramters dictionary with values used.
        updated_params = [lq, Vsets, None, None, Gmax]
        for key, value in zip(self.lattice_params.keys(), updated_params):
            self.lattice_params[key] = value


def solver(solve_matrix, z, kappa, emit, Q):
    """Solve KV-envelope equations with Euler-cromer method.

    The solve_matrix input will be an Nx4 matrix where N is the number of steps
    to take in the solve and the four columns are the transverse position and
    angle in x and y. The solver will take a step dz and evaluate the equations
    with a fixed emittance and perveance Q. Kappa is assumed to be an array.
    """

    ux, uy = solve_matrix[:, 0], solve_matrix[:, 1]
    vx, vy = solve_matrix[:, 2], solve_matrix[:, 3]

    for n in range(1, solve_matrix.shape[0]):
        this_dz = z[n] - z[n - 1]
        # Evaluate term present in both equations
        term = 2 * Q / (ux[n - 1] + uy[n - 1])

        # Evaluate terms for x and y
        term1x = pow(emit, 2) / pow(ux[n - 1], 3) - kappa[n - 1] * ux[n - 1]
        term1y = pow(emit, 2) / pow(uy[n - 1], 3) + kappa[n - 1] * uy[n - 1]

        # Update v_x and v_y first.
        vx[n] = (term + term1x) * this_dz + vx[n - 1]
        vy[n] = (term + term1y) * this_dz + vy[n - 1]

        # Use updated v to update u
        ux[n] = vx[n] * this_dz + ux[n - 1]
        uy[n] = vy[n] * this_dz + uy[n - 1]

    return solve_matrix


def calc_energy_gain(Vg, phi_s):
    return Vg * np.cos(phi_s)


def solver_with_accel(
    solve_matrix,
    z,
    kappa,
    emit,
    Q,
    gap_centers,
    Vg,
    phi_s,
    E,
    history=False,
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
    gap1_ind = np.argmin(abs(gap_centers[0] - z))
    gap2_ind = np.argmin(abs(gap_centers[1] - z))

    # Initialize history arrays for Q and emittance
    history_Q = [Q]
    history_emit = [emit]

    # Do the first part of the advancement.
    for n in range(1, gap1_ind + 1):
        history_Q.append(Q)
        history_emit.append(emit)

        this_dz = z[n] - z[n - 1]
        # Evaluate term present in both equations
        term = 2 * Q / (ux[n - 1] + uy[n - 1])

        # Evaluate terms for x and y
        term1x = pow(emit, 2) / pow(ux[n - 1], 3) - kappa[n - 1] * ux[n - 1]
        term1y = pow(emit, 2) / pow(uy[n - 1], 3) + kappa[n - 1] * uy[n - 1]

        # Update v_x and v_y first.
        vx[n] = (term + term1x) * this_dz + vx[n - 1]
        vy[n] = (term + term1y) * this_dz + vy[n - 1]

        # Use updated v to update u
        ux[n] = vx[n] * this_dz + ux[n - 1]
        uy[n] = vy[n] * this_dz + uy[n - 1]

    # save counter for next loop. Update angles, Q and emittance
    current_ind = n
    dE = calc_energy_gain(Vg, phi_s[0])
    rx_kick = vx[current_ind - 1] / (1 + np.sqrt(dE / current_energy))
    ry_kick = vy[current_ind - 1] / (1 + np.sqrt(dE / current_energy))
    Q = Q / pow(1 + dE / current_energy, 3.0 / 2.0)
    emit = emit / np.sqrt(1 + dE / current_energy)

    # update values
    current_energy += dE
    vx[current_ind] = rx_kick
    vy[current_ind] = ry_kick

    for n in range(current_ind + 1, gap2_ind + 1):
        history_Q.append(Q)
        history_emit.append(emit)

        this_dz = z[n] - z[n - 1]
        # Evaluate term present in both equations
        term = 2 * Q / (ux[n - 1] + uy[n - 1])

        # Evaluate terms for x and y
        term1x = pow(emit, 2) / pow(ux[n - 1], 3) - kappa[n - 1] * ux[n - 1]
        term1y = pow(emit, 2) / pow(uy[n - 1], 3) + kappa[n - 1] * uy[n - 1]

        # Update v_x and v_y first.
        vx[n] = (term + term1x) * this_dz + vx[n - 1]
        vy[n] = (term + term1y) * this_dz + vy[n - 1]

        # Use updated v to update u
        ux[n] = vx[n] * this_dz + ux[n - 1]
        uy[n] = vy[n] * this_dz + uy[n - 1]

    # save counter for next loop. Update angles, Q and emittance
    current_ind = n
    dE = calc_energy_gain(Vg, phi_s[1])
    rx_kick = vx[current_ind - 1] / (1 + np.sqrt(dE / current_energy))
    ry_kick = vy[current_ind - 1] / (1 + np.sqrt(dE / current_energy))
    Q = Q / pow(1 + dE / current_energy, 3.0 / 2.0)
    emit = emit / np.sqrt(1 + dE / current_energy)

    # update values
    current_energy += dE
    vx[current_ind] = rx_kick
    vy[current_ind] = ry_kick

    for n in range(current_ind + 1, solve_matrix.shape[0]):
        history_Q.append(Q)
        history_emit.append(emit)

        this_dz = z[n] - z[n - 1]
        # Evaluate term present in both equations
        term = 2 * Q / (ux[n - 1] + uy[n - 1])

        # Evaluate terms for x and y
        term1x = pow(emit, 2) / pow(ux[n - 1], 3) - kappa[n - 1] * ux[n - 1]
        term1y = pow(emit, 2) / pow(uy[n - 1], 3) + kappa[n - 1] * uy[n - 1]

        # Update v_x and v_y first.
        vx[n] = (term + term1x) * this_dz + vx[n - 1]
        vy[n] = (term + term1y) * this_dz + vy[n - 1]

        # Use updated v to update u
        ux[n] = vx[n] * this_dz + ux[n - 1]
        uy[n] = vy[n] * this_dz + uy[n - 1]

    history_Q = np.array(history_Q)
    history_emit = np.array(history_emit)

    if history:
        return history_Q, history_emit
    else:
        return solve_matrix


# ------------------------------------------------------------------------------
#    Optimizer
# Find a solution for the four quadrupole voltages to shape the beam. The final
# coordinates rx,ry, rxp, ryp are to meet the target coordinate to match the
# acceleration lattice.
# ------------------------------------------------------------------------------
class Optimizer(Lattice):
    def __init__(self, initial_conds, scales, target, norms, filenames, parameters):
        super().__init__()
        self.initial_conds = initial_conds
        self.scales = scales
        self.target = target
        self.cost_norms = norms
        self.filenames = filenames
        self.parameters = parameters
        self.z = None
        self.grad = None
        self.optimize_matching = False
        self.optimize_acceleration = False
        self.sol = None
        self.optimum = None
        self.bounds = None
        self.cost_hist = []

    def calc_cost(self, data, target, norm):
        """Calculate cost function

        The cost here is the mean-squared-error (MSE) which takes two vectors.
        The data vector containing the coordinates extracted from simulation and
        the target variables we are seeking. The norm is used to
        normalize the coordinate and angle vectors so that they are of similar
        magnitude.
        """
        cost = pow((data - target) * norm, 2)
        return np.sum(cost)

    def func_to_optimize_matching(self, V_scales):
        """Single input function to min/maximize

        Most optimizers take in a function with a single input that are the
        parameters to optimize for. The voltage scales are used here that
        scale the focusing strength. The gradient is then created from the
        lattice class and the KV-envelope equation solved for.
        The final coordinates are then extracted and the MSE is computed for
        the cost function.
        A cost function of zero would be mean that for the given input parameters
        (initial conditions, Q, emittance) the optimizer found the required
        voltage settings on the quadrupole to shape the envelop into the final
        conditions."""

        # Instantiate lattice and unpack/calculate parameters
        self.user_input_match(self.filenames, self.parameters["Nq"], scales=V_scales)
        z, gradz = self.z, self.grad
        emit, Q = self.parameters["emit"], self.parameters["Q"]
        E_s = self.parameters["E"]
        dz = z[1] - z[0]
        kappa = wp.echarge * gradz / 2.0 / E_s / wp.jperev

        # Solve KV equations
        soln_matrix = np.zeros(shape=(len(z), 4))
        soln_matrix[0, :] = self.initial_conds
        solver(soln_matrix, z, kappa, emit, Q)

        # Store solution
        self.sol = soln_matrix[-1, :]

        # Compute cost and save to history
        cost = self.calc_cost(self.sol, self.target, self.cost_norms)
        self.cost_hist.append(cost)

        return cost

    def func_to_optimize_acceleration(self, coordinates):
        """Single input function to min/maximize

        Most optimizers take in a function with a single input that is the
        parameters to optimize for. Here, the desired coordinates are the input
        values compared to the voltage scales for the matching section.
        A 0 cost function here would be mean that for fixed voltages, and
        starting parameters, the initial coordinates and final coordinates are
        equal creating a matched condition. In other words, the initial
        positions and angles were found such that the final position and angle are
        equal to initial."""

        # Unpack/calculate parameters
        emit = self.parameters["emit"]
        Q = self.parameters["Q"]
        gap_centers = self.parameters["gap centers"]
        Lp = self.parameters["Lp"]
        Vg = self.parameters["Vg"]
        phi_s = self.parameters["phi_s"]
        E_s = self.parameters["E"]

        # Instantiate lattice and calculate the rest of the parameters
        z, grad = self.z, self.grad
        dz = z[1] - z[0]
        kappa = wp.echarge * grad / 2.0 / E_s / wp.jperev

        # Solve KV equations
        soln_matrix = np.zeros(shape=(len(z), 4))
        soln_matrix[0, :] = coordinates
        solver_with_accel(soln_matrix, z, kappa, emit, Q, gap_centers, Vg, phi_s, E_s)

        # Store solution
        self.sol = soln_matrix[-1, :]

        # Compute cost and save to history
        cost = self.calc_cost(self.sol, coordinates, self.cost_norms)
        self.cost_hist.append(cost)

        return cost

    def minimize_cost(self, function, max_iter=200):
        """Function that will run optimizer and output results

        This function contains the actual optimizer that will be used. Currently
        it is a prepackaged optimizer from the Scipy library. There are numerous
        options for the optimizer and this function can be modified to include
        options in the arguments."""

        if self.optimize_matching:
            print("--Optimizing for the matching section")
            res = sciopt.minimize(
                function,
                self.scales,
                method="nelder-mead",
                options={"xatol": 1e-8, "maxiter": max_iter, "disp": True},
                bounds=self.bounds,
            )
            self.optimum = res

            return print("--Optimization completed")

        if self.optimize_acceleration:
            print("--Optimizing for the acceleration lattice")
            res = sciopt.minimize(
                function,
                self.initial_conds,
                method="nelder-mead",
                options={"xatol": 1e-8, "maxiter": max_iter, "disp": True},
                bounds=self.bounds,
            )
            self.optimum = res

            return print("--Optimization completed")

        return print(
            "Neither matching section or acceleration weere chosen to optimize"
        )
