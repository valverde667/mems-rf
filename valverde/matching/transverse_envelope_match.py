# Solve KV-envelope equations with given initial conditions and different
# lattice creations.

import numpy as np
import scipy.optimize as sciopt
import itertools
import scipy.constants as SC
import matplotlib.pyplot as plt
import matplotlib as mpl
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

# ------------------------------------------------------------------------------
#    Useful constants and Parameter initialization
# Define useful constants such as units. Initialize various paramters for the
# system/simulation.
# ------------------------------------------------------------------------------
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

# System and Geometry settings
Vg = 7 * kV
Ng = 4
phi_s = np.ones(Ng) * 0
f = 13.6 * MHz
lq = 0.695 * mm
lq_eff = 1.306 * mm
d = 3.0 * mm
g = 2 * mm
Vq = 0.6 * kV
Nq = 4
rp = 0.55 * mm
rsource = 0.25 * mm
G_hardedge = 2 * Vq / pow(rp, 2)
res = 10 * um

# Beam Settings
Q = 6.986e-5
emit = 1.336 * mm * mrad
E_s = 7 * keV
init_I = 10 * uA
div_angle = 3.78 * mrad
Tb = 0.1  # eV

# Beam specifications
beam = wp.Species(type=wp.Argon, charge_state=1)
mass_eV = beam.mass * pow(SC.c, 2) / wp.jperev
beam.ekin = E_s
beam.ibeam = init_I
beam.a0 = rsource
beam.b0 = rsource
beam.ap0 = 0.0
beam.bp0 = 0.0
beam.ibeam = init_I
beam.vbeam = 0.0
beam.ekin = E_s
vth = np.sqrt(Tb * wp.jperev / beam.mass)
wp.derivqty()


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


# ------------------------------------------------------------------------------
#    Lattice Setup and Solver
# The lattice is generated that will be used to solve the KV-envelope equations.
# The two options are available for setting up the lattice:
#   1) The user-input options will use the extracted gradient for the desired
#   matching section simulated in a separate script. Both the gradient and the
#   simulated mesh is extracted.
#   2) If the user-input option is set to False then the hard edge model is used.
#   The hard edge model will place the ESQs using the physical length of the ESQ
#   but the effective length will be used to place the gradient on the mesh.
# Once the lattice is created all the data is stored in the Lattice class and then
# used for the solver. Lastly, the gradients are scaled by to simulate different
# voltage settings.
# ------------------------------------------------------------------------------
gap_centers = calc_gap_centers(E_s, mass_eV, phi_s, f, Vg)
Lp = gap_centers[2] - g / 2
user_input = True
if user_input:
    # Instantiate the class and use the extracted fields to create the mesh.
    lattice = Lattice()
    file_names = ("iso_zgrad.npy", "iso_esq_grad.npy")
    scales = []
    # Scale the voltages and make the scales focus-defocus-focus-defocus
    for i in range(Nq):
        if i % 2 == 0:
            scales.append(-1.0)
        else:
            scales.append(1.0)

    scales = np.array(scales)
    scales *= (0.15, 0.2, 0.10, 0.20)
    scales = np.array([0.46240597, -1.09299698, 0.69792102, 0.19377574])

    lattice.user_input_match(file_names, Nq, scales=scales)
    z, gradz = lattice.z, lattice.grad

else:
    scales = []
    for i in range(Nq):
        if i % 2 == 0:
            scales.append(-1.0)
        else:
            scales.append(1.0)

    scales = np.array(scales)
    scales *= (0.15, 0.2, 0.10, 0.20)

    lattice = Lattice()
    # Create the hard edge equivalent. The max gradient is hardcoded and
    # corresponds to the gradient in a quadrupole with Vq=400 V.
    lattice.hard_edge_match(
        lq, lq_eff, d, Vq, Nq, rp, scales, max_grad=2566538624.836261
    )
    z, gradz = lattice.z, lattice.grad

# Solve KV equations
dz = z[1] - z[0]
kappa = wp.echarge * gradz / 2.0 / E_s / wp.jperev
ux_initial, uy_initial = rsource, rsource
vx_initial, vy_initial = div_angle, div_angle

soln_matrix = np.zeros(shape=(len(z), 4))
soln_matrix[0, :] = ux_initial, uy_initial, vx_initial, vy_initial

solver(soln_matrix, dz, kappa, emit, Q)
# solver_with_accel(soln_matrix, dz, kappa, emit, Q, z, gap_centers, Vg=1*kV)
solutions = soln_matrix[-1, :]

ux, uy = soln_matrix[:, 0], soln_matrix[:, 1]
vx, vy = soln_matrix[:, 2], soln_matrix[:, 3]

uxf, uyf = ux[-1], uy[-1]
vxf, vyf = vx[-1], vy[-1]

data = np.vstack((ux, uy, vx, vy, z))
if user_input:
    np.save("matching_solver_data", data)
else:
    np.save("matching_solver_data_hardedge", data)
    np.save("kappa_he", kappa)


# ------------------------------------------------------------------------------
#    Optimizer
# Find a solution for the four quadrupole voltages to shape the beam. The final
# coordinates rx,ry, rxp, ryp are to meet the target coordinate to match the
# acceleration lattice.
# ------------------------------------------------------------------------------
class Optimizer(Lattice):
    def __init__(self, initial_conds, guess, target, norms, filenames, parameters):
        super().__init__()
        self.Nq = Nq
        self.initial_conds = initial_conds
        self.guess = guess
        self.target = target
        self.cost_norms = norms
        self.filenames = filenames
        self.parameters = parameters
        self.sol = None
        self.optimum = None
        self.cost_hist = []

    def calc_cost(self, data, target, norm):
        """Calculate cost function

        The cost here is the mean-squared-error (MSE) which takes two vectors.
        The data vector containing the coordinates extracted from simulation and
        the target variables are we are seeking. The scales are used to
        normalize the coordinate and angle vectors so that they are of similar
        scale.
        """
        cost = pow((data - target) * norm, 2)
        return np.sum(cost)

    def func_to_optimize(self, V_scales):
        """Single input function to min/maximize

        Most optimizers take in a function with a single input that is the
        parameters to optimize for. The voltage scales are used here that
        scale the focusing strength. The gradient is then created from the
        lattice class and the KV-envelope equation solved for.
        The final coordinates are then extracted and the MSE is computed for
        the cost function."""

        # Solve KV equations for lattice design and input Voltage scales
        self.user_input_match(self.filenames, self.Nq, scales=V_scales)
        z, gradz = self.z, self.grad
        emit, Q = self.parameters["emit"], self.parameters["Q"]

        # Solve KV equations
        dz = z[1] - z[0]
        kappa = wp.echarge * gradz / 2.0 / E_s / wp.jperev
        soln_matrix = np.zeros(shape=(len(z), 4))
        soln_matrix[0, :] = self.initial_conds
        solver(soln_matrix, dz, kappa, emit, Q)

        # Store solution
        self.sol = soln_matrix[-1, :]

        # Compute cost and save to history
        cost = self.calc_cost(self.sol, self.target, self.cost_norms)
        self.cost_hist.append(cost)

        return cost

    def func_to_optimize_accel(self, V_scales):
        """Single input function to min/maximize

        Most optimizers take in a function with a single input that is the
        parameters to optimize for. The voltage scales are used here that
        scale the focusing strength. The gradient is then created from the
        lattice class and the KV-envelope equation solved for.
        The final coordinates are then extracted and the MSE is computed for
        the cost function."""

        # unpack the acceleration  paramters
        emit = self.parameters["emit"]
        Q = self.parameters["Q"]
        gap_centers = self.parameters["gap centers"]
        Lp = self.parameters["Lp"]
        Vg = self.parameters["Vg"]
        phi_s = self.parameters["phi_s"]
        E = self.parameters["E"]

        # Solve KV equations for lattice design and input Voltage scales
        self.accel_lattice(gap_centers, self.filenames, V_scales, Lp)
        z, gradz = self.z, self.grad

        # Solve KV equations
        dz = z[1] - z[0]
        kappa = wp.echarge * gradz / 2.0 / E_s / wp.jperev
        soln_matrix = np.zeros(shape=(len(z), 4))
        soln_matrix[0, :] = self.initial_conds
        solver_with_accel(soln_matrix, dz, kappa, emit, Q, z, gap_centers, Vg, phi_s, E)

        # Store solution
        self.sol = soln_matrix[-1, :]

        # Compute cost and save to history
        cost = self.calc_cost(self.sol, self.target, self.cost_norms)
        self.cost_hist.append(cost)

        return cost

    def minimize_cost(self, function, max_iter=200):
        """Function that will run optimizer and output results

        This function contains the actual optimizer that will be used. Currently
        it is a prepackaged optimizer from the Scipy library. There are numerous
        options for the optimizer and this function can be modified to include
        options in the arguments."""

        res = sciopt.minimize(
            function,
            self.guess,
            method="nelder-mead",
            options={"xatol": 1e-8, "maxiter": max_iter, "disp": True},
        )
        self.optimum = res


find_voltages = True
if find_voltages:
    x0 = np.array([rsource, rsource, div_angle, div_angle])
    guess = scales
    guess_accel = [0.2, 0.2]
    target = np.array([0.15 * mm, 0.28 * mm, 0.847 * mrad, -11.146 * mrad])
    rp_norm = 27 * mrad
    norms = np.array([1 / rp, 1 / rp, 1 / rp_norm, 1 / rp_norm])
    parameters = {
        "emit": emit,
        "Q": Q,
        "gap centers": gap_centers,
        "Vg": Vg,
        "Lp": Lp,
        "phi_s": phi_s,
        "E": E_s,
    }
    opt = Optimizer(x0, guess, target, norms, file_names, parameters)
    opt.minimize_cost(opt.func_to_optimize, max_iter=300)
    errors = abs((target - solutions) / target)
    print(f"Fraction Errors: {errors}")

# ------------------------------------------------------------------------------
#    Plot and Save
# Plot various quanities and save the data.
# ------------------------------------------------------------------------------
k0 = 2 * 0.5 * kV
k0 = k0 / 7 / kV / pow(rp, 2)
Fsc = 2 * Q / (ux + uy)
Femitx = pow(emit, 2) / pow(ux, 3)
Femity = pow(emit, 2) / pow(uy, 3)
kappa_he = np.load("kappa_he.npy")
data_he = np.load("matching_solver_data_hardedge.npy")

fig, ax = plt.subplots()
ax.set_xlabel("z (mm)")
ax.set_ylabel(r"$\kappa (z)/\hat{\kappa}$")
plt.plot(z / mm, kappa / k0)
ax.axhline(y=0, c="k", lw=0.5)
plt.show()

fig, ax = plt.subplots()
ax.set_title(r"Envelope Solutions for $r_x$ and $r_y$")
ax.set_xlabel("z (mm)")
ax.set_ylabel("Transverse Position (mm)")
ax.plot(z / mm, ux / mm, label=r"$r_x(s)$", c="k")
ax.plot(z / mm, uy / mm, label=r"$r_y(s)$", c="b")
ax.scatter(z[-1] / mm, target[0] / mm, marker="*", c="k", s=90, alpha=0.6)
ax.scatter(z[-1] / mm, target[1] / mm, marker="*", c="b", s=90, alpha=0.6)
ax.legend()

fig, ax = plt.subplots()
ax.set_title(r"Envelope Solutions for $rp_x$ and $rp_y$")
ax.set_xlabel("z (mm)")
ax.set_ylabel("Transverse Angle (mrad)")
ax.plot(z / mm, vx / mm, label=r"$rp_x(s)$", c="k")
ax.plot(z / mm, vy / mm, label=r"$rp_y(s)$", c="b")
ax.scatter(z[-1] / mm, target[2] / mrad, marker="*", c="k", s=90, alpha=0.6)
ax.scatter(z[-1] / mm, target[3] / mrad, marker="*", c="b", s=90, alpha=0.6)
ax.legend()

fig, ax = plt.subplots()
plt.plot(data[-1, :] / mm, kappa / k0, c="b", label="Extracted")
plt.plot(data_he[-1, :] / mm, kappa_he / k0, c="g", alpha=0.5, label="Hard-Edge")
ax.axhline(y=0, c="k", lw=0.5)
ax.set_xlabel("z (mm)")
ax.set_ylabel(r"$\kappa(z)/\hat{\kappa}$")
plt.savefig("/Users/nickvalverde/Desktop/kappa_lattice.svg")
ax.legend()

fig, ax = plt.subplots()
ax.plot(z / mm, 2 * Q / (ux + uy), c="r", label=r"$F_\mathrm{SC}$")
ax.plot(z / mm, pow(emit, 2) / pow(ux, 3), c="k", label=r"$F_\mathrm{emit-x}$")
ax.plot(z / mm, pow(emit, 2) / pow(uy, 3), c="b", label=r"$F_\mathrm{emit-y}$")
ax.set_ylabel("Defocusing Term Strength (1/m)")
ax.legend()

Fig, ax = plt.subplots()
ax.plot(z / mm, Fsc / Femitx, c="k", label=r"$F_\mathrm{sc} / F_\mathrm{x-emit}$")
ax.plot(z / mm, Fsc / Femity, c="b", label=r"$F_\mathrm{sc} / F_\mathrm{y-emit}$")
ax.set_xlabel("z (mm)")
ax.set_ylabel("Ratio of Defocusing Terms")
ax.legend()


plt.show()


print(f"Final (rx, ry) mm: {uxf/mm:.4f}, {uyf/mm:.4f}")
print(f"Final (rpx, rpy) mrad: {vxf/mrad:.4f}, {vyf/mrad:.4f}")

# ------------------------------------------------------------------------------
#    Acceleration Treatment
# Additional section to do acceleration matching
# ------------------------------------------------------------------------------
accel_fnames = ("accel_zmesh.npy", "accel_esq_grad.npy")
accel_scales = np.array([0.25, 0.25])
accel_scales = np.array([-0.08961465, -0.12368815])
lattice_with_accel = Lattice()
lattice_with_accel.accel_lattice(gap_centers, accel_fnames, accel_scales, Lp)
accel_z, accel_grad = lattice_with_accel.z, lattice_with_accel.grad
accel_dz = accel_z[1] - accel_z[0]
accel_kappa = wp.echarge * accel_grad / 2.0 / E_s / wp.jperev

fig, ax = plt.subplots()
ax.set_title(r"Initial (Non-optimized) $\kappa(z)$ ")
ax.set_xlabel("z (mm)")
ax.set_ylabel(r"$\kappa (z)/\hat{\kappa}$")
plt.plot(accel_z / mm, accel_kappa / k0)
ax.axhline(y=0, c="k", lw=0.5)
plt.show()

# Plot the kv-solver
accel_x0 = np.array([0.27954 * mm, 0.34075 * mm, 1.8497 * mrad, -6.8141 * mrad])
accel_target = accel_x0.copy()

ux_initial, uy_initial = accel_x0[:2]
vx_initial, vy_initial = accel_x0[2:]

accel_soln_matrix = np.zeros(shape=(len(accel_z), 4))
accel_soln_matrix[0, :] = ux_initial, uy_initial, vx_initial, vy_initial

solver_with_accel(
    accel_soln_matrix,
    accel_dz,
    accel_kappa,
    emit,
    Q,
    accel_z,
    gap_centers,
    Vg=Vg,
    phi_s=phi_s,
    E=E_s,
)
# solver_with_accel(soln_matrix, dz, kappa, emit, Q, z, gap_centers, Vg=1*kV)
accel_solutions = accel_soln_matrix[-1, :]

ux, uy = accel_soln_matrix[:, 0], accel_soln_matrix[:, 1]
vx, vy = accel_soln_matrix[:, 2], accel_soln_matrix[:, 3]

uxf, uyf = ux[-1], uy[-1]
vxf, vyf = vx[-1], vy[-1]


fig, ax = plt.subplots()
ax.set_title(r"Envelope Solutions for $r_x$ and $r_y$ with accel.")
ax.set_xlabel("z (mm)")
ax.set_ylabel("Transverse Position (mm)")
ax.plot(accel_z / mm, ux / mm, label=r"$r_x(s)$", c="k")
ax.plot(accel_z / mm, uy / mm, label=r"$r_y(s)$", c="b")
ax.scatter(accel_z[-1] / mm, accel_target[0] / mm, marker="*", c="k", s=90, alpha=0.6)
ax.scatter(accel_z[-1] / mm, accel_target[1] / mm, marker="*", c="b", s=90, alpha=0.6)
ax.legend()

fig, ax = plt.subplots()
ax.set_title(r"Envelope Solutions for $rp_x$ and $rp_y$ with accel.")
ax.set_xlabel("z (mm)")
ax.set_ylabel("Transverse Angle (mrad)")
ax.plot(accel_z / mm, vx / mm, label=r"$rp_x(s)$", c="k")
ax.plot(accel_z / mm, vy / mm, label=r"$rp_y(s)$", c="b")
ax.scatter(accel_z[-1] / mm, accel_target[2] / mrad, marker="*", c="k", s=90, alpha=0.6)
ax.scatter(accel_z[-1] / mm, accel_target[3] / mrad, marker="*", c="b", s=90, alpha=0.6)
ax.legend()

# Use coordinate vector to match from mathematica script
guess_accel = accel_scales
rp_norm = 15 * mrad
norms = np.array([1 / rp, 1 / rp, 1 / rp_norm, 1 / rp_norm])
parameters = {
    "emit": emit,
    "Q": Q,
    "gap centers": gap_centers,
    "Vg": Vg,
    "Lp": Lp,
    "phi_s": phi_s,
    "E": E_s,
}
opt = Optimizer(accel_x0, guess_accel, accel_target, norms, accel_fnames, parameters)
opt.minimize_cost(opt.func_to_optimize_accel, max_iter=300)
errors = abs((target - solutions) / target)
print(f"Fraction Errors: {errors}")
