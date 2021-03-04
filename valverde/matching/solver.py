"""Script for solivng KV-envelope equation."""

import numpy as np
import matplotlib.pyplot as plt
import pdb

import warp as wp

import parameters

# Define useful variables
mm = wp.mm
kV = wp.kV


def hard_edge_kappa(
    voltage,
    pos_array,
    drift=9.3 * mm,
    gap=2 * mm,
    length_quad=0.695 * mm,
    N=500,
    injection_energy=8 * kV,
    rp=0.55 * mm,
):
    """Function to populate kappa(s) with kappa assuming hard edge model.

    The function locates ESQs in an array and sets values to +/- kappa. This
    is to approximate an ESQ doublet by hard edge. Thus, the entries in the
    kappa_array will be in those positions where the ESQ lives according to
    the position array. The default value drift is the distance between the RF
    stacks (center to center). This value is computed from the 1D simulation
    script in the MEQALAC codes for Argon by computing beta*lambda/2.
    This value is computed for given frequencies, energies, and voltages and
    should be changed accordingly if necessary. For convenience, the values
    are listed here:
    - Frequency : 14.86 MHz
    - KE : 7 KeV
    - Real & Design Energy : 7 KeV
    - Real & Design Voltage : 7 KeV
    - Species : Argon single charge

    The geometric setup is in accordancee with the schematic (see Nick Valverde
    for image). One lattice period contains three RF-stacks where the drift
    between the first stacks are not field free, and the second drift is field
    free and thus can have ESQs.

    Parameters
    ----------
    voltage : list
        Value of ESQ voltages form which kappa will be computed.

    pos_array : ndarray
        One dimensional array giving the discretized space values that the KV
        enevlope equation is to be solved one.

    drift : float
        Distance between RF stacks (center to center).

    gap : float
        Distance of acceleration gap.

    length_quad : float
        Physical length of the quadrupole.

    N : int
        Number of simulation points.

    Returns
    -------
    kappa_array : ndarray
        The array representing two ESQs via a hard edge model. This array will
        be of same size as the position array.

    """

    # Rename parameters for tersity
    d = drift
    g = gap
    lq = length_quad
    L = 2 * d  # total length of lattice
    space = (d - g - 2 * lq) / 3  # Spacing between ESQs

    # Identify interval for +/- ESQ according to position. The first chunk
    # represents the drift region between two stacks and is not field free. The
    # second chunk is the field free region and has the ESQs.
    chnk1 = d
    esq1_lowbnd = chnk1 + g / 2 + space
    esq1_highbnd = esq1_lowbnd + lq

    esq2_lowbnd = esq1_highbnd + space
    esq2_highbnd = esq2_lowbnd + lq

    # Find indices that correspond to the ESQs
    splus = np.where((pos_array >= esq1_lowbnd) & (pos_array <= esq1_highbnd))[0]
    sminus = np.where((pos_array >= esq2_lowbnd) & (pos_array <= esq2_highbnd))[0]

    # Create kappa array and assign values
    kappa_array = np.zeros(len(pos_array))
    voltage_array = np.zeros(len(pos_array))

    # Designate voltage setting and compute focusing strength for each ESQ
    V1 = voltage[0]
    V2 = voltage[1]
    voltage_array[splus], voltage_array[sminus] = V1, V2

    k1 = V1 / injection_energy / rp / rp
    k2 = V2 / injection_energy / rp / rp
    kappa_array[splus] = k1
    kappa_array[sminus] = k2

    return kappa_array, voltage_array


def solve_KV():
    """Differential equation solver for KV equation

    Function will use Euler-Cromer method to solve the KV-envelope equation.
    Denoting derivative by primes, the coupled equation is

    r_x'' + k(s)r_x - 2Q/(r_x + r_y) - emittance_x^2/r_x^3 = 0
    r_y'' + k(s)r_y - 2Q/(r_x + r_y) - emittance_y^2/r_y^3 = 0

    The method transforms the two second-order equations into four first-order
    equations by defining u_x = r_x, v_x = u'_x and similarly for r_y. The
    input parameter is assumed to be an Nx4 matrix where the columns reprsent,
    in order: u_x, u_y, v_x, v_y. These arrays are then updated and the soln
    soln matrix is returned. The initial conditions should be initialized in the
    matrix calling the function. Else, the ICs will be zero.

    Parameters
    ----------
    soln_matrix : ndarray
        An Nx4 matrix where N is the number of steps to take in the solver
        and the four columns are u_x, u_y, v_x, v_y respectively.

    Returns
    -------
    kv_soln : ndarray
        The Nx4 matrix where the columns are now the solutions to the KV-envelope
        equation.

    """

    # Use global variables
    global Q, karray, ds, emittance, soln_marix

    # Identify arrays
    ux = soln_matrix[:, 0]
    uy = soln_matrix[:, 1]
    vx = soln_matrix[:, 2]
    vy = soln_matrix[:, 3]

    # Main loop to update equation. Loop through matrix and update entries.
    for n in range(1, len(soln_matrix)):
        # Evaluate term present in both equations
        term = 2 * Q / (ux[n - 1] + uy[n - 1])

        # Evaluate terms for x and y
        term1x = pow(emittance, 2) / pow(ux[n - 1], 3) - karray[n - 1] * ux[n - 1]
        term1y = pow(emittance, 2) / pow(uy[n - 1], 3) + karray[n - 1] * uy[n - 1]

        # Update v_x and v_y first.
        vx[n] = (term + term1x) * ds + vx[n - 1]
        vy[n] = (term + term1y) * ds + vy[n - 1]

        # Use updated v to update u
        ux[n] = vx[n] * ds + ux[n - 1]
        uy[n] = vy[n] * ds + uy[n - 1]

    return True


class matching_section:
    def __init__(
        self, drift=0.5 * mm, gap=2 * mm, inj_energy=8 * kV, rp=0.55 * mm, N_esq=6
    ):
        self.drift = drift
        self.gap = gap
        self.inj_energy = inj_energy
        self.rp = rp
        self.N_esq = N_esq

    def calc_kappa(self, voltage):
        kappa = voltage / self.inj_energy / self.rp / self.rp

        return kappa

    def create_mesh(self, prec=0.01 * mm):
        # Calculate length of matching section
        section = self.N_esq * self.gap + self.drift * (self.N_esq - 1)

        # Add some padding on the endpoints. Calculate total length of mesh
        # with padding and use the precision paramter to create array.
        pad = 0.35 * section
        length = 2 * pad + section
        mesh = np.linspace(-length, length, int(length / prec))

        return mesh

    def create_section(self, voltages=[500, -500, 200, -100], cent=0):
        """Create array of focusing strengths.

        Function will use the created mesh from the create_mesh function and
        create a dimensionally identical array holding the the values of the
        ESQ focusing strengths."""

        # Initialize simulation mesh and kappa array
        mesh = self.create_mesh()
        kappa_array = np.zeros(len(mesh))

        # Loop through voltage list and evaluate kappa values. For each voltage
        # assign values in kappa array with direct correspondance to mesh. The
        # centers can be given by the user or calculated from the first center.
        # The try and except block handles the two cases.

        try:
            for this_cent, this_volt in zip(cent, voltages):
                this_kappa = self.calc_kappa(this_volt)
                # Find indices for ESQ
                upr_bnd = this_cent + self.gap / 2
                lwr_bnd = this_cent - self.gap / 2
                indices = np.where((mesh >= lwr_bnd) & (mesh <= upr_bnd))[0]

                # Assign kappa values to array
                kappa_array[indices] = this_kappa

        except TypeError:
            # Will loop through voltages and calculate new centers on the fly
            # Calcaulte separation (center-center) of ESQs
            separation = self.gap + self.drift
            for i, this_volt in enumerate(voltages):
                this_kappa = self.calc_kappa(this_volt)
                this_cent = cent + (i * separation)
                # Find indices for ESQ
                upr_bnd = this_cent + self.gap / 2
                lwr_bnd = this_cent - self.gap / 2
                indices = np.where((mesh >= lwr_bnd) & (mesh <= upr_bnd))[0]

                # Assign kappa values to array
                kappa_array[indices] = this_kappa

        return mesh, kappa_array


if __name__ == "__main__":
    # Read in parameters frome paramters script.
    param_dict = parameters.main()

    # Set system parameters/variables
    Q = param_dict["Q"]
    volt = [3 * kV, -3 * kV]
    emittance = param_dict["emittance"]
    ux_initial = param_dict["inj_radius"]
    uy_initial = param_dict["inj_radius"]
    vx_initial = param_dict["inj_xprime"]
    vy_initial = param_dict["inj_yprime"]

    # Set up solver paramters
    d = 9.3 * mm
    g = 2 * mm
    lq = 0.695 * mm
    N = 501
    L = 2 * d
    ds = L / N

    # Construct arrays for differential equation. Columns ux, uy, vx, vy
    soln_matrix = np.zeros(shape=(N, 4))
    s = np.array([ds * i for i in range(N)])

    # Enter initial condition into array
    soln_matrix[0, :] = ux_initial, uy_initial, vx_initial, vy_initial

    # Create kappa array
    # pdb.set_trace()
    karray, Varray = hard_edge_kappa(volt, s)
    # Call solver
    solve_KV()

    # --Plotting outputs
    # Grab x,x',y,and y'
    x = soln_matrix[:, 0]
    y = soln_matrix[:, 1]
    xprime = soln_matrix[:, 2]
    yprime = soln_matrix[:, 3]

    # Visualize kappa array to verify correct geometry. Identify gaps with black
    # dashed lines and fill ESQs with blue(+ bias) and red (- bias)
    fig, ax = plt.subplots()
    ax.plot(s / mm, Varray / kV)
    ax.fill_between(s[Varray > 0] / mm, max(Varray), y2=0, alpha=0.2, color="b")
    ax.fill_between(s[Varray < 0] / mm, min(Varray), y2=0, alpha=0.2, color="r")
    plates = np.array([g / 2, d - g / 2, d + g / 2, 2 * d - g / 2])
    for pos in plates:
        ax.axvline(x=pos / mm, c="k", ls="--", lw=2)
    ax.set_xlabel("s [mm]")
    ax.set_ylabel(r"$V$ [kV]")
    ax.set_title("Schematic of Simulation Geometry")
    plt.show()

    # Create plots for solution to KV equtions and overlay ESQ and gap positions
    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(s / mm, x / mm, c="k")
    ax[0].set_ylabel(r"$r_x(s)$ [mm]")
    ax[0].fill_between(
        s[Varray > 0] / mm, max(x) / mm, y2=min(x) / mm, alpha=0.2, color="b"
    )
    ax[0].fill_between(
        s[Varray < 0] / mm, max(x) / mm, y2=min(x) / mm, alpha=0.2, color="r"
    )
    for pos in plates:
        ax[0].axvline(x=pos / mm, c="k", ls="--", lw=2)

    ax[1].plot(s / mm, xprime, c="k")
    ax[1].set_ylabel(r"$r_x'(s)$")
    ax[1].set_xlabel(r"$s$ [mm]")
    ax[1].fill_between(
        s[Varray > 0] / mm, max(xprime), y2=min(xprime), alpha=0.2, color="b"
    )
    ax[1].fill_between(
        s[Varray < 0] / mm, max(xprime), y2=min(xprime), alpha=0.2, color="r"
    )
    for pos in plates:
        ax[1].axvline(x=pos / mm, c="k", ls="--", lw=2)
    plt.show()

    fig, ax = plt.subplots(nrows=2, sharex=True)
    ax[0].plot(s / mm, y / mm, c="k")
    ax[0].set_ylabel(r"$r_y(s)$")
    ax[0].fill_between(
        s[Varray > 0] / mm, max(y) / mm, y2=min(y) / mm, alpha=0.2, color="b"
    )
    ax[0].fill_between(
        s[Varray < 0] / mm, max(y) / mm, y2=min(y) / mm, alpha=0.2, color="r"
    )
    for pos in plates:
        ax[0].axvline(x=pos / mm, c="k", ls="--", lw=2)

    ax[1].plot(s / mm, yprime, c="k")
    ax[1].set_xlabel(r"$s$ [mm]")
    ax[1].fill_between(
        s[Varray > 0] / mm, max(yprime), y2=min(yprime), alpha=0.2, color="b"
    )
    ax[1].fill_between(
        s[Varray < 0] / mm, max(yprime), y2=min(yprime), alpha=0.2, color="r"
    )
    for pos in plates:
        ax[1].axvline(x=pos / mm, c="k", ls="--", lw=2)

    ax[1].set_ylabel(r"$r_y'(s)$")
    plt.show()
