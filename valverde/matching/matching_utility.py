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


def calc_energy_gain(Vg, phi_s):
    return Vg * np.cos(phi_s)


def calc_Q_change(Q_prev, delta_E, Ebeam):
    """Calculate Q after acceleration kick

    When the beam is accelerated by a thin-lens kick, the generalized perveance
    changes as well. The perveance before the acceleration changes by a scaling
    factor related to the change in energy and the energy of the beam before the
    acceleration kick.
    """

    denom = pow(1 + delta_E / Ebeam, 1.5)
    return Q_prev / denom


def calc_emit_change(emit_prev, delta_E, Ebeam):
    """Calculate emmitance after acceleration kick

    When the beam is accelerated by a thin-lens kick, the rms-edge emittance
    changes as well. The emittance before the acceleration changes by a scaling
    factor related to the change in energy and the energy of the beam before the
    acceleration kick.
    """

    denom = np.sqrt(1 + delta_E / Ebeam)
    return emit_prev / denom


def calc_angle_change(rp_prev, delta_E, Ebeam):
    """Calculate angle after acceleration kick

    Calcluate the new angle based on the previous angle before the acceleration
    and the acceleration characteristics.
    """

    denom = 1.0 + np.sqrt(delta_E / Ebeam)
    return rp_prev / denom


class Lattice:
    def __init__(self):
        self.zmin = 0.0
        self.zmax = None
        self.z = None
        self.dz = None

        self.Ng = None
        self.gap_centers = None
        self.gap_field_data = None
        self.dz_gap = None

        self.Nq = None
        self.quad_centers = None
        self.quad_field_data = None
        self.dz_quad = None

        self.kappa_quad = None
        self.kappa_gap = None

        self.beam_energy = None
        self.dbeam_energy = None
        self.ddbeam_energy = None

    def build_lattice(self, zstart, zend, scheme, quad_info, gap_info, res):
        """Function to build lattice given the scheme and length Lp

        This function will build the 1D mesh to be used for integration. The scheme
        is given by a string with q for quad and g for gap. The goal is to return
        three arrays. One will be the zmesh for the lattice, another will be the quadrupole
        data, and lastly the gap data. All arrays should have the same size.
        The quad and gap data will undoubtedly have different sizes and presents a challenge
        to build three commensurate arrays.
        To do this, an overall mesh if first created with from zstart to zend with
        the given res on the mesh for dz. Then, the quad and gap info are used to find
        the centers and extents of the conductor objects within the z-array. The z-data
        for each conductor is then stitched into the zmesh while simultaneously
        stitching in the field data in the commensurate field arrays. The scheme is
        used to determine when zeros need to be added to the other conductor array
        in order to maintain sizing.

        Parameters
        ----------
        zstart: float
            Start of z array.

        zend: float
            End of z array.

        scheme: string
            Contains the scheme of the lattice. For example, to have an acceleration
            gap, quad, quad, gap would be 'g-q-q-g'. The separator '-' is needed. Order
            does not matter but for the sake of clarity should be preserved in asecending
            z-center in the lattice.

        quad_info: tuple containing lists
            A tuple of information containing:
                ([z-centers], [zextents of extracted field], [array of field data]).
            The field data is centered on the z-center and extends for [zc-zextent/2, zc+zextent/2].
            The inputs in the tuple have to be lists even if it is one element.
            If no quad data is to be provided, input None.

        gap_info: tuple
            A tuple of information for the acceleration gap. See quad_info parameter.

        res: float
            Overall resolution to use. Most of the array might be empty space and this
            parameter can be used to lower computational cost for the zero entries.
        """
        Lp = zend - zstart
        Nz = int(Lp / res)
        z = np.linspace(zstart, zend, Nz)

        # Intialize counters to correctly index data
        q_counter = 0
        g_counter = 0

        # Initial field arrays where the data will be stitched in to.
        quad_data = np.zeros(len(z))
        gap_data = np.zeros(len(z))

        # Separate the strings in the conductors. Loop through the elements and then
        # build arrays based on the scheme.
        conductors = scheme.split("-")

        quad_centers = []
        gap_centers = []

        for i, cond in enumerate(conductors):
            if cond == "q":
                z_patch = quad_info[1][q_counter]
                this_zc = quad_info[0][q_counter]
                this_zext = z_patch[-1] - z_patch[0]
                this_field = quad_info[2][q_counter]

                # Append some data before incrementation
                quad_centers.append(this_zc)
                self.dz_quad = z_patch[1] - z_patch[0]
                q_counter += 1

                field_loc = np.where(
                    (z > this_zc - this_zext / 2.0) & (z < this_zc + this_zext / 2.0)
                )[0]
                patch_start = field_loc[0]
                patch_end = field_loc[-1]

                z_left = z[:patch_start]
                z_right = z[patch_end:]
                l_qfield = quad_data[:patch_start]
                r_qfield = quad_data[patch_end:]
                l_gfield = gap_data[:patch_start]
                r_gfield = gap_data[patch_end:]

                # Check for overlap between patched area and zmesh. If there is, remove
                # overlap and stitch together the patch.
                left_overlap = np.where((z_patch[0] - z_left) < 0)[0]
                if len(left_overlap) != 0:
                    z_left = np.delete(z_left, left_overlap)
                    l_qfield = np.delete(l_qfield, left_overlap)
                    l_gfield = np.delete(l_gfield, left_overlap)

                right_overlap = np.where((z_right - z_patch[-1]) < 0)[0]
                if len(right_overlap) != 0:
                    z_right = np.delete(z_right, right_overlap)
                    r_qfield = np.delete(r_qfield, right_overlap)
                    r_gfield = np.delete(r_gfield, right_overlap)

                # Stitch fields together
                z_patched = np.concatenate((z_left, z_patch, z_right))
                qpatched = np.concatenate((l_qfield, this_field, r_qfield))
                gpatched = np.concatenate((l_gfield, 0 * this_field, r_gfield))

                # Rename previously defined meshs for continuity
                z = z_patched
                quad_data = qpatched
                gap_data = gpatched

            elif cond == "g":
                z_patch = gap_info[1][g_counter]
                this_zc = gap_info[0][g_counter]
                this_zext = z_patch[-1] - z_patch[0]
                this_field = gap_info[2][g_counter]

                # Append some data before incrementation
                gap_centers.append(this_zc)
                self.dz_gap = z_patch[1] - z_patch[0]
                g_counter += 1

                field_loc = np.where(
                    (z > this_zc - this_zext / 2.0) & (z < this_zc + this_zext / 2.0)
                )[0]
                patch_start = field_loc[0]
                patch_end = field_loc[-1]

                z_left = z[:patch_start]
                z_right = z[patch_end:]
                l_qfield = quad_data[:patch_start]
                r_qfield = quad_data[patch_end:]
                l_gfield = gap_data[:patch_start]
                r_gfield = gap_data[patch_end:]

                # Check for overlap between patched area and zmesh. If there is, remove
                # overlap and stitch together the patch.
                left_overlap = np.where((z_patch[0] - z_left) < 0)[0]
                if len(left_overlap) != 0:
                    z_left = np.delete(z_left, left_overlap)
                    l_qfield = np.delete(l_qfield, left_overlap)
                    l_gfield = np.delete(l_gfield, left_overlap)

                right_overlap = np.where((z_right - z_patch[-1]) < 0)[0]
                if len(right_overlap) != 0:
                    z_right = np.delete(z_right, right_overlap)
                    r_qfield = np.delete(r_qfield, right_overlap)
                    r_gfield = np.delete(r_gfield, right_overlap)

                # Stitch fields together
                z_patched = np.concatenate((z_left, z_patch, z_right))
                qpatched = np.concatenate((l_qfield, 0 * this_field, r_qfield))
                gpatched = np.concatenate((l_gfield, this_field, r_gfield))

                # Rename previously defined meshs for continuity
                z = z_patched
                quad_data = qpatched
                gap_data = gpatched

            else:
                print("No conductor or scheme input incorrect. Check inputs.")

        # Check if quadrupoles and gaps were used. If so, store some information.
        # If not, store 0's and also create an array of zeros.
        if q_counter > 0:
            self.Nq = q_counter
            self.quad_centers = quad_centers
            self.quad_field_data = quad_data

        else:
            self.Nq = 0
            self.quad_centers = 0
            self.quad_field_data = np.zeros(len(z))

        if g_counter > 0:
            self.Ng = g_counter
            self.gap_centers = gap_centers
            self.gap_field_data = gap_data

        else:
            self.Ng = 0
            self.gap_centers = 0
            self.gap_field_data = np.zeros(len(z))

        self.z = z
        self.quad_field_data = quad_data
        self.gap_field_data = gap_data

        return print("Lattice built and data stored.")

    def adv_particle(self, init_E):
        """Advance particle through field with simple forward advance."""

        z = self.z
        dz = self.dz
        Ez = self.gap_field_data
        energy = np.zeros(len(z))
        energy[0] = init_E

        for n in range(1, len(z)):
            this_dz = z[n] - z[n - 1]
            energy[n] = energy[n - 1] + Ez[n - 1] * this_dz

        self.beam_energy = energy

        return print("Beam energy stored")

    def calc_lattice_kappa(self):
        """Calculate the focusing function kappa on the lattice.

        There will be contributions to the kappa function from both the
        quadruople field data (the gradient) and the acceleration gap.
            Quad kappa: qG/2E
            Gap kappa: -0.25 * (E''/Ei) / (1 + (E-Ei)/Ei)

        """
        E = self.beam_energy
        init_E = E[0]

        kq = 0.5 * self.quad_field_data / E

        # Compute deriviatives of energy and then calculate. Note, in computing
        # the derivatives a uniform dz is assumed. However, this is usually not
        # the case in this procedure. However, assuming the energy changes are
        # restricted to the gaps, then the dz in the gap can be used since
        # anywhere else the field is zero.
        # TODO: treat nonuniform grid spacing case.

        zgap = np.argmin(abs(self.z - self.gap_centers[0]))
        dz = self.dz_gap

        dE = np.gradient(E, dz)
        ddE = np.gradient(dE, dz)
        self.dbeam_energy, self.ddbeam_energy = dE, ddE

        kg = -0.25 * (ddE / init_E) / (1 + (E - init_E) / init_E)

        self.kappa_quad = kq
        self.kappa_gap = kg

        return print("Kappa functions calculated and stored.")


class Integrate_KV_equations:
    def __init__(self, lattice_obj):
        self.lattice = lattice_obj

        Q = None
        emit = None

        rx = None
        ry = None
        rxp = None
        ryp = None

        # Initialize statistical quantities calculated after integrating equations.
        avg_rx = None
        avg_ry = None
        avg_rxp = None
        avg_ryp = None

        max_rx = None
        max_ry = None
        max_rxp = None
        max_ryp = None

        min_rx = None
        min_ry = None
        min_rxp = None
        min_ryp = None

        max_spread_rx = None
        max_spread_ry = None
        max_spread_rxp = None
        max_spread_ryp = None

        measure_prod = None
        measure_sum = None

    def calc_envelope_statistics(self):
        """Calculate varous quantities from envelope solutions and store."""

        # Calculate averages over r and r'.
        avg_rx, avg_ry = np.mean(self.rx), np.mean(self.ry)
        avg_rxp, avg_ryp = np.mean(self.rxp), np.mean(self.ryp)

        # Grab max excursions
        max_rx, max_ry = np.max(self.rx), np.max(self.ry)
        max_rxp, max_ryp = np.max(self.rxp), np.max(self.ryp)

        # Grab min excursions
        min_rx, min_ry = np.min(self.rx), np.min(self.ry)
        min_rxp, min_ryp = np.min(self.rxp), np.min(self.ryp)

        # Calculate max spread
        spread_rx = max_rx - min_rx
        spread_ry = max_ry - min_ry
        spread_rxp = max_rxp - min_rxp
        spread_ryp = max_ryp - min_ryp

        # Calculate measures
        measure_prod = np.sqrt(np.mean(self.rx * self.ry))
        measure_sum = (avg_rx + avg_ry) / 2.0

        # Store values
        self.avg_rx = avg_rx
        self.avg_ry = avg_ry
        self.avg_rxp = avg_rxp
        self.avg_ryp = avg_ryp

        self.max_rx = max_rx
        self.max_ry = max_ry
        self.max_rxp = max_rxp
        self.max_ryp = max_ryp

        self.min_rx = min_rx
        self.min_ry = min_ry
        self.min_rxp = min_rxp
        self.min_ryp = min_ryp

        self.max_spread_rx = spread_rx
        self.max_spread_ry = spread_ry
        self.max_spread_rxp = spread_rxp
        self.max_spread_ryp = spread_ryp

        self.measure_prod = measure_prod
        self.measure_sum = measure_sum

    def output_statistics(self):
        """Output envelope information after integrating the equations."""
        mm = 1e-3
        mrad = 1e-3
        print("")
        print("#--------- Envelope Statistics")
        print("   Radii, rx = 2sqrt(<x**2>), rx = 2sqrt(<x**2>):")
        print(
            f"{'   Avg, <rx>, <ry> (mm)':<40} {self.avg_rx/mm:.4f}, {self.avg_ry/mm:.4f}"
        )
        print(
            f"{'   Max, Max[rx], Max[ry] (mm)':<40} {self.max_rx/mm:.4f}, {self.max_ry/mm:.4f}"
        )
        print(
            f"{'   Min, Min[rx], Min[ry] (mm)':<40} {self.min_rx/mm:.4f}, {self.min_ry/mm:.4f}"
        )
        print("")
        print("   Angles, rx`, ry`:")
        print(
            rf"{'   Avg, <rx`>, <ry`> (mrad)':<40} {self.avg_rxp/mrad:.4f}, {self.avg_ryp/mrad:.4f}"
        )
        print(
            rf"{'   Max, Max[rx`], Max[ry`] (mrad)':40} {self.max_rxp/mrad:.4f}, {self.max_ryp/mrad:.4f}"
        )
        print(
            rf"{'   Min, Min[rx`], Min[ry`] (mrad)':<40} {self.min_rxp/mrad:.4f}, {self.min_ryp/mrad:.4f}"
        )
        print("")
        print("   Average Radius Measures:")
        print(f"{'   sqrt(<rx*ry>) (mm)':<40} {self.measure_prod/mm:.4f}")
        print(f"{'   (<rx> + <ry>)/2 (mm)':<40} {self.measure_sum/mm:.4f}")
        print("")

    def integrate_eqns(self, init_r, init_rp, init_Q, init_emit, verbose=False):
        """Integrate the KV equations for the given lattice class.

        The function uses the lattice object provided to integrate the KV
        envelope equations. The lattice_obj will provide the essential arrays
        and overall lattice geometry. The function also takes the initial conditions
        needed to begin the integration which are the initial generalized
        perveance Q and rms-edge emittance. Along with the initial positions
        (rx, ry) and angle (rxp, ryp).

        Parameters
        ----------
        init_r: tuple
            Contains the initial positions (rx, ry).

        init_rp: tuple
            Contains the initial angle (rxp, ryp).

        init_Q: float
            Initial perveance.

        init_emit: float
            Inital emittance.

        verbose: bool
            If true, the various statistics of the envelope solutions will be
            printed out.

        """
        # Grab parameters from the inherited lattice class.
        z = self.lattice.z
        energy = self.lattice.beam_energy
        init_E = energy[0]
        dE = self.lattice.dbeam_energy
        ddE = self.lattice.ddbeam_energy
        kq = self.lattice.kappa_quad
        kg = self.lattice.kappa_gap

        # Initialize arrays with initial conditions.
        vx = np.zeros(len(z))
        vy = np.zeros(len(z))

        ux = np.zeros(len(z))
        uy = np.zeros(len(z))

        Q = np.zeros(len(z))
        emit = np.zeros(len(z))

        ux[0], uy[0] = init_r
        vx[0], vy[0] = init_rp
        Q[0], emit[0] = init_Q, init_emit

        # Integrate equations
        for n in range(1, len(z)):
            this_dz = z[n] - z[n - 1]

            denom_factor = 1 + (energy[n - 1] - init_E) / init_E

            C1 = -0.5 * dE[n - 1] / denom_factor / init_E
            C2x = -kq[n] + kg[n]
            C2y = kq[n] + kg[n]
            C3 = init_Q / denom_factor
            C4 = pow(init_emit, 2) / denom_factor

            vx_term = (
                C1 * vx[n - 1]
                + C2x * ux[n - 1]
                + C3 / (ux[n - 1] + uy[n - 1])
                + C4 / pow(ux[n - 1], 3)
            )
            vy_term = (
                C1 * vy[n - 1]
                + C2y * uy[n - 1]
                + C3 / (ux[n - 1] + uy[n - 1])
                + C4 / pow(uy[n - 1], 3)
            )

            # Update v_x and v_y first.
            vx[n] = vx[n - 1] + vx_term * this_dz
            vy[n] = vy[n - 1] + vy_term * this_dz

            # Use updated v to update u
            ux[n] = ux[n - 1] + vx[n] * this_dz
            uy[n] = uy[n - 1] + vy[n] * this_dz

            # Store Perveance and emittance values.
            Q[n] = C3
            emit[n] = np.sqrt(C4)

        # Store values
        self.rx, self.ry = ux, uy
        self.rxp, self.ryp = vx, vy

        # Calculate and store statistics. If verbose option used, print out
        # statistics.
        self.calc_envelope_statistics()
        if verbose == True:
            self.output_statistics()


class Optimizer:
    def __init__(self, kv_eqn_parameters, lattice_obj, cost_norms):
        self.parameters = kv_eqn_parameters
        self.lattice = lattice_obj
        self.cost_norms = cost_norms

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

    def match_coordinates(self, coordinates):
        """Single input function to min/maximize

        Most optimizers take in a function with a single input that is the
        parameters to optimize for. Here, the desired coordinates are the input
        values compared to the voltage scales for the matching section.
        A 0 cost function here would be mean that for fixed voltages, and
        starting parameters, the initial coordinates and final coordinates are
        equal creating a matched condition. In other words, the initial
        positions and angles were found such that the final position and angle are
        equal to initial."""

        # Unpack parameters
        emit = self.parameters["emit"]
        Q = self.parameters["Q"]

        # Solve KV equations
        init_rx, init_ry, init_rxp, init_ryp = coordinates
        kv_integrator = Integrate_KV_equations(self.lattice)
        kv_integrator.integrate_eqns(
            (init_rx, init_ry), (init_rxp, init_ryp), Q, emit, verbose=False
        )

        # Store solution
        final_rx, final_ry = kv_integrator.rx[-1], kv_integrator.ry[-1]
        final_rxp, final_ryp = kv_integrator.rxp[-1], kv_integrator.ryp[-1]
        self.sol = np.array([final_rx, final_ry, final_rxp, final_ryp])

        # Compute cost and save to history
        cost = self.calc_cost(self.sol, coordinates, self.cost_norms)
        self.cost_hist.append(cost)

        return cost

    def minimize_cost(self, function, init_coords, max_iter=200):
        """Function that will run optimizer and output results

        This function contains the actual optimizer that will be used. Currently
        it is a prepackaged optimizer from the Scipy library. There are numerous
        options for the optimizer and this function can be modified to include
        options in the arguments."""

        print("--Finding match solution.")
        res = sciopt.minimize(
            function,
            init_coords,
            method="nelder-mead",
            options={
                "xatol": 1e-10,
                "fatol": 1e-10,
                "maxiter": max_iter,
                "disp": True,
            },
            bounds=self.bounds,
        )
        self.optimum = res

        return print("--Optimization completed")
