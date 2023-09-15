# Script contains functions and classes used in the transverse matching
# modeling. They were moved here to reduce clutter and also provide a place
# to focus on improving the functions and classes.
# Improvement suggestions are provided in the notes below.

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as SC
import scipy.optimize as sciopt
import scipy.integrate as integrate
import itertools
import csv
import os
import pdb


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
def write_envelope_data(file_name, data, header=None):
    """ "Helper function to write envelope data to csv file.

    This function will check if the file_name already exists. If it does, then
    it will write to a new row the data provided. If it doesnt, it will create
    the file with the headers given. It may be easier to create the file in
    advance with the headers then to write them out as an array explicitly.

    Parameters
    ----------
    file_name: string
        Name of csv file that the data is to be written to.

    data: array:
        Array of data. This is up to the user based on what data is meant to be
        saved.

    header: list or array of strings
        This will be the header labels for the csv file.
    """

    csv_file = file_name

    file_exists = os.path.isfile(csv_file)

    with open(csv_file, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(header)

        writer.writerow(data)

    if file_exists:
        return print(f"Data have been appended to {csv_file}.")
    else:
        return print(f"CSV file {csv_file} has been created with headers and data.")


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


def calc_gap_centers(E_s, mass, phi_s, gap_mode, dsgn_freq, dsgn_gap_volt):
    """Calculate Gap centers based on energy gain and phaseing.

    When the gaps are operating at the same phase then the distance is
    beta*lambda / 2. However, if the gap phases are different then there is
    additional distance that the particle must cover to arrive at this new phase.

    Parameters:
    -----------
    E_s: float
        Initial beam energy in units of (eV).

    mass: float
        Mass of the ion in question in unites of (eV).

    phi_s: list or array
        The synchronous phases of arrival for each gap. Assumed to be rising in
        magnitude with the max phase being <= 0. In units of (radians).

    gap_mode: list or array
        Integer values that will give the n-value for 2npi if additional spacing is
        needed.

    dsgn_freq: float
        Operating frequency of the gaps. Assumed constant througout.

    dsgn_gap_volt: float
        Operating voltage of gaps. Assumed constant throughout.

    Returns:
    --------
    gap_centers: array
        Cumulative sum of the gap distances (center-to-center) for the lattice.
    """

    # Initialize arrays and any useful constants.
    gap_dist = np.zeros(len(phi_s))
    h = SC.c / dsgn_freq

    # Loop through number of gaps and assign the center-to-center distance.
    # Update the energy at end.
    for i in range(len(phi_s)):
        this_beta = beta(E_s, mass)
        this_cent = this_beta * h / 2.0
        shift = this_beta * h * gap_mode[i]
        cent_offset = (phi_s[i] - phi_s[i - 1]) * this_beta * h / twopi
        if i < 1:
            gap_dist[i] = (phi_s[i] + np.pi) * this_beta * h / twopi + shift
        else:
            gap_dist[i] = this_cent + cent_offset + shift

        dsgn_Egain = dsgn_gap_volt * np.cos(phi_s[i])
        E_s += dsgn_Egain

    # gap locations from zero are given by cumulative sum of center-to-center
    # distances.
    return gap_dist.cumsum()


def calc_quad_centers(gap_centers, lq, d, g, spacing):
    """Calculate quad centers within a region given a quad length and desired
    separation.

    The quadrupoles will need to fit between the drift space of the acceleration
    plates. In early lattice periods when this distance has not grown appreciably
    there may not be enough space. The length of the ESQ whether this is the
    physical length of physical length + fringe field might class with the plates.
    Additionally, some separation may be desired between the quads.
    This function will attempt to find these centers where the quads can fit and
    output some messages if there are overlaps or not enough space.

    Parameters
    ----------
    gap_centers: list or array
        The lattice period contains two acceleration gaps and then a drift region
        between gaps2 and gaps3. Within this drift region the ESQs can be placed
        assuming sufficient space. To ensure a drift region, the list should have
        2n+1 gaps included.
    lq: float
        Length of quadrupole being used. Length is assumed to be the same for both
        in a doublet structure.

    d: float
        Desired separation of quadrupoles. This separation is taken at the end
        points. That is, if z1 the center for quad1, z2 the center for quad2,
        and z1 < z2, then:
            d = (z2 - lq/2) - (z1 + lq/2).
    g: float
        Width of the gap being used.

    spacing: str or float
        The spacing scheme to use for distancing the ESQs. If set to 'equal' the
        drift region will be evenly divided and the doublet placed to evenly
        separate the region into thirds. If 'maxsep', the ESQs are moved as close
        as possibly to the end plates on either sides and the distance between
        ESQ is max. If a float is given, then the ESQ separation end-to-end is
        set to this distance.

    Returns
    -------
    quad_centers: array
        Array containing the two centers [z1, z2]
    """
    quad_centers = []
    # Loop through the lattice periods availabe in gaps. For every 3 gaps there
    # is information for one lattice period. The third gap start plate is the start
    # of the next lattice period (end of the current lattice period).
    for i in range(1, len(gap_centers)):
        if 2 * i > int(len(gap_centers) - 1):
            return np.array(quad_centers)
        else:
            zdrift = gap_centers[2 * i] - gap_centers[2 * i - 1] - g
            if type(spacing) == float:
                zquad = 2 * lq + d

                # Test if there is enough space for desired placement.
                # If there is, go for it. if not, find a range and output to user.
                if zquad < zdrift:
                    # Calculate midpoint and then palce quads using d as a shift.
                    zmid = (gap_centers[2 * i] + gap_centers[2 * i - 1]) / 2.0
                    z1 = zmid - d / 2.0 - lq / 2.0
                    z2 = zmid + d / 2.0 + lq / 2.0
                    quad_centers.append(z1)
                    quad_centers.append(z2)
                else:
                    # Calculate how much space is left.
                    overlap = abs(zdrift - zquad)
                    if overlap < d:
                        print(f"Overlap(mm): {overlap/mm:.4f}.")
                        print("Overlap is less than d. Try decreasing d.")
                    else:
                        print(f"Overlap(mm): {overlap/mm:.4f}.")
                        print("Not enough space even if d is decreased.")
                        print(
                            "Try using an overlapping field and use 'one' quad by centering",
                            "the doublet field at the midpoint of the quads.",
                        )
                        print("")
                        return quad_centers
            elif type(spacing) == str:
                if spacing == "equal":
                    zquad = 2.0 * lq
                    free_space = zdrift - zquad
                    d = free_space / 4.0

                    z1 = gap_centers[2 * i - 1] + g / 2 + d + lq / 2.0
                    z2 = gap_centers[2 * i] - g / 2 - d - lq / 2.0
                    quad_centers.append(z1)
                    quad_centers.append(z2)

                elif spacing == "maxsep":
                    pad = 0.05 * lq
                    z1 = gap_centers[2 * i - 1] + g + lq / 2
                    z2 = gap_centers[2 * i] - g - lq / 2
                    quad_centers.append(z1)
                    quad_centers.append(z2)

            else:
                "Select either 'maxsep', 'equal', or input a float for spacing."
                return False


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


def env_radii_free_expansion(init_rx, init_rxp, emit, z):
    """Analytic formula for KV envelope rms-radii evolution in drift space"""

    c_linear = 2.0 * init_rxp / init_rx
    linear_term = c_linear * z

    c_sq = (
        (1.0 + pow(init_rx, 2) * pow(init_rxp, 2) / pow(init_emit, 2))
        * pow(init_emit, 2)
        / pow(init_rx, 4)
    )
    sq_term = c_sq * pow(z, 2)

    return init_rx * np.sqrt(1.0 + linear_term + sq_term)


def prepare_quad_inputs(
    quad_centers, quad_info, Vq, scale_factor=6.40e6, per_lattice=False
):
    """Helper function to package the inputs so they can be fed into Lattice class.

    The inputs for building the lattice take in the quad and gap data as a
    tuple contatining lists ([centers], [z-extents], [field data]) where the
    centers are where the field is to be centered on the mesh, the zextents is the
    length of the field, and the field data the extracted field or hard edge value.

    This function assumes that the same field structure, i.e., same z-extents
    will be used for each gap center provided. The fields themselves will be scaled
    appropriately to match the given voltages Vq and Vg.

    Parameters
    ----------
    quad_centers: list or array
        Location on z-mesh of where the field is to be placed (centered).

    quad_info: tuple
        A tuple containing the ([zdata], [field data]) of the quadrupole gradient.
        The zdata is expected to be centered on z=0 due to the Warp extraction.
        The data will be shifted so that the center is coincident with the given
        center.

    Vq: list or array
        Desired voltages for the quadrupole. Each gradient corresponds to a given
        voltage for the ESQ design. This scaling was found external to this script
        and used here. The field is normalized by the max and then scaled up to match
        the gradient produced by the given voltage.

    per_lattice: bool
        If the quadrupoles are close enough to eachother, their fringe fields
        well zero eachother causing the fields to be altered. This case is not
        treated in this function and the user is recommended to extract the field
        data from a Warp sim with both quadrupoles being modeled. In this case,
        the option can be set and the fields will be altered by first assuming
        the zero point is at the geometric center between the two quads and then
        scaling the left and right of the field data.
        Note, if this option is selected then the geometric center will be placed
        at the provided z center.
    """

    # The manipulations that need to happen are shifting the spatial arrays by
    # the gap centers and scaling the voltage. First, unpack the data.
    zgrad, grad = quad_info
    z_data = []
    grad_data = []

    if per_lattice == True:
        # Loop through quad centers. In this case there is one center and one
        # field array containing data for two quadrupoles that must have their
        # respective voltages scaled.
        for i in range(len(quad_centers)):
            zind = np.argmin(abs(zgrad))
            this_grad = grad.copy()

            # Scale left and right sides of the field array.
            this_grad[: zind + 1] *= Vq[i] * scale_factor
            this_grad[zind + 1 :] *= Vq[i + 1] * scale_factor

            z_data.append(zgrad + quad_centers[i])
            grad_data.append(this_grad)

    else:
        for i in range(len(quad_centers)):
            z_data.append(zgrad + quad_centers[i])
            grad_data.append(grad.copy() * Vq[i] * scale_factor)

    qinfo = (quad_centers, z_data, grad_data)
    return qinfo


def prepare_gap_inputs(gap_centers, gap_info, Vg):
    """Helper function to package the inputs so they can be fed into Lattice class.

    The inputs for building the lattice take in the gap data as a
    tuple contatining lists ([centers], [z-extents], [field data]) where the
    centers are where the field is to be centered on the mesh, the zextents is the
    length of the field, and the field data the extracted field or hard edge value.

    This function assumes that the same field structure, i.e., same z-extents
    will be used for each gap center provided. The fields themselves will be scaled
    appropriately to match the given voltages Vq and Vg.

    Parameters
    ----------
    gap_centers: list or array
        Location on z-mesh of where the field is to be placed (centered).

    gap_info: tuple
        A tuple containing the ([zdata], [field data]) of the gap electric field.
        The zdata is expected to be centered on z=0 due to the Warp extraction.
        The data will be shifted so that the center is coincident with the given
        center.

    Vg: list or array
        Desired voltages for the gap. Each electric field provile corresponds to a given
        voltage for the gap design. This scaling was found external to this script
        and used here. The field is normalized by the max and then scaled up to match
        the field produced by the given voltage.
    """

    # The only manipulations that need to happen are the voltage scaling.
    # Once this is done, package into a tuple.
    zgap, Ez = gap_info
    z_data = []
    Ez_data = []
    for i in range(len(gap_centers)):
        z_data.append(zgap + gap_centers[i])
        Ez_data.append(Ez * 657.3317 * Vg[i])

    ginfo = (gap_centers, z_data, Ez_data)
    return ginfo


class Lattice:
    def __init__(self):
        self.zmin = 0.0
        self.zmax = None
        self.z = None
        self.dz = None

        self.Ng = None
        self.gap_centers = None
        self.lg = None
        self.gap_field_data = None
        self.dz_gap = None

        self.Nq = None
        self.quad_centers = None
        self.lq = None
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
        quad_lengths = []
        gap_centers = []
        gap_lengths = []

        for i, cond in enumerate(conductors):
            if cond == "q":
                z_patch = quad_info[1][q_counter]
                this_zc = quad_info[0][q_counter]
                this_zext = z_patch[-1] - z_patch[0]
                this_field = quad_info[2][q_counter]

                # Append some data before incrementation
                quad_centers.append(this_zc)
                quad_lengths.append(this_zext)
                self.dz_quad = z_patch[1] - z_patch[0]
                q_counter += 1

                field_loc = np.where((z > z_patch[0]) & (z < z_patch[-1]))[0]

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
                quad_lengths.append(this_zext)
                self.dz_gap = z_patch[1] - z_patch[0]
                g_counter += 1

                field_loc = np.where((z > z_patch[0]) & (z < z_patch[-1]))[0]
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
            self.quad_lengths = quad_lengths
            self.quad_field_data = quad_data

        else:
            self.Nq = 0
            self.quad_centers = 0
            self.quad_lengths = 0
            self.quad_field_data = np.zeros(len(z))

        if g_counter > 0:
            self.Ng = g_counter
            self.gap_centers = gap_centers
            self.gap_lengths = gap_lengths
            self.gap_field_data = gap_data

        else:
            self.Ng = 0
            self.gap_centers = 0
            self.gap_lengths = 0
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

        # Compute deriviatives of energy and then calculate. Note, there can be
        # spikes where the field is stitched in. numpy's gradient can handle
        # uniform spacing. However, if the spacing is jagged there may be a
        # spike that is of order .01% the max. Probablly not be an issue, but
        # will look confusing when presenting.
        dE = np.gradient(E, self.z)
        ddE = np.gradient(dE, self.z)
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

        # Phase advance period lattice period in rads
        sigmax = None
        sigmay = None

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

        # Denote flutter of a as Fa = (Max[a]-Min[a]) / Avg[a]
        Fx = None
        Fy = None
        Fxp = None
        Fyp = None

        measure_prod = None
        measure_sum = None

    def calc_envelope_statistics(self):
        """Calculate varous quantities from envelope solutions and store."""

        # Integrate over the lattice period to fine phase advance.
        sigmax = integrate.simps(self.emit / pow(self.rx, 2), self.lattice.z)
        sigmay = integrate.simps(self.emit / pow(self.ry, 2), self.lattice.z)

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
        self.sigmax = sigmax
        self.sigmay = sigmay

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

        self.Fx = spread_rx / avg_rx
        self.Fy = spread_ry / avg_ry
        self.Fxp = spread_rxp / avg_rxp
        self.Fyp = spread_ryp / avg_ryp

        self.measure_prod = measure_prod
        self.measure_sum = measure_sum

    def output_statistics(self):
        """Output envelope information after integrating the equations."""
        mm = 1e-3
        mrad = 1e-3
        print("")
        print("#--------- Envelope Statistics")
        print("   Phase-adv per lattice period:")
        print(
            f"{'   sigmax, sigmay (deg/period)':<40} {self.sigmax/np.pi*180:.4f}, {self.sigmay/np.pi*180:.4f}"
        )
        print("")
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
        print(f"{'   Flutter-x (rx-spread)/<rx>':<40} {self.Fx:.4f}")
        print(f"{'   Flutter-y (ry-spread)/<ry>':<40} {self.Fy:.4f}")
        print("")

    def integrate_eqns(self, init_coordinates, init_Q, init_emit, verbose=False):
        """Integrate the KV equations for the given lattice class.

        The function uses the lattice object provided to integrate the KV
        envelope equations. The lattice_obj will provide the essential arrays
        and overall lattice geometry. The function also takes the initial conditions
        needed to begin the integration which are the initial generalized
        perveance Q and rms-edge emittance. Along with the initial positions
        (rx, ry) and angle (rxp, ryp).

        Parameters
        ----------
        init_coordinates: list, array, or tuple
            Contains the initial radii and angle [rx, ry, rx', ry'].

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

        ux[0], uy[0], vx[0], vy[0] = init_coordinates
        Q[0], emit[0] = init_Q, init_emit

        # Integrate equations
        for n in range(1, len(z)):
            this_dz = z[n] - z[n - 1]

            denom_factor = 1 + (energy[n - 1] - init_E) / init_E

            C1 = -0.5 * dE[n - 1] / denom_factor / init_E
            C2x = -kq[n] + kg[n]
            C2y = kq[n] + kg[n]
            C3 = 2.0 * init_Q / denom_factor
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
            Q[n] = C3 / 2.0
            emit[n] = np.sqrt(C4)

        # Store values
        self.rx, self.ry = ux, uy
        self.rxp, self.ryp = vx, vy
        self.Q = Q
        self.emit = emit

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
        self.sol_voltage = None
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

    def match_fixed_voltage(self, coordinates):
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
        kv_integrator = Integrate_KV_equations(self.lattice)
        kv_integrator.integrate_eqns(coordinates, Q, emit, verbose=False)

        # Store solution
        final_rx, final_ry = kv_integrator.rx[-1], kv_integrator.ry[-1]
        final_rxp, final_ryp = kv_integrator.rxp[-1], kv_integrator.ryp[-1]
        self.sol = np.array([final_rx, final_ry, final_rxp, final_ryp])

        # Compute cost and save to history
        cost = self.calc_cost(self.sol, coordinates, self.cost_norms)
        self.cost_hist.append(cost)

        return cost

    def match_fixed_coordinates(self, voltage, init_coordinates):
        """Single input function to min/maximize

        Most optimizers take in a function with a single input that is the
        parameters to optimize for. Here, the coorditaes rx,ry,rxp,ryp are fixed
        and we wish to find the voltages that will give the matched condition so
        that the initial and final coordinates are the same.
        The initial lattice is built with a voltage setting used. Instead of
        reinstating and building the lattice each time, we can instead locate the
        quadrupoles using the quad_centers attribute and the length of the
        quadrupoles.
        """

        # Grab the quadrupole centers and array length of the field data from the
        # lattice object.
        lattice = self.lattice
        quad_centers = lattice.quad_centers
        lq = lattice.quad_lengths[0]
        Vq = voltage

        print("*****HERE:", init_coordinates)
        print("*****HERE:", voltage)

        # There should be a 1:1 correspondance between quad_centers and voltages
        assert_message = "Number of voltages != number of quad centers in lattice."
        assert len(quad_centers) == len(Vq), assert_message

        # Loop through quad_centers and voltages. For each center, locate the center
        # and the ESQ extent in the z-array.
        for i in range(len(quad_centers)):
            zq = quad_centers[i]
            esq_start = np.argmin(abs(self.lattice.z - (zq - lq / 2)))
            esq_end = np.argmin(abs(self.lattice.z - (zq + lq / 2)))

            # Normalize the field first and then scale it.
            # TODO: Incorporate the scaling factor as a variable to be grabbed.
            #   With the current implementation, the user may catch the neccessity
            # to change the scale factor in the preparation function but not here.
            this_field = lattice.quad_field_data[esq_start : esq_end + 1]
            this_field = this_field / np.max(abs(this_field))
            this_field = this_field * Vq[i] / 1.562e-7
            lattice.quad_field_data[esq_start : esq_end + 1] = this_field

        # With the fields rescaled, the kappa function has to be revaluated. Since
        # the ESQ fields do not affect acceleration, only one method needs to be
        # called and the beam energy has a function of z does not need to be
        # recalculated.
        lattice.calc_lattice_kappa()

        # Unpack parameters
        emit = self.parameters["emit"]
        Q = self.parameters["Q"]

        # Solve KV equations
        kv_integrator = Integrate_KV_equations(self.lattice)
        kv_integrator.integrate_eqns(init_coordinates, Q, emit, verbose=False)

        # Store solution
        final_rx, final_ry = kv_integrator.rx[-1], kv_integrator.ry[-1]
        final_rxp, final_ryp = kv_integrator.rxp[-1], kv_integrator.ryp[-1]
        final_coordinates = np.array([final_rx, final_ry, final_rxp, final_ryp])
        self.sol = np.array([final_rx, final_ry, final_rxp, final_ryp])
        self.sol_voltage = voltage

        # Compute cost and save to history
        cost = self.calc_cost(self.sol, init_coordinates, self.cost_norms)
        self.cost_hist.append(cost)

        return cost

    def minimize_cost_fixed_voltage(self, function, init_coords, max_iter=120):
        """Function that will run optimizer and output results

        This function contains the actual optimizer that will be used. Currently
        it is a prepackaged optimizer from the Scipy library. There are numerous
        options for the optimizer and this function can be modified to include
        options in the arguments."""

        print("--Finding matched solution.")
        res = sciopt.minimize(
            function,
            init_coords,
            method="nelder-mead",
            options={
                "xatol": 1e-8,
                "fatol": 1e-8,
                "maxiter": max_iter,
                "disp": True,
            },
            bounds=self.bounds,
        )
        self.optimum = res

    def minimize_cost_fixed_coordinates(
        self, function, voltage, init_coordinates, max_iter=120
    ):
        """Function that will run optimizer and output results

        This function contains the actual optimizer that will be used. Currently
        it is a prepackaged optimizer from the Scipy library. There are numerous
        options for the optimizer and this function can be modified to include
        options in the arguments."""

        print("--Finding matched solution.")
        res = sciopt.minimize(
            function,
            voltage,
            args=(init_coordinates),
            method="nelder-mead",
            options={
                "xatol": 1e-8,
                "fatol": 1e-8,
                "maxiter": max_iter,
                "disp": True,
            },
            bounds=self.bounds,
        )
        self.optimum = res
