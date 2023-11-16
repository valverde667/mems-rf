# Utility file for the 1D longitudinal energy gain script. The main script
# advances a 1D beam of particles distributed in energy and time relative to the
# design particle. This script holds various tools to keep the algorithm less
# encumbered and more readable.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.constants as SC
import periodictable

# different particle masses in eV
# amu in eV
amu_to_eV = SC.physical_constants["atomic mass constant energy equivalent in MeV"][0]
kV = 1000
keV = 1000
MHz = 1e6
cm = 1e-2
mm = 1e-3
um = 1e-6
us = 1e-6
ns = 1e-9  # nanoseconds
mA = 1e-3
uA = 1e-6
twopi = 2 * np.pi


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

    Returns
    -------
    index : int
        Index for the mesh-value closest to the desired value.
    """

    # Check if value is already in mesh
    if value in mesh:
        index = np.where(mesh == value)[0][0]

    else:
        index = np.argmin(abs(mesh - value))

    return index


def rf_volt(t, freq=13.6 * MHz):
    return np.cos(2 * np.pi * freq * t)


def calc_dipole_deflection(voltage, energy, length=50 * mm, g=11 * mm, drift=185 * mm):
    """Calculate an ion's deflection from a dipole field"""

    coeff = voltage * length / g / energy
    deflection = coeff * (length / 2 + drift)

    return deflection


def calc_Hamiltonian(W_s, phi_s, W, phi, f, V, m, g=2 * mm, T=1, ret_coeffs=False):
    """Calculate the Hamiltonian.

    The non-linear Hamiltonian is dependent on the energy E, RF-frequency f,
    gap voltage V, gap width g, ion mass m and transit time factor T.

    Parameters
    ----------
    W_s : float
        Synchronous kinetic energy
    phi_s : float
        Synchronous phase
    W : float or array
        Kinetic energy for particles.
    phi : float or array
        phase of particles
    f : float
        Frequency of RF gaps.
    V : float
        Voltage applied on RF gaps.
    g : float
        Width of acceleration gap.
    m : float
        Ion mass.
    T : float
        Transit time factor.
    ret_coeffs : bool
        If True, the coeffcients A and B will be returned.

    Returns
    -------
    H : float or array
        Hamiltonian values.

    """

    bs = utils.beta(W_s, mass=m)
    hrf = SC.c / f
    A = twopi / hrf / pow(bs, 3)
    B = V * T / g / m

    term1 = 0.5 * A * pow((W - W_s) / m, 2)
    term2 = B * (np.sin(phi) - phi * np.cos(phi_s))
    H = term1 + term2

    if ret_coeffs:
        return H, (A, B)

    else:
        return H


def calc_root_H(phi, phi_s=-np.pi / 2):
    """ "Function for finding the 0-root for the Hamiltonian.

    The non-linear Hamiltonian has roots at phi=phi_s and phi_2.

    """
    term1 = np.sin(phi) + np.sin(phi_s)
    term2 = -np.cos(phi_s) * (phi - phi_s)

    return term1 + term2


def calc_emittance(x, y):
    """Calculate the RMS emittance for x,y"""
    term1 = np.mean(pow(x, 2)) * np.mean(pow(y, 2))
    term2 = pow(np.mean(x * y), 2)
    emit = np.sqrt(term1 - term2)

    return emit


def uniform_elliptical_load(Np, xmax, ymax, xc=0, yc=0, seed=42):
    """Create Np particles within ellipse with axes xmax and ymax


    Parameters:
    -----------
    Np : Int
        Number of sample points

    xmax : float
        Half-width of x-axis

    ymax : float
        Half-width of y-axis

    xc: float
        Center of ellipse in x

    yc: float
        Center of ellipse in y

    """

    keep_x = np.zeros(Np)
    keep_y = np.zeros(Np)
    kept = 0
    np.random.seed(seed)
    while kept < Np:
        x = np.random.uniform(xc - xmax, xc + xmax)
        y = np.random.uniform(yc - ymax, yc + ymax)
        coord = np.sqrt(pow((x - xc) / xmax, 2) + pow((y - yc) / ymax, 2))
        if coord <= 1:
            keep_x[kept] = x
            keep_y[kept] = y
            kept += 1

    return (keep_x, keep_y)


def uniform_box_load(Np, xlims, ylims, xc=0, yc=0, seed=42):
    """Create Np particles within ellipse with axes xmax and ymax


    Parameters:
    -----------
    Np : Int
        Number of sample points

    xlims : tuple
        Tuple containing (xmin, xmax)

    ylims : tuple
        Tuple containing (ymin, ymax)

    xc: float
        Center of box in x

    yc: float
        Center of box in y

    """
    xmin, xmax = xlims
    ymin, ymax = ylims

    x_coords = np.zeros(Np)
    y_coords = np.zeros(Np)
    np.random.seed(seed)
    for i in range(Np):
        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(ymin, ymax)
        x_coords[i] = x
        y_coords[i] = y

    return (x_coords, y_coords)


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


def calc_gap_centers(
    E_s,
    mass,
    phi_s,
    gap_mode,
    dsgn_freq,
    dsgn_gap_volt,
    match_after_gap=None,
    match_length=None,
):
    """Calculate Gap centers based on energy gain and phaseing.

    When the gaps are operating at the same phase then the distance is
    beta*lambda / 2. However, if the gap phases are different then there is
    additional distance that the particle must cover to arrive at this new phase.

    The initial phasing is assumed to start with the first gap having a minimum
    E-field (maximally negative). If this is not desired, the offset should be
    calculated outside this function.

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
    energy = [E_s]

    # Loop through number of gaps and assign the center-to-center distance.
    # Update the energy at end.
    for i in range(len(phi_s)):
        this_E = energy[i]
        this_beta = beta(this_E, mass)
        this_cent = this_beta * h / 2.0
        shift = this_beta * h * gap_mode[i]
        cent_offset = (phi_s[i] - phi_s[i - 1]) * this_beta * h / twopi
        if i < 1:
            gap_dist[i] = (phi_s[i] + np.pi) * this_beta * h / twopi + shift
        else:
            gap_dist[i] = this_cent + cent_offset + shift

        dsgn_Egain = dsgn_gap_volt * np.cos(phi_s[i])
        energy.append(this_E + dsgn_Egain)

    if match_after_gap != None:
        # Need to adjust gap spacing so that resonance condition is greater than
        # the length of the matching section.
        this_gap_dist = gap_dist[match_after_gap]
        shift = beta(energy[match_after_gap], mass) * h
        while this_gap_dist < match_length:
            this_gap_dist += shift
        gap_dist[match_after_gap] = this_gap_dist

    # gap locations from zero are given by cumulative sum of center-to-center
    # distances.
    return gap_dist.cumsum()


def make_dist_plot(
    xdata,
    ydata,
    xlabel="",
    ylabel="",
    auto_clip=True,
    xclip=(None, None),
    yclip=(None, None),
    levels=30,
    bins=50,
    xref=None,
    yref=None,
    weight=None,
    dx_bin=None,
    dy_bin=None,
):
    """Quick Function to make a joint plot for the distribution of x and y data.

    The function uses seaborns JointGrid to create a KDE plot on the main figure and
    histograms of the xdata and ydata on the margins.

    Parameters
    ----------
    xdata: array
        Data to plot on x-axis and make a histogram on top of main plot.

    ydata: array
        Data to plot on y-axis and make histogram to the right of main plot.

    xlabel: string
        Label for xdata

    ylabel: string
        Label for ydata

    auto_clip: bool
        This option will clip the KDE plot so that areas without data will not
        be represented. For example, if there is data to some xmax point because
        the KDE plot is uses a distribution to estimate, the heat map will show
        have contours greater than xmax. However, if auto_clip is set to True, the
        data has a hard cut off and there will be no contour outside the data limits.

    xclip: tuple
        Tuple of bool values. The first entry is the lower limit to clip values.
        The second entry is the upper limit. One or both can be set to a value.

    yclip: tuple
        Option for the y-values (see xclip).

    levels: int
        Number of contours to map.

    bins: int
        Number of bins to use for binning data.

    xref: float
        This value will add a reference line to the main plot for assistance in
        visualizing the x-data. For example, setting xref=1.0 will plot a vertical
        dashed line on the plot at x=1.0. This line will also show on the histogram.

    yref:
        Reference value for ydata (see xref).

    weight: float
        Value to weight the data by. Useful if one wants to bin the fractional
        counts rather than total counts.

    dx_bin: float
        This value will set the width of the histogram bins. This option superced
        the bins argument and thus should only be set if one desires a fixed resolution
        and not a fixed number of bins.

    dy_bin: float
        Fixed bin width of y-data (see dx_bin).

    Returns
    -------
    g: object
        Plot object that can be manipulated through matplotlib options for plot
        objects.

    """
    if auto_clip:
        cut = 0
    with sns.axes_style("darkgrid"):
        g = sns.JointGrid(x=xdata, y=ydata, marginal_ticks=True, height=6, ratio=2)
        if auto_clip:
            g.plot_joint(sns.kdeplot, levels=levels, fill=True, cmap="flare", cut=0)
        else:
            g.plot_joint(
                sns.kdeplot, fill=True, levels=levels, cmap="flare", clip=(xclip, yclip)
            )
        sns.histplot(
            x=xdata,
            bins=bins,
            edgecolor="k",
            lw=0.5,
            alpha=0.7,
            stat="count",
            weights=weight,
            binwidth=dx_bin,
            ax=g.ax_marg_x,
        )
        sns.histplot(
            y=ydata,
            bins=bins,
            edgecolor="k",
            lw=0.5,
            alpha=0.7,
            stat="count",
            weights=weight,
            binwidth=dy_bin,
            ax=g.ax_marg_y,
        )
        if xref != None:
            g.refline(x=xref)
        if yref != None:
            g.refline(y=yref)

        g.set_axis_labels(
            xlabel=r"Relative Time Difference $\Delta t / \tau_{rf}$",
            ylabel=rf"Kinetic Energy $\mathcal{{E}}$ (keV)",
        )
        g.ax_marg_x.set_ylabel(r"Counts/$N_p$")
        g.ax_marg_y.set_xlabel(r"Counts/$N_p$")

    plt.tight_layout()

    return g


def make_lattice_plot(
    zmesh, Efield, gap_centers, phi_s, gap_voltage, gap_width, Fcup_dist
):
    "Plot the acceleration lattice with some labeling."

    E_DC = gap_voltage / gap_width
    Ng = len(gap_centers)

    fig, ax = plt.subplots()
    ax.set_title("Accel. Lattice with Applied Field at t=0", fontsize="large")
    ax.set_xlabel("z (mm)", fontsize="large")
    ax.set_ylabel(r"$E_z(r=0, z)/E_{DC}$ (kV/mm)", fontsize="large")
    ax.plot(zmesh / mm, Efield / E_DC)
    ax.axvline(gap_centers[0] / mm, c="grey", lw=1, ls="--", label="Gap Center")
    if Ng > 1:
        for i, cent in enumerate(gap_centers[1:]):
            ax.axvline(cent / mm, c="grey", lw=1, ls="--")

    ax.axvline(
        (gap_centers[-1] + gap_width / 2 + Fcup_dist) / mm,
        c="r",
        lw=1,
        ls="dashdot",
        label="Fcup",
    )
    ax.legend()
    plt.tight_layout()

    fig, ax = plt.subplots()
    ax.yaxis.grid(True)
    ax.set_title("Synchronous Phase at Gap")
    ax.set_xlabel(r"$N_g$")
    ax.set_ylabel(r"$\phi_s$ [deg]")
    ax.scatter([i + 1 for i in range(Ng)], phi_s / np.pi * 180)
    plt.tight_layout()
