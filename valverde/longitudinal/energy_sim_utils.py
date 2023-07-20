# Utility file for the 1D longitudinal energy gain script. The main script
# advances a 1D beam of particles distributed in energy and time relative to the
# design particle. This script holds various tools to keep the algorithm less
# encumbered and more readable.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.constants as SC

import warp as wp

# different particle masses in eV
# amu in eV
amu_to_eV = wp.amu * pow(SC.c, 2) / wp.echarge
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

    """
    if auto_clip:
        cut = 0
    with sns.axes_style("darkgrid"):
        g = sns.JointGrid(x=xdata, y=ydata, marginal_ticks=True, height=6, ratio=2)
        if auto_clip:
            g.plot_joint(sns.kdeplot, levels=levels, cmap="flare", cut=0)
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
            ylabel=rf"Kinetic Energy $E$ (keV)",
        )
        g.ax_marg_x.label_outer = r"Counts/$N_p$"
        g.ax_marg_y.label_outer = r"Counts/$N_p$"

    return g


def make_lattice_plot(
    zmesh, Efield, gap_centers, phi_s, gap_voltage, gap_width, Fcup_dist
):
    "Plot the acceleration lattice with some labeling."

    E_DC = gap_voltage / gap_width
    Ng = len(gap_centers)

    fig, ax = plt.subplots()
    ax.set_title("Accel. Lattice with Applied Field at t=0", fontsize="large")
    ax.set_xlabel("z [mm]", fontsize="large")
    ax.set_ylabel(r"$E(r=0, z)/E_{DC}$ [kV/mm]", fontsize="large")
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
