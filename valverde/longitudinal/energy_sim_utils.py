# Utility file for the 1D longitudinal energy gain script. The main script
# advances a 1D beam of particles distributed in energy and time relative to the
# design particle. This script holds various tools to keep the algorithm less
# encumbered and more readable.

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def make_dist_plot(
    xdata,
    ydata,
    xlabel="",
    ylabel="",
    xclip=(None, None),
    yclip=(None, None),
    levels=30,
    bins=50,
    xref=None,
    yref=None,
):
    """Quick Function to make a joint plot for the distribution of x and y data.

    The function uses seaborns JointGrid to create a KDE plot on the main figure and
    histograms of the xdata and ydata on the margins.

    """
    with sns.axes_style("darkgrid"):
        g = sns.JointGrid(x=xdata, y=ydata)
        g.plot_joint(
            sns.kdeplot, fill=True, levels=levels, cmap="flare", clip=(xclip, yclip)
        )
        g.plot_marginals(sns.histplot, bins=30, edgecolor="k", lw=0.5, alpha=0.7)
        if xref != None:
            g.refline(x=xref)
        if yref != None:
            g.refline(y=yref)

        g.set_axis_labels(
            xlabel=r"Relative Time Difference $\Delta t / \tau_{rf}$",
            ylabel=rf"Kinetic Energy $E$ (keV)",
        )

    return g
