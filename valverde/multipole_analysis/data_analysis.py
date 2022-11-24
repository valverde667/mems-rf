import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import interp1d


mpl.rcParams["xtick.direction"] = "in"
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.minor.top"] = True
mpl.rcParams["ytick.direction"] = "in"
mpl.rcParams["ytick.minor.visible"] = True
mpl.rcParams["ytick.right"] = True
mpl.rcParams["ytick.major.right"] = True
mpl.rcParams["ytick.minor.right"] = True
mpl.rcParams["figure.max_open_warning"] = 60

data_path = "/Users/nickvalverde/Desktop/ESQ_files"
df = pd.read_csv(os.path.join(data_path, "multipole_data.csv"))
# df = pd.read_csv(os.path.join(data_path, "vary_rod-radius_rod-fraction.csv"))
data = df.copy()
mm = 1.0e-3
# data.drop_duplicates(subset='R_pole/R_aper', keep='last', inplace=True)


def make_rscale_plot(df, save=False, panel=True):
    """Make Tri-panel of plots for optimizing R-pole"""

    # Pull the n=6, 10, and 14 coefficients. Plot the points where there is zero
    # crossing
    r_scales = df["R_pole/R_aper"].to_numpy()
    a6 = df["Norm A6"].to_numpy()
    a10 = df["Norm A10"].to_numpy()
    a14 = df["Norm A14"].to_numpy()

    # Grab index in r corresponding to absolute minimum multipole coefficient
    r6_min_ind = np.argmin(abs(a6))
    r10_min_ind = np.argmin(abs(a10))
    r14_min_ind = np.argmin(abs(a14))

    if panel:
        fig, (axa6, axa10, axa14) = plt.subplots(
            nrows=3, ncols=1, sharex=True, figsize=(10, 8)
        )
        axa6.set_title(
            "Minimizing n=6 Coefficient w.r.t. ESQ Rod Radius", fontsize="small"
        )
        axa10.set_title(
            "Minimizing n=10 Coefficient w.r.t. ESQ Rod Radius", fontsize="small"
        )
        axa14.set_title(
            "Minimizing n=14 Coefficient w.r.t. ESQ Rod Radius", fontsize="small"
        )
        axa14.set_xlabel(
            r"ESQ Pole Radius as Fraction of Aperture Radius [$R_\mathrm{pole}/r_p$]"
        )

        axa6.scatter(r_scales, a6, s=10)
        axa6.axhline(y=0, c="k", lw=0.5, ls="--")
        axa6.axvline(
            x=r_scales[r6_min_ind],
            lw=1,
            ls="--",
            c="k",
            label=fr"$R_\mathrm{{pole}}/r_p$ = {r_scales[r6_min_ind]:.3f}",
        )
        axa6.axvline(x=r_scales[r10_min_ind], lw=0.7, ls="--", alpha=0.3)
        axa6.axvline(x=r_scales[r14_min_ind], lw=0.7, ls="--", alpha=0.3)
        axa6.set_ylabel(r"Normalized Coefficient $\frac{A_6}{|A_2|}$")

        axa10.scatter(r_scales, a10, s=10)
        axa10.axhline(y=0, c="k", lw=0.5, ls="--")
        axa10.axvline(
            x=r_scales[r10_min_ind],
            lw=1,
            ls="--",
            c="k",
            label=fr"$R_\mathrm{{pole}}/r_p$ = {r_scales[r10_min_ind]:.3f}",
        )
        axa10.axvline(x=r_scales[r6_min_ind], lw=0.7, ls="--", alpha=0.3)
        axa10.axvline(x=r_scales[r14_min_ind], lw=0.7, ls="--", alpha=0.3)
        axa10.set_ylabel(r"Normalized Coefficient $\frac{A_{10}}{|A_2|}$")

        axa14.scatter(r_scales, a14, s=10)
        axa14.axhline(y=0, c="k", lw=0.5, ls="--")
        axa14.axvline(
            x=r_scales[r14_min_ind],
            lw=1,
            ls="--",
            c="k",
            label=fr"$R_\mathrm{{pole}}/r_p$ = {r_scales[r14_min_ind]:.3f}",
        )
        axa14.axvline(x=r_scales[r6_min_ind], lw=0.7, ls="--", alpha=0.3)
        axa14.axvline(x=r_scales[r10_min_ind], lw=0.7, ls="--", alpha=0.3)
        axa14.set_ylabel(r"Normalized Coefficient $\frac{A_{14}}{|A_2|}$")

        axa6.legend()
        axa10.legend()
        axa14.legend()

        plt.tight_layout()
        if save != False:
            plt.savefig(f"{save}.pdf", dpi=400)
        else:
            plt.savefig("coefficient_rscale.pdf", dpi=400)
        plt.show()

    else:
        # Plot each individually
        fig, axa6 = plt.subplots()
        axa6.set_title("Minimizing n=6 Coefficient w.r.t. ESQ Rod Radius")
        axa6.scatter(r_scales, a6, s=10)
        axa6.axhline(y=0, c="k", lw=0.5, ls="--")
        axa6.axvline(
            x=r_scales[r6_min_ind],
            lw=1,
            ls="--",
            c="k",
            label=fr"$R_\mathrm{{pole}}/r_p$ = {r_scales[r6_min_ind]:.3f}",
        )
        axa6.axvline(x=r_scales[r10_min_ind], lw=0.7, ls="--", alpha=0.3)
        axa6.axvline(x=r_scales[r14_min_ind], lw=0.7, ls="--", alpha=0.3)
        axa6.set_xlabel(
            r"ESQ Pole Radius as Fraction of Aperture Radius [$R_\mathrm{pole}/r_p$]"
        )
        axa6.set_ylabel(r"Normalized Coefficient $\frac{A_6}{|A_2|}$")
        axa6.legend()
        plt.tight_layout()
        plt.savefig("a6plot.svg", dpi=400)

        fig, axa10 = plt.subplots()
        axa10.set_title("Minimizing n=10 Coefficient w.r.t. ESQ Rod Radius")
        axa10.scatter(r_scales, a10, s=10)
        axa10.axhline(y=0, c="k", lw=0.5, ls="--")
        axa10.axvline(
            x=r_scales[r10_min_ind],
            lw=1,
            ls="--",
            c="k",
            label=fr"$R_\mathrm{{pole}}/r_p$ = {r_scales[r10_min_ind]:.3f}",
        )
        axa10.axvline(x=r_scales[r6_min_ind], lw=0.7, ls="--", alpha=0.3)
        axa10.axvline(x=r_scales[r14_min_ind], lw=0.7, ls="--", alpha=0.3)
        axa10.set_xlabel(
            r"ESQ Pole Radius as Fraction of Aperture Radius [$R_\mathrm{pole}/r_p$]"
        )
        axa10.set_ylabel(r"Normalized Coefficient $\frac{A_{10}}{|A_2|}$")
        axa10.legend()
        plt.tight_layout()
        plt.savefig("a10plot.svg", dpi=400)

        fig, axa14 = plt.subplots()
        axa14.set_title("Minimizing n=14 Coefficient w.r.t. ESQ Rod Radius")
        axa14.scatter(r_scales, a14, s=10)
        axa14.axhline(y=0, c="k", lw=0.5, ls="--")
        axa14.axvline(
            x=r_scales[r14_min_ind],
            lw=1,
            ls="--",
            c="k",
            label=fr"$R_\mathrm{{pole}}/r_p$ = {r_scales[r14_min_ind]:.3f}",
        )
        axa14.axvline(x=r_scales[r6_min_ind], lw=0.7, ls="--", alpha=0.3)
        axa14.axvline(x=r_scales[r10_min_ind], lw=0.7, ls="--", alpha=0.3)
        axa14.set_xlabel(
            r"ESQ Pole Radius as Fraction of Aperture Radius [$R_\mathrm{pole}/r_p$]"
        )
        axa14.set_ylabel(r"Normalized Coefficient $\frac{A_{14}}{|A_2|}$")
        axa14.legend()
        plt.tight_layout()
        plt.savefig("a14plot.svg", dpi=400)


def plot_rod_ratios(df, interp=False, interp_val=0.0):

    fig, ax = plt.subplots()
    rod_fracs = df["R_rod/R_aper"].to_numpy()
    a6 = df["Norm A6"].to_numpy()
    ax.set_xlabel(r"Ratio of Rod-radius to Aperture Radius $(R_{rod}/r_p)$")
    ax.set_ylabel(r"Normalized Error $A_6/|A_2|$")
    ax.scatter(rod_fracs, a6)
    ax.axhline(y=0, c="k", lw=1)
    plt.tight_layout()

    # Interpolate between two points fo find desired interpolation value
    if interp:
        # Find index of closest two points
        dists = np.sqrt(pow(a6 - interp_val, 2))
        nearest_inds = dists.argsort()[:2]
        yvals = a6[nearest_inds]
        xvals = rod_fracs[nearest_inds]

        s = pd.Series(
            [xvals[0], np.nan, xvals[1]], index=[yvals[0], interp_val, yvals[1]]
        )
        interp_data = s.interpolate(method="index")
        val = interp_data.iloc[1]

        ax.axvline(x=val, label=f"{val:.3f}")
        ax.legend()


# Load in data as numpy arrays
rod_fracs = data["rod-fraction"].to_numpy()
rod_lengths = data["L_esq/R_aper"].to_numpy()
a6 = data["Norm A6"].to_numpy()
a10 = data["Norm A10"].to_numpy()
a14 = data["Norm A14"].to_numpy()

# ------------------------------------------------------------------------------
#     Logical Functions
# Various logical flags for which plots are to be made
# ------------------------------------------------------------------------------
l_plot_rod_fracs = True
l_plot_rod_lengths = True
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title("Leading Order Multipole Error for Various ESQ Lengths")
ax.set_xlabel(r"Ratio of ESQ Length to Aperture Radius $\ell_q/r_p$")
ax.set_ylabel(r"Normalized Multipole Moment $|A_6/A_2|$")
ax.scatter(rod_lengths, abs(a6))
ax.axhline(y=0, ls="--", lw=1, c="k")
plt.savefig("vary_rodlengths.png", dpi=400)
plt.show()
stop
# fig, ax = plt.subplots(figsize=(10,8))
# # Plot curves for 3 rod_radius settings
# r1 = rod_rad[0]  # rp=0.8
# r2 = rod_rad[200]  # rp=1.024138
# r3 = rod_rad[400]  # rp=1.248276
#
# # Grab indices for fraction and a6
# r1_inds = np.where(rod_rad == r1)[0]
# r2_inds = np.where(rod_rad == r2)[0]
# r3_inds = np.where(rod_rad == r3)[0]

# ax.scatter(rod_fracs[r1_inds] / 2, a6[r1_inds], label=fr"$R/r_p$ = {r1:.2f}")
# ax.scatter(rod_fracs[r2_inds] / 2, a6[r2_inds], label=fr"$R/r_p$ = {r2:.2f}")
# ax.scatter(rod_fracs[r3_inds] / 2, a6[r3_inds], label=fr"$R/r_p$ = {r3:.2f}")
# ax.set_title("Leading Order Multipole Error for Various Rod Radii and Rod Area")
# ax.set_xlabel(r"Fraction of Rod Used")
# ax.set_ylabel(r"Normalizeded Multipole Moment $A_6/|A_2|$")
# ax.legend()
# plt.savefig("vary_rod-radius_rod-fraction.png", dpi=400)
# plt.show()

# fig, ax = plt.subplots()
# ax.set_title("n=6 Coefficient for Length of ESQ")
# ax.set_xlabel("Length of ESQ [mm]")
# ax.set_ylabel(r"Normalized Coefficient $\frac{A_6}{|A_2|}$")
# ax.scatter(rod_lengths / mm, a6)
# ax.axhline(y=0, c="k", lw=0.2)
# plt.tight_layout()
# plt.savefig("varylengths_a6.png", dpi=400)
# plt.show()

fig, ax = plt.subplots()
ax.set_title("Leading Order Multipole Error for Various Rod Radii")
ax.set_xlabel(r"Ratio of Rod Radius to Aperture Radius $R_{rod}/r_p$")
ax.set_ylabel(r"Normalized Coefficient $\frac{A_6}{|A_2|}$")
ax.scatter(rod_rad, a6, s=5)
ax.axhline(y=0, c="k", lw=0.2)
ax.axvline(x=1.146, c="k", ls="--", lw=1, label="Faltens")
ax.legend()
plt.tight_layout()
plt.savefig("varyradius_a6.png", dpi=400)
plt.show()
stop
# fig, ax = plt.subplots()
# ax.set_title("Leading Order Multipole Error for Chopped Rods")
# ax.set_xlabel(r"Fraction of Rod Used $F$ ")
# ax.set_ylabel(r"Normalized Coefficient $\frac{A_6}{|A_2|}$")
# ax.scatter(rod_fracs / 2, a6)
# ax.axhline(y=0, c="k", lw=0.2)
# plt.tight_layout()
# plt.savefig("choppedA6_zoomed.png", dpi=400)
# plt.show()

# Make r-plots for various lengths
plot_all_lengths = False
if plot_all_lengths:
    esq_lengths = data["Lesq[mm]"].unique()
    rvals = np.zeros(len(esq_lengths))
    for i, ell in enumerate(esq_lengths):
        this_data = data[data["Lesq[mm]"] == ell]
        make_rscale_plot(this_data)
