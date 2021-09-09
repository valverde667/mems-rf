import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

data_path = "/Users/nickvalverde/Desktop/ESQ_files"
df = pd.read_csv(os.path.join(data_path, "multipole_data.csv"))
data = df.copy()
# data.drop_duplicates(subset='R_pole/R_aper', keep='last', inplace=True)


def make_rscale_plot(df, save=False):
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

    fig, (axa6, axa10, axa14) = plt.subplots(
        nrows=3, ncols=1, sharex=True, figsize=(10, 8)
    )
    axa6.set_title("Minimizing n=6 Coefficient w.r.t. ESQ Rod Radius", fontsize="small")
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
        label=fr"$R_\mathrm{{pole}}/r_p$ = {r_scales[r6_min_ind]}",
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
        label=fr"$R_\mathrm{{pole}}/r_p$ = {r_scales[r10_min_ind]}",
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
        label=fr"$R_\mathrm{{pole}}/r_p$ = {r_scales[r14_min_ind]}",
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


# Make r-plots for various lengths
esq_lengths = data["Lesq[mm]"].unique()
for ell in esq_lengths:
    this_data = data[data["Lesq[mm]"] == ell]
