import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import pdb

# Useful constants
mm = 1e-3
kV = 1e3
MHz = 1e6
u_ev = sc.physical_constants["atomic mass unit-electron volt relationship"][0]
Ar_mass = 39.948 * u_ev

# Parameters
f = 15 * MHz
inj_energy = 7 * kV
source_radius = 0.25 * mm
esq_radius = 0.55 * mm
esq_volt = 0.250 * kV
lq = 0.695 * mm

# ------------------------------------------------------------------------------
# This section is a small part dedicated to verifying the equivlance of the
# two matrix traces numerically. The matrices are created one being the
# M1 = Defocus:Short Drift: Focus: Long Drift and the other
# M2 = Focus:Shorft Drift: Focus: Long Drift.
# It is not immediately obvious from matrix theory whether these two matrices
# have equaivlent trace values.
# ------------------------------------------------------------------------------

# Matrix elements
long_drift = 15.24 * mm
short_drift = 1.97 * mm
focus = esq_volt / inj_energy / esq_radius / esq_radius

# Create the transfer matrices: long drift, short drift, focusing, defocus.
Drift_long = np.array([1, long_drift, 0, 1]).reshape(2, 2)
Drift_short = np.array([1, short_drift, 0, 1]).reshape(2, 2)
MFoc = np.array([1, 0, -1 / focus, 1]).reshape(2, 2)
MDefoc = np.array([1, 0, 1 / focus, 1]).reshape(2, 2)

# Create the total transfer matrix.
M1 = MDefoc @ Drift_short @ MFoc @ Drift_long
M2 = MFoc @ Drift_short @ MDefoc @ Drift_long

diff = np.trace(M1) - np.trace(M2)
perturb = long_drift / focus
print("Difference in Respective Traces: {}".format(abs(diff)))

# ------------------------------------------------------------------------------
# This section will create the transfer matrix in x for a given voltage. This
# setting will lead to calculating kappa values for the resulting matrix
# elements. After computing the trace, the stability condition
# cos(sigma_0) = 0.5*Tr(M) will be computed.
# ------------------------------------------------------------------------------
def beta(E, mass=Ar_mass, q=1):
    """Velocity of a particle with energy E."""
    gamma = (E + mass) / mass
    beta = np.sqrt(1 - 1 / gamma / gamma)
    return beta


# Create functions for creating matrices and calculating values
def calc_kappa(voltage, energy=7000, radius=0.55e-3):
    energy_ratio = voltage / energy
    return energy_ratio / pow(radius, 2)


def thin_focus(kappa, len_elemnt):
    m11 = 1
    m12 = 0
    m21 = kappa * len_elemnt
    m22 = 1

    M = np.array([m11, m12, -1 * m21, m22]).reshape(2, 2)
    return M


def thick_focus(kappa, len_elemnt):

    arg = np.sqrt(kappa) * len_elemnt
    m11 = np.cos(arg)
    m12 = np.sin(arg) / np.sqrt(kappa)
    m21 = -np.sqrt(kappa) * np.sin(arg)
    m22 = np.cos(arg)

    M = np.array([m11, m12, m21, m22]).reshape(2, 2)
    return M


def thin_defocus(kappa, len_elemnt):
    m11 = 1
    m12 = 0
    m21 = kappa * len_elemnt
    m22 = 1

    M = np.array([m11, m12, m21, m22]).reshape(2, 2)
    return M


def thick_defocus(kappa, len_elemnt):
    arg = np.sqrt(abs(kappa)) * len_elemnt

    m11 = np.cosh(arg)
    m12 = np.sinh(arg) / np.sqrt(abs(kappa))
    m21 = -np.sqrt(abs(kappa)) * np.sinh(arg)
    m22 = np.cosh(arg)

    M = np.array([m11, m12, m21, m22]).reshape(2, 2)
    return M


def drift(length):
    m11 = 1
    m12 = length
    m21 = 0
    m22 = 1

    M = np.array([m11, m12, m21, m22]).reshape(2, 2)
    return M


def stable_cond(matrix):
    trace = np.trace(matrix)
    sigma_0 = np.arccos(trace / 2) * 180 / np.pi

    return sigma_0


def calc_thin_trace(
    voltage,
    L=15.24 * mm,
    ell=1.97 * mm,
    length_quad=0.695 * mm,
    rp=0.55 * mm,
    energy=7 * kV,
):
    kappa = voltage / energy / pow(rp, 2)
    f = 1 / kappa / length_quad
    tr = 2 - ell * L / pow(f, 2)
    return tr


def thick_phase_adv(kappa, Lp=18.6 * mm, lq=0.695 * mm):
    eta = 2 * lq / Lp
    theta = eta * np.sqrt(abs(kappa)) * Lp / 2

    term1 = np.cos(theta) * np.cosh(theta)

    term2a = (1 - eta) * theta
    term2b = np.cos(theta) * np.sinh(theta) - np.sin(theta) * np.cosh(theta)
    term2 = term2a * term2b / eta

    term3a = (1 - eta) ** 2
    term3b = theta ** 2 * np.sin(theta) * np.sinh(theta)
    term3c = 2 * pow(eta, 2)
    term3 = term3a * term3b / term3c

    complete_term = term1 + term2 - term3
    sigma = np.arccos(complete_term) * 180 / np.pi

    return sigma


# ------------------------------------------------------------------------------
# This section is devoted to analyzing the stability conditions. The breadkdown
# voltage is calculated using the maximum E-field (magnitude). From here,
# the voltage settings are created from some minimum value to breakdown and then
# the corresponding foucsing strengths are calculated. The rest is analysis and
# plotting for whatever is desired.
# ------------------------------------------------------------------------------

# Calculate breakdown using max E-field 3kV/mm
brkdown = 3 / mm * esq_radius / 2 / np.sqrt(2)
voltage_list = np.linspace(0.01, brkdown, 1000) * kV
kappa_list = calc_kappa(voltage_list)

# Create drift length corresponding to notes with matrix permutations
drift1 = 15.24 * mm
drift2 = 1.97 * mm

# Initialize containers to hold trace values and phase advance values.
traces = np.array([calc_thin_trace(V) for V in voltage_list])
sigma_list = np.zeros(len(kappa_list))

# This first loop uses the thin lenses for the transfer matrices. The second
# the thick lenses
do_thin_lens = False
if do_thin_lens:
    for i, k in enumerate(kappa_list):
        d2 = drift(drift2)
        d1 = drift(drift1)
        MF = thin_focus(k, lq)
        MD = thin_defocus(k, lq)
        M = MD @ d2 @ MF @ d1

        this_sigma = stable_cond(M)
        sigma_list[i] = this_sigma

do_thick_lens = False
if do_thick_lens:
    for i, k in enumerate(kappa_list):
        d1 = drift(drift1)
        d2 = drift(drift2)
        MF = thick_focus(k, len_elemnt=lq)
        MD = thick_defocus(k, len_elemnt=lq)
        M = MD @ d2 @ MF @ d1

        this_sigma = stable_cond(M)
        sigma_list[i] = this_sigma

        # Compute sigma from analytic result
        analytic_sigma = thick_phase_adv(k, lq=lq)
        # print("Numeric - Analytic: {}".format(abs(this_sigma - analytic_sigma)))


# find voltages correspoinding to between 60 and 80º.
sigma_list = thick_phase_adv(kappa_list, lq=lq)
indices = np.where((sigma_list > 60) & (sigma_list < 80))[0]
vmin, vmax = voltage_list[indices][0], voltage_list[indices][-1]
forty_ind = np.where((sigma_list > 40))[0][0]
vforty = voltage_list[forty_ind]

make_plots = False
if make_plots:
    fig, ax = plt.subplots()
    ax.set_title(
        r"Phase Advance Given ESQ Voltage Setting ($V_2 = -V_1$)", fontsize="medium"
    )
    ax.plot(voltage_list / kV, sigma_list)
    # ax.axvline(x=brkdown, ls="--", lw=1, label=r"Breakdown Voltage $= 3 \times 2r_p$[KV]")
    ax.axvline(
        x=vforty / kV,
        ls="--",
        lw=1,
        c="g",
        label=r"$\sigma_0 = 40^\circ$, V={:.3f}KV".format(vforty / kV),
    )
    ax.axvline(
        x=vmin / kV,
        ls="--",
        lw=1,
        c="g",
        label=r"$\sigma_0 = 60^\circ$, V={:.3f}KV".format(vmin / kV),
    )
    ax.axvline(
        x=vmax / kV,
        ls="--",
        lw=1,
        c="g",
        label=r"$\sigma_0 = 80^\circ$, V={:.3f}KV".format(vmax / kV),
    )
    ax.axvline(
        x=brkdown,
        ls="--",
        lw=1,
        c="r",
        label=r"$V_\mathrm{{max}}$, V={:.3f}KV".format(brkdown),
    )
    ax.set_xlabel("Voltage [kV]")
    ax.set_ylabel(r"Phase Advance $\sigma_0^\circ$")
    ax.legend()
    plt.savefig("phase_adv_voltage.png", dpi=400)
    plt.show()

# ------------------------------------------------------------------------------
# This section will alter the drift length in d1. d2 is the spacing distance
# between the ESQs in accord with the notes. This spacing is for equal filling
# of the ESQs in the drift space. So, for now, this wont be altered. Although,
# this will change with acceleration. But for now, I can see how changing a drift
# affects the stablity.
# ------------------------------------------------------------------------------
# Calculate gap center distancs of wafers from previous gap using energy
# gain from previous gap. First gap is always placed at zero. So the loop
# starts at finding the distances of the second gap.
Ngaps = 6
E = inj_energy
voltage = 7 * kV
wafers = [0]
V = np.ones(Ngaps) * voltage
Egain = E
for i in range(1, Ngaps):
    # Update energy gain from previous gap
    Egain += V[i - 1]
    # Calculate beta using energy gain from previous gap
    b = beta(Egain, Ar_mass, q=1)
    # Calculate constant using frequency if next gap
    gap_cent_dist = sc.c * b / 2 / f
    wafers.append(gap_cent_dist)

# Distance beteween each sequential gap center. Note, the actualy position is
# the cumulative sum. But here, we only care about the gap-gap distance.
pos = np.array(wafers).cumsum()

# Loop through pos array and use position as drift.
