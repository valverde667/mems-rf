import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sc
import scipy.optimize as optimize
import csv
import pdb

savepath = "/Users/nickvalverde/Desktop/"
# Useful constants
mm = 1e-3
kV = 1e3
MHz = 1e6
u_ev = sc.physical_constants["atomic mass unit-electron volt relationship"][0]
Ar_mass = 39.948 * u_ev

# Parameters
Vmax = 0.583 * kV
f = 13.56 * MHz
inj_energy = 7 * kV
source_radius = 0.25 * mm
esq_radius = 0.55 * mm
esq_volt = 0.250 * kV
gap_volt = 7.0 * kV
lq = 1.278 * mm
gap = 2 * mm

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


def thin_accel_gap(volt, Energy, g, phi, mass=Ar_mass, transit_factor=1, q=1):
    """Build a thin RF-gap matrix element.

    Modeling the RF acceleration gap as a thin element requires knowledge of the
    transit time factor which should be solved numerically. However, a parametric
    search can be done by using different values. Also, it is not appropriate to
    model the RF gaps as thin in the start of the lattice since the particles
    spend a non-neglible amount of time traversing the gap. However, some
    intuition can still be gained. This function assumes non-relativistic and
    sets gamma=1.

    Parameters
    ----------
    volt : float
        Voltage setting on gap.

    Energy : float
        Energy of particle before entering the gap in eV.

    g : float
        Thickness of RF-gap.

    phi : float
        Phase of arrival in gap. Here the gap field is at maximum when phi=0.

    mass : float
        Mass of simulated ion in eV.

    transit_factor: float
        The value of the transit time factor ranging from [0, 1].

    q : int
        Charge state of simulated ion.

    Returns
    -------
    M : ndarray
       Returns 2x2 transfer matrix for thin RF acceleration Gap

    """

    # Calculate focal length. See Wangler 7.11.
    b = beta(Energy, mass=mass, q=q)
    E0 = volt * g
    numerator = pow(b, 2) * mass
    denom = np.pi * q * E0 * transit_factor * np.sin(-phi)

    focal = numerator / denom

    # Build transport matrix
    m11 = 1.0
    m12 = 0.0
    m21 = 1.0 / focal
    m22 = 1.0
    M = np.array([m11, m12, m21, m22]).reshape(2, 2)

    return M


def build_M(centers, enrgy_pairs, voltage=384.667, g=2 * mm, lq=1.28 * mm):
    """Temporary function to build lattice and find phase advance."""

    # Calculate distances from center to center gaps distances
    cent12, cent23 = centers[0], centers[1]
    w = (cent23 - g - 2 * lq) / 3
    d = cent12 + g + 2 * w
    eta = 2 * lq / (cent23 - g)
    if eta > 1:
        print("Max occupancy reached for ESQ length.")

    k = calc_kappa(voltage, enrgy_pairs[-1])

    # Build First dirft
    O = drift(d)
    D = thick_defocus(k, lq)
    Ow = drift(w)
    F = thick_focus(k, lq)

    M = D @ Ow @ F @ O
    condition = np.trace(M) / 2

    return condition


def volt_root(voltage, centers, energy_pairs, target, lq, g=2 * mm, verbose=True):
    # Calculate distances from center to center gaps distances
    cent12, cent23 = centers[0], centers[1]
    inter_esq_drift = 1.8 * mm
    w = (cent23 - g - 2 * lq - inter_esq_drift) / 2
    d = w + cent12 + g + w
    eta = 2 * lq / (cent23 - g)
    if eta > 1:
        print("Max occupancy reached for ESQ length.")

    k = calc_kappa(voltage, energy_pairs[-1])

    # Build Transfer matrix
    O = drift(d)
    D = thick_defocus(k, lq)
    Ointer = drift(inter_esq_drift)
    F = thick_focus(k, lq)

    M = D @ Ointer @ F @ O
    target_zeroed = target - np.trace(M) / 2

    return target_zeroed


# ------------------------------------------------------------------------------
#                   Stability modeling wihout RF gap acceleration elements
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
Ngaps = 100 * 3
E = inj_energy
voltage = 7 * kV
f = 13.56 * MHz
wafers = []
energies = [
    7 * kV,
]
V = np.ones(Ngaps) * voltage
Egain = E
for i in range(1, Ngaps):
    # Update energy gain from previous gap
    Egain += V[i - 1]
    energies.append(Egain)
    # Calculate beta using energy gain from previous gap
    b = beta(Egain, Ar_mass, q=1)
    # Calculate constant using frequency if next gap
    gap_cent_dist = sc.c * b / 2 / f
    wafers.append(gap_cent_dist)

# Distance beteween each sequential gap center. Note, the actualy position is
# the cumulative sum. But here, we only care about the gap-gap distance.
wafer_copy = np.array(wafers.copy())
energies = np.array(energies[1:])
pos = np.array(wafers).cumsum()

# Loop through pos array and use position as drift.
wafer_copy / mm
pos / mm
len(wafer_copy)
energies
dpairs = np.array(
    [(wafer_copy[i], wafer_copy[i + 1]) for i in range(0, len(wafer_copy) - 1, 2)]
)
Epairs = energies[:-1].reshape(len(dpairs), 2)
dpairs / mm
sigma_list = []
target = np.cos(80 * np.pi / 180)
length_multiplier = 5
eff_lengths = np.array([1.1977, 1.816, 2.507, 3.1996, 3.895]) * mm
length_quad = eff_lengths[length_multiplier - 1]
Vcap = 0.75 * Vmax
g = 2 * mm

# Main Loop. Loop through energy and center pairs and evaluate desired value.
# The volt_root function is used to find the needed voltage to maintain
# set phase advance. All other quanitities are geometrically determined except
# for the inter ESQ spacing. This is held fixed at 1.8mm, a value that was found
# minimize the mutal zeroing of the ESQ gradient at the symmetry point betweeen
# the two. The quads could be kept further apart, but, it is better for transport
# to keep the optics close to eachother. At least, it is thought to be but there
# is some investigation that can be done there.
with open(savepath + "lattice_stability.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(
        [
            "Lp",
            "En",
            "En+1",
            "n-ell",
            "TotalSpace",
            "eta",
            "w",
            "theta",
            "cell-length",
            "Voltage",
            "Freq",
        ]
    )
    for i, (energy, cent) in enumerate(zip(Epairs[:], dpairs[:])):
        # Calculate information
        cent12, cent23 = cent[0], cent[1]
        inter_esq_drft = 1.8 * mm
        w = (cent23 - g - 2 * length_quad - inter_esq_drft) / 2
        d = w + cent12 + g + w
        cell_length = cent12 + cent23
        esq_space_tot = cent23 - g
        eta = 2 * length_quad / (esq_space_tot)
        if eta > 1:
            print("Max occupancy reached for ESQ length.")
            break

        # Initialize solver with voltage as knob and additional arguments being
        # the gap-gap centers and gap-gap ion energy
        init_Vguess = np.array([300])
        sol = optimize.root(
            volt_root,
            init_Vguess,
            args=(cent, energy, target, length_quad),
            method="hybr",
        )
        volt_found = sol.x[0]

        # Set switches to kill loop. The first would be exceeding the breakdown
        # limit. The next cap is falling below the eta cap. This means that there
        # is much drift space where the ESQs live and the Frequency should be
        # increased to decrease the wasted drift space.
        if volt_found > Vcap:
            print("Max Volt exceeded. Lp: ", i)
            break
        if eta < 0.3:
            print("Eta cap exceeded. Eta: ", eta)
            break
        else:
            print("Energy: ", energy)
            print("Voltage Found: ", volt_found)
            kappa = calc_kappa(volt_found, energy[-1])
            theta = np.sqrt(kappa) * length_quad
            writer.writerow(
                [
                    i + 1,
                    energy[0] / kV,
                    energy[1] / kV,
                    length_multiplier,
                    esq_space_tot / mm,
                    eta,
                    w / mm,
                    theta,
                    cell_length / mm,
                    volt_found / kV,
                    f / MHz,
                ]
            )

# ------------------------------------------------------------------------------
#                    Including Acceleration
# This section will build upon the previous section by adding the acceleration
# optic. The optic is taken to be a thin lens kick which is not the most accurate
# depiction at injection. The transit time factor is chosen rather than
# calculated and can be varied to give a parametric scan for different transit
# time factors. The same can be done for the arrival phase which should be
# constricted to the half-open interval [np.pi, 0).
# ------------------------------------------------------------------------------
