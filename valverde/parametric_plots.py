import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as SC


# units and constants
mm = 1e-3
kV = 1e3
mrad = 1e-3
micro = 1e-6
amu = SC.physical_constants["atomic mass constant energy equivalent in MeV"][0] * 1e6
U = 1.6605e-27  # kg
r_source = 0.25 * mm
Isource = 10 * micro
Vsource = 7 * kV
dextract = 100 * mm
Tsource = 5600  # degress kelvin
kBoltz = 8.617e-5  # eV/Kelvin
Ar_mass = 39.948 * U

# Argon charge state and atomic number
Z = 1
A = 18


def current_density(
    Volt_extract=Vsource, extract_gap=dextract, atom_num=A, charge_state=Z
):

    # There is an overall constant that is associated for ions. Different for
    # electrons.
    overall_const = 5.44e-8

    # Atomic term
    atom_term = np.sqrt(charge_state / atom_num)

    # Parameter term
    param_term = pow(Volt_extract, 3 / 2) / pow(extract_gap, 2)

    return overall_const * atom_term * param_term


def rms_emit(source_radius=r_source, temp=Tsource, energy=Vsource):
    global kBoltz

    term1 = np.sqrt(2) * source_radius
    term2 = np.sqrt(kBoltz * temp / energy)

    return term1 * term2


def perveance(current, charge_state=Z, mass=Ar_mass, energy=Vsource * SC.e):
    pref = Z * SC.e / 2 / np.pi / mass / SC.epsilon_0
    term = current * (mass / 2 / energy) ** (3 / 2)

    return pref * term


# Evaluate some useful geometric values
source_area = np.pi * r_source * r_source

# testing
print("RMS-emittance: ", rms_emit(0.55 * mm))
print("Child Current: ", current_density() * source_area)
print("Q: ", perveance(10 * micro))
dddd
duppr = 30 * mm
dlow = 10 * mm

d = np.linspace(dlow, duppr, 970)
I_child = current_density(extract_gap=d) * source_area
Q_child = perveance(I_child)

fig, (ax, axx) = plt.subplots(ncols=2, sharey=True)
ax.set_title("Acheivable Extraction Current by Child Law", fontsize="small")
ax.set_xlabel("Extraction Gap d [mm]")
ax.set_ylabel(r"$\log(j_0 \pi r_s^2$) [$\mu$A]")
ax.plot(d / mm, np.log(I_child / micro))
ax.axhline(
    y=np.log(I_child.max() / micro),
    c="b",
    ls="--",
    lw=1,
    label=r"I={:.3f}$\mu$A".format(I_child.max() / micro),
)
ax.axhline(
    y=np.log(I_child.min() / micro),
    c="r",
    ls="--",
    lw=1,
    label=r"I={:.3f}$\mu$A".format(I_child.min() / micro),
)
ax.legend()

axx.set_title("Corresponding Perveance", fontsize="small")
axx.set_xlabel(r"Generalized Perveance Q $\times 10^{-5}$")
axx.plot(Q_child / 1e-5, np.log(I_child / micro))
axx.axvline(
    x=Q_child.max() / 1e-5,
    c="b",
    ls="--",
    lw=1,
    label=r"Q={:.3f}$\times 10^{{-5}}$".format(Q_child.max() / 1e-5),
)
axx.axvline(
    x=Q_child.min() / 1e-5,
    c="r",
    ls="--",
    lw=1,
    label=r"Q={:.3f}$\times 10^{{-5}}$".format(Q_child.min() / 1e-5),
)
axx.legend()

plt.tight_layout()
plt.savefig("child_current_Q", dpi=400)
plt.show()


# Emittances for inputs. I want to sample source radii ranging from .55mm to
# 1.5 mm. I'll do 10 steps.
radii = np.linspace(0.01, 0.001, 4) * mm
emits = rms_emit(radii)
for r, e in zip(radii, emits):
    print("{:.3f}: {:.3f}".format(r / mm, e / mm / mrad))
