"""Script holds parameters for adjusting or calculating"""

import numpy as np
import scipy.constants as SC
from math import sqrt
import argparse

import warp as wp
import MEQALAC.simulation as meqsim

# Define useful constants
amu = SC.physical_constants["atomic mass constant energy equivalent in MeV"][0] * 1e6
uA = 1e-6  # microampers
mm = 1e-3  # milimeters
mrad = 1e-3  # miliradians
permitivity_freespace = 8.85e-12


# ------------------------------------------------------------------------------
# This section will initialize the input paramters used in the calculation.
# This is the portion that should be edited for different cacluated values.
# ------------------------------------------------------------------------------
# Beam Specifications
inj_energy = 7 * wp.kV
charge_state = +1
species = "Ar"
Ar_mass = 39.948 * amu
inj_current = 10 * uA
inj_temperature = 0.1  # eV
inj_radius = 0.25 * mm
inj_xprime = 5 * mrad
inj_yprime = -5 * mrad
dext = 11 * mm

# ------------------------------------------------------------------------------
# This section creates the functions for calcated the generalied perveance Q,
# RMS-edge emittance, and the child langumuir current density.
# ------------------------------------------------------------------------------
def calc_perveance(current, energy, mass, return_density=True, charge_state=+1):
    """Calculate perveance Q in KV-envelope equation

    Function to calculate the perveance Q in the KV-envelope equation. The
    input paramters are set to be the most general but more parameters can be
    added. The perveance is a nasty mess of values to compute and thus, this
    function splits up the calculation to avoid errors.
    - Term1 is responsible for calculating the constants in the formula.
    - Term2 is responsible for calculating the current density.
    - Term3 is responsible for the denominator portion consiting of mass, betta,
      and gamma.

    Parameters:
    -----------
    current : float
        Beam current. Assumed to be in Amperes

    energy : float
        Beam energy. Assumed to be in eV

    mass : float
        Mass of the particle species the beam is composed of. Assumed to be in
        atomic units

    return_density : bool
        If true, the function will also return the charge density.

    charge_state = int
        Charge state of particles in beams. defaulted to singly charged ions.


    Returns:
    ---------
    perveance : float
        The dimensionless constant Q called the perveance.

    term2 : float
        Current density of beam.

    """

    # Calculate term1 consisting of mostly constants
    term1 = wp.echarge * charge_state / 2 / np.pi / permitivity_freespace

    # Calculate variables gamma, beta
    gamma = (energy + mass) / mass
    beta = meqsim.beta(energy, mass=Ar_mass, q=charge_state)

    # Calculate term2 current density
    term2 = current / beta / SC.c

    # Calculate term3. Convert mass to Kg
    mass = mass * wp.jperev / SC.c / SC.c
    term3 = mass * pow(gamma, 3) * pow(beta * SC.c, 2)

    # Calculate perveance using terms
    perveance = term1 * term2 / term3

    if return_density:
        return perveance, term2
    else:
        return perveance


def calc_emittance(inj_energy, inj_temperature, inj_radius):
    """Function will calculate emittance of beam at injection.

    The function assumes that the transverse emittances are equal and that
    initial transverse correlations between positions and angle are zero. For
    example <xx'> = 0.

    Parameters
    ----------
    inj_radius : float
        The raidus of the aperture at injection assumed to be in units of
        meters.

    inj_temperature : float
        Temperature of beam at injection assumed to be in units of eV.

    inj_energy : float
        Energy of beam at injectioin assumed to be in units of eV.

    Returns
    -------
    emittance : float
        Emittance of beam at injection given in units of mm-mrad.

    """

    # Evaluate prefactor
    prefact = sqrt(2)

    # Calculate emittance
    emittance = prefact * inj_radius * sqrt(inj_temperature / inj_energy)

    return emittance


def calc_JCL_ion(energy=7000, dext=0.25e-3, Z=1.0, A=18.0):
    """Calculate child langumuir current density.

    Parameters
    ----------
    energy: float
         Extraction voltage in units of V

    dext: float
         The length of the extraction gap

    Z: float
         Charge state of ions

    A: float
         The atomic number of the ion in question. Defaulted to Argon.
    """

    const = 5.44e-8
    term1 = np.sqrt(Z / A)
    term2 = pow(energy, 1.5) / pow(dext, 2)

    value = const * term1 * term2

    return value


# ------------------------------------------------------------------------------
# This section will start the data output. The paramters for inputs will be
# outputed to the screen in respective units. Then, the function calls will be
# used to calculate and output paramters.
# ------------------------------------------------------------------------------
# Print input paramters.
print("")
print("Injection energy: {:.3f} [KV]".format(inj_energy / wp.kV))
print("Species: {}".format(species))
print("Charge State: {}".format(charge_state))
print("Mass: {:.4f} [MeV]".format(Ar_mass / 1e6))
print("Injection Current: {:.4f} [micro-Amps]".format(inj_current / uA))
print("Ion Temp at Injection: {:.3f} [eV]".format(inj_temperature))
print("Source Radius: {:.2f} [mm]".format(inj_radius / mm))
print("x-angle at Source: {:.3f} [mrad]".format(inj_xprime / mrad))
print("y-angle at Source: {:.3f}[mrad]".format(inj_yprime / mrad))

# Print calcuated paramter values
Q, charge_density = calc_perveance(
    current=inj_current,
    energy=inj_energy,
    mass=Ar_mass,
    return_density=True,
    charge_state=1,
)
rms_emit = calc_emittance(
    inj_energy=inj_energy, inj_temperature=inj_temperature, inj_radius=inj_radius
)
jcl = calc_JCL_ion(energy=inj_energy, dext=dext)

print("")
print(42 * "=")
print("Perveance Q: {:.5e}".format(Q))
print("RMS-edge emittance: {:.4f} [mm-mrad]".format(rms_emit / 1e-6))
print("CL Current Density: {:.4f} [micro-Amps / mm^2]".format(jcl))
print(42 * "=")


# Create dictionary of parameters
param_dict = {
    "species": species,
    "mass": Ar_mass,
    "aperture_rad": inj_radius,
    "inj_energy": inj_energy,
    "inj_current": inj_current,
    "inj_radius": inj_radius,
    "Q": Q,
    "emittance": rms_emit,
    "charge_density": charge_density,
    "inj_xprime": inj_xprime,
    "inj_yprime": inj_yprime,
}
