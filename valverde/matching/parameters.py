"""Script holds parameters for adjusting or calculating"""

import numpy as np
import scipy.constants as SC
from math import sqrt

import warp as wp
import MEQALAC.simulation as meqsim

# Define useful constants
amu = SC.physical_constants["atomic mass constant energy equivalent in MeV"][0] * 1e6
uA = 1e-6  # microampers
mm = 1e-3  # milimeters
permitivity_freesapce = 8.85e-12

# Beam Specifications
inj_energy = 8 * wp.kV
charge_state = +1
species = "Ar"
Ar_mass = 40 * amu
inj_current = 10 * uA
inj_radius = 0.5 * mm


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
    term1 = wp.echarge * charge_state / 2 / np.pi / permitivity_freesapce

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


def calc_emittance(inj_radius, inj_temperature, inj_energy):
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
    prefact = 2 * sqrt(2)

    # Calculate emittance
    emittance = prefact * inj_radius * sqrt(inj_temperature / inj_energy)

    return emittance


# Evaluate parameters
perveance, charge_density = calc_perveance(inj_current, inj_energy, Ar_mass,)

emittance = calc_emittance(inj_radius, inj_temperature, inj_energy)

# Create dictionary of parameters
param_dict = {
    "species": species,
    "mass": mass,
    "inj_energy": inj_energy,
    "inj_current": inj_current,
    "inj_radius": inj_radius,
    "Q": perveance,
    "emittance": emittance,
    "charge_density": charge_density,
}

print("Perveance and lambda is:")
print(calc_perveance(injection_current, injection_energy, Ar_mass))
