# Dictionary of various system parameters. This file can be modified and imported
# so that definitions do not need to be repeated and common values can be calc/
# looked up.

import numpy as np
from math import sqrt
import scipy.constants as sc
import warp as wp

# Useful constants
permitivity_freespace = sc.epsilon_0

mm = 1.0e-3
um = 1.0e-6

mrad = 1.0e-3

ms = 1.0e-3
us = 1.0e-6
ns = 1.0e-9

kV = 1.0e3

keV = 1.0e3
meV = 1.0e6
GeV = 1.0e9

MHz = 1.0e6

uA = 1e-6


# ------------------------------------------------------------------------------
#     Function Definitions
# Various quanitites are calculated given parameter values for the system. These
# functions are created here.
# ------------------------------------------------------------------------------
def calc_beta(E, mass, q=1, nonrel=True):
    """Velocity of a particle with energy E."""
    if nonrel:
        sign = np.sign(E)
        beta = np.sqrt(2 * abs(E) / mass)
        beta *= sign
    else:
        gamma = (E + mass) / mass
        beta = np.sqrt(1 - 1 / gamma / gamma)

    return beta


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
        Mass of the particle species the beam is composed of in units of eV

    return_density : bool
        If true, the function will also return the charge density.

    charge_state = int
        Charge state of particles in beams. defaulted to singly charged ions.


    Returns:
    ---------
    perveance : float
        The dimensionless constant Q called the perveance.

    J : float
        Current density of beam.

    """

    # Calculate term1 consisting of mostly constants
    term1 = wp.echarge * charge_state / 2 / np.pi / permitivity_freespace

    # Calculate variables gamma, beta
    gamma = (energy + mass) / mass
    beta = calc_beta(energy, mass=mass, q=charge_state)

    # Calculate current density J
    J = current / beta / sc.c

    # Calculate term3. Convert mass to Kg
    mass = mass * wp.jperev / pow(sc.c, 2)
    term3 = mass * pow(gamma, 3) * pow(beta * sc.c, 2)

    # Calculate perveance using terms
    perveance = term1 * J / term3

    if return_density:
        return perveance, J
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

    # Calculate emittance
    emittance = sqrt(2) * inj_radius * sqrt(inj_temperature / inj_energy)

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
#     System Parameters
# Define quantities for the system such as the system geometry, beam settings,
# injection quantities, etc.
# ------------------------------------------------------------------------------
# Ion Type
ion = "Argon"

# System parameters
gap_voltage = 7.0 * kV
gap_frequency = 13.6 * MHz
gap_distance = 2.0 * mm
source_radius = 0.25 * mm
aperture_radius = 0.55 * mm
extraction_distance = 11 * mm

# Injected Beam Parameters
beam_energy = 7.0 * keV
beam_temp = 0.1  # eV
beam_current = 10.0 * uA
beam_rx = source_radius
beam_ry = source_radius
beam_rpx = 3.78 * mrad
beam_rpy = 3.78 * mrad
mass_kg = wp.periodic_table[ion]["A"] * wp.amu
mass_eV = wp.periodic_table[ion]["A"] * wp.amu * pow(sc.c, 2) / wp.jperev
beam_emittance = calc_emittance(beam_energy, beam_temp, beam_rx)  # 4*RMS-edge emittance
Q, _ = calc_perveance(
    beam_current, beam_energy, mass_eV
)  # Generalized dimensionless perveance

# ------------------------------------------------------------------------------
#     Dictionary of Data
# Create a dictionary to hold all the parameters.
# ------------------------------------------------------------------------------
param_dict = {
    "species": ion,
    "mass": mass_eV,
    "source radius": source_radius,
    "aperture radius": aperture_radius,
    "inj energy": beam_energy,
    "inj current": beam_current,
    "inj radius": beam_rx,
    "Q": Q,
    "emittance": beam_emittance,
    "inj xp": beam_rpx,
    "inj yp": beam_rpy,
}
unit_dict = {
    "mass": (GeV, "(GeV)"),
    "source radius": (mm, "(mm)"),
    "aperture radius": (mm, "(mm)"),
    "inj energy": (keV, "(keV)"),
    "inj current": (uA, "(uA)"),
    "inj radius": (mm, "(mm)"),
    "emittance": (mm * mrad, "(mm-mrad)"),
    "inj xp": (mrad, "(mrad)"),
    "inj yp": (mrad, "(mrad)"),
}


def print_info(params, units):
    """Function to print system parameters

    A quick function that nicely prints out the the system parameters in use.

    Parameters:
    ----------
    params: dict
        Key-value pairs of the various system parameters

    units: dict
        Key-value pairs of the various units for nice outputs. The keys in this
        dicionary should match the applicable keys in the params dict.

    """
    print("")
    print("#--------------- System Parameters")

    # Loop through the params dictionary and print the key with its value.
    # Check if the key is in the units dictionary. If it us, use those units.
    for key in params:
        if key in units.keys():
            print(f"{key:<18}: {params[key]/units[key][0]:.4f} {units[key][-1]}")
        else:
            print(f"{key:<18}: {params[key]}")

    print("#---------------")
