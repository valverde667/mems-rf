# This script will load the accelerating gaps and then fit a function to the
# time varying RF gap field.

import numpy as np
import scipy
import scipy.constants as SC
from scipy.optimize import curve_fit
from scipy.special import jv
import matplotlib.pyplot as plt
import os
import pdb

import warp as wp
from warp.utils.timedependentvoltage import TimeVoltage

# different particle masses in eV
# amu in eV
amu = SC.physical_constants["atomic mass constant energy equivalent in MeV"][0] * 1e6

Ar_mass = 39.948 * amu
He_mass = 4 * amu
p_mass = amu
kV = 1000.0
keV = 1000.0
MHz = 1e6
cm = 1e-2
mm = 1e-3
ns = 1e-9  # nanoseconds
uA = 1e-6
twopi = 2 * np.pi


def beta(E, mass=Ar_mass, q=1, nonrel=True):
    """Velocity of a particle with energy E."""
    if nonrel:
        beta = np.sqrt(2 * E / mass)
    else:
        gamma = (E + mass) / mass
        beta = np.sqrt(1 - 1 / gamma / gamma)

    return beta


def calc_pires(energy, freq, mass=Ar_mass, q=1):
    """RF resonance condition in pi-mode"""
    beta_lambda = beta(energy, mass=mass, q=q) * SC.c / freq
    return beta_lambda / 2


def gap_voltage(t):
    """ "Sinusoidal function of the gap voltage"""

    v = 7 * kV * np.cos(2 * np.pi * 13.6 * MHz * t)

    return v


def neg_gap_voltage(t):
    """ "Sinusoidal function of the gap voltage"""

    v = 7 * kV * np.cos(2 * np.pi * 13.6 * MHz * t)

    return v


def create_gap(
    cent,
    left_volt,
    right_volt,
    width=2 * mm,
    length=0.035 * mm,
    rin=0.55 * mm,
    rout=0.75 * mm,
):
    """Create an acceleration gap consisting of two wafers."""

    left_wafer = wp.Annulus(
        rmin=rin, rmax=rout, length=length, zcent=cent - width / 2, voltage=left_volt
    )

    right_wafer = wp.Annulus(
        rmin=rin, rmax=rout, length=length, zcent=cent + width / 2, voltage=right_volt
    )
    gap = left_wafer + right_wafer
    return gap


# Find gap positions. The gap positions will be calculated for 12 gaps giving
# three lattice periods.
rout = 0.75 * mm
rin = 0.50 * mm
length = 0.035 * mm
gap_width = 2.0 * mm
zcenter = abs(0.0 - gap_width / 2.0)
f = 13.6 * MHz
Ng = 2
Vg = 7.0 * kV
E_DC = Vg / gap_width
dsgn_phase = -0.0 * np.pi
gap_cent_dist = []
Einit = 7.0 * keV
rf_wave = beta(Einit) * SC.c / f
fcup_dist = 50.0 * mm
Energy = [Einit]

for i in range(Ng):
    this_dist = calc_pires(Energy[i], freq=f)
    gap_cent_dist.append(this_dist)
    Egain = Vg * np.cos(dsgn_phase)  # Max acceleration
    Energy.append(Egain + Energy[i])

Energy = np.array(Energy)

# Real gap positions are the cumulative sums
gap_cent_dist = np.array(gap_cent_dist)
gap_centers = gap_cent_dist.cumsum()
# Shift gaps by drift space to allow for the field to start from minimum and climb.
zs = beta(Einit) * SC.c / 2.0 / np.pi / f * (dsgn_phase + np.pi)
gap_centers += zs

print("--Gap Centers")
print(gap_centers / mm)

# Create beam
beam = wp.Species(type=wp.Argon, charge_state=+1, name="Argon Beam")
beam.a0 = 0.25 * mm
beam.b0 = 0.25 * mm
beam.emit = 1.344e-6
beam.ap0 = 0.0
beam.bp0 = 0.0
beam.ibeam = 10 * uA
beam.beam = 0.0
beam.ekin = 7.0 * keV
wp.derivqty()

beam.vthz = 0.5 * beam.vbeam * beam.emit / wp.sqrt(beam.a0 * beam.b0)

# Set up the 3D simulation

# Create mesh
wp.w3d.xmmin = -2.5 * mm
wp.w3d.xmmax = 2.5 * mm
wp.w3d.nx = 60

wp.w3d.ymmin = -2.5 * mm
wp.w3d.ymmax = 2.5 * mm
wp.w3d.ny = 60

# use gap positioning to find limits on zmesh. Add some spacing at end points.
# Use enough zpoint to resolve the wafers. In this case, resolve with 2 points.
wp.w3d.zmmin = -rf_wave / 2
wp.w3d.zmmax = gap_centers[-1] + fcup_dist
wp.w3d.nz = round(1 * (wp.w3d.zmmax - wp.w3d.zmmin) / length)
wp.top.dt = (wp.w3d.zmmax - wp.w3d.zmmin) / wp.w3d.nz / beam.vbeam

# Add boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.periodic


wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb
wp.top.pboundxy = wp.absorb


beam.zimin = -rf_wave / 1.95
beam.zimax = rf_wave / 2

wp.top.npmax = 10000

wp.w3d.distrbtn = "uniform"
wp.w3d.distr_l = "uniform"


wp.w3d.cigarld = True
beam.straight = 0.5
wp.fstype = 0


wp.w3d.l4symtry = True
solver = wp.MRBlock3D()
wp.registersolver(solver)


# Create accleration gaps with correct coordinates and settings. Collect in
# list and then loop through and install on the mesh.
conductors = []
for i, cent in enumerate(gap_centers):
    if i % 2 == 0:
        this_cond = create_gap(cent, left_volt=0, right_volt=Vg)
    else:
        this_cond = create_gap(cent, left_volt=Vg, right_volt=0)

    conductors.append(this_cond)

for cond in conductors:
    wp.installconductor(cond)

# Perform initial field solve for mesh.
wp.package("w3d")
wp.generate()
stop

# Collect data from the mesh and initialize useful variables.
z = wp.w3d.zmesh
steps = 1
time = np.zeros(steps)
Ez_array = np.zeros((steps, len(z)))
Ez0 = wp.getselfe(comp="z")[0, 0, :]
phi0 = wp.getphi()[0, 0, :]

# Save arrays
np.save("potential_array", phi0)
np.save("field_array", Ez0)
np.save("gap_centers", gap_centers)
np.save("zmesh", z)

# load arrays
# Ez0 = np.load('field_array.npy')
# phi0 = np.load('potential_array.npy')
# z = np.load('zmesh.npy')


# Plot potential and electric field (z-direction) on-axis.
fig, ax = plt.subplots()
ax.plot(z / mm, phi0 / kV)
ax.set_xlabel("z [mm]")
ax.set_ylabel("Potential [kV]")
ymin, ymax = ax.get_ylim()
for cent in gap_centers:
    left = cent - length / 2
    right = cent + length / 2
    ax.axvline(x=left / mm, c="gray", lw=0.7)
    ax.axvline(x=right / mm, c="gray", lw=0.7)
    ax.fill_between([left / mm, right / mm], [ymin, ymax], color="grey", alpha=0.5)
ax.axhline(y=0, c="k", lw=1)

fig, ax = plt.subplots()
ax.plot(z / mm, Ez0 / E_DC)
ax.set_xlabel("z [mm]")
ax.set_ylabel("On-axis E-field E(x=0, y=0, z) [V/m]")
ymin, ymax = ax.get_ylim()
for cent in gap_centers:
    left = cent - length / 2
    right = cent + length / 2
    ax.axvline(x=left / mm, c="gray", lw=0.7)
    ax.axvline(x=right / mm, c="gray", lw=0.7)
    ax.fill_between([left / mm, right / mm], [ymin, ymax], color="grey", alpha=0.5)
ax.axhline(y=0, c="k", lw=1)
plt.show()


# Warp plotting for verification that mesh and conductors were created properly.
warpplots = True
if warpplots:
    wp.setup()
    wp.pfzx(fill=1, filled=1)
    wp.fma()

for i in range(steps):
    Ez = wp.getselfe(comp="z")[0, 0, :]
    Ez_array[i, :] = Ez
    time[i] = wp.top.time
    wp.step()

np.save("Ez_gap_field_151", Ez_array)
np.save("zmesh", z)
np.save(f"time_{steps}", time)

fig, ax = plt.subplots()
ax.axhline(y=1, c="r", lw=1, label="Average DC Field")
ax.plot(
    z / mm, Ez_array[0, :] / E_DC, c="k", label=f"Time: {time_array[0]/ns:.2f} [ns]"
)
ax.set_xlabel("z [mm]")
ax.set_ylabel(fr"On-axis Electric field $E_z(r=0, z,t)/E_{{dc}}$ [kV/mm]")
ax.axvline(x=-zcenter / mm, c="gray", lw=1)
ax.axvline(x=zcenter / mm, c="gray", lw=1)
ax.legend()
plt.show()

stop
# Post process
# Ez_array = np.load("Ez_gap_field_151.npy")
# time_array = np.load(f"time_{steps}.npy")
# z_array = np.load("zmesh.npy")


# fig,ax = plt.subplots()
# ax.axhline(y=1 , c='r', lw=1, label='Average DC Field')
# ax.plot(z/mm, Ez_array[0, :]/E_DC, c='k', label=f'Time: {time_array[0]/ns:.2f} [ns]')
# ax.plot(z/mm, Ez_array[20, :]/E_DC, c='b', label=f'Time: {time_array[20]/ns:.2f} [ns]')
# ax.plot(z/mm, Ez_array[35, :]/E_DC, c='g', label=f'Time: {time_array[35]/ns:.2f} [ns]')
# ax.set_xlabel('z [mm]')
# ax.set_ylabel(fr"On-axis Electric field $E_z(r=0, z,t)/E_{{dc}}$ [kV/mm]")
# ax.axvline(x=-zcenter/mm, c='g', lw=1)
# ax.axvline(x=zcenter/mm, c='g', lw=1)
# ax.legend()
# plt.show()
