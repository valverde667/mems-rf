# Script is used to develop tools in order to evaluate the gain in kinetic energy.
# The starting point will be evaluating a flat top gap field that oscillates in
# time. The particle is going to be assumed constant velocity within in the gap
# in order to perform the line integral over the spatial coordinate.
# Once this is shown to give good results, the script will then attempt to
# calculated the energy gain for more complicated fields with severe fringes.

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Ellipse
import scipy.constants as SC
import scipy.integrate as integrate
import os


# different particle masses in eV
# amu in eV
amu = SC.physical_constants["atomic mass constant energy equivalent in MeV"][0] * 1e6

Ar_mass = 39.948 * amu
He_mass = 4 * amu
p_mass = amu
kV = 1000
keV = 1000
MHz = 1e6
mm = 1e-3
ns = 1e-9  # nanoseconds
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


def plot_phase(phi, E):
    fig, ax = plt.subplots()
    ax.scatter(phi, E, s=2)
    ax.set_xlabel(r"$\Delta \phi$ [rad]")
    ax.set_ylabel(r"$\Delta {\cal E}$ [kV]")
    plt.tight_layout()
    plt.show()


def rf_volt(t, freq=13.6 * MHz):
    return np.cos(2 * np.pi * freq * t)


# ------------------------------------------------------------------------------
#     Simulation Parameters
# Here the various parameters for the gaps and devices are initialized. The
# particle number and gaps to be used are also set.
# ------------------------------------------------------------------------------
# Simulation Parameters for design particle
design_phase = -0
dsgn_initE = 7 * kV
Np = 10000
gap_width = 2 * mm

# Simulation parameters for gaps and geometries
design_gap_volt = 7 * kV
design_freq = 13.6 * MHz
design_omega = 2 * np.pi * design_freq
E_DC = design_gap_volt / gap_width
Ng = 2
Fcup_dist = 30 * mm
Emax = dsgn_initE + Ng * design_gap_volt * np.cos(design_phase)

# ------------------------------------------------------------------------------
#     Initial Setup
# Start design particle with design phase at z=0. Place first gap so that design
# particle arrives at design phase. Since gaps are initialized to have peak
# voltage at t=0, the first gap is placed such that the field oscillates one full
# period before arrival of design particle. Other particles can be centered on
# the design particle, or distributed from the gap center towards z=0 or z<0.
# ------------------------------------------------------------------------------

# Initialize simulation by setting up first gap commensurate with the design
# particle. Gaps are initialized to have peak output at t=0
dsgn_pos = np.zeros(Ng + 1)
dsgn_E = np.zeros(Ng + 1)
dsgn_time = np.zeros(Ng + 1)

dsgn_pos[0] = 0.0
dsgn_E[0] = dsgn_initE
dsgn_time[0] = 0.0

# Calculate full DC beam length and start position of design particle. The
# convention here is to start with a gaining gap voltage. Since the design particle
# convention enters the gap when the fielding is going from neg -> pos, the
# first gap needs to be placed a RF cycle away
coeff = np.sqrt(2 * dsgn_initE / Ar_mass)

tDC = 1.0 / design_freq
that = 0.5 / design_freq
ts = (np.pi - design_phase) / 2 / np.pi / design_freq

DC_length = coeff * SC.c * tDC
zhat = coeff * SC.c * that
zs = coeff * SC.c * ts
init_gap = zhat + zs

# Instantiate the design particle metrics to first gap
vs_start = coeff * SC.c
ts_start = init_gap / vs_start

# Create simulation particles and initialize data arrays
beta_lambda = vs_start / design_freq
particle_dist = np.linspace(-DC_length / 2, DC_length / 2, Np)

# Create particle arrays to store histories
parts_pos = np.zeros(shape=(Np, Ng + 1))
parts_pos[:, 0] = particle_dist
parts_E = np.zeros(shape=(Np, Ng + 1))
parts_time = np.zeros(shape=(Np, Ng + 1))
parts_E[:, 0] = dsgn_initE

# initialize particles to z=0 along with times
vparts = np.sqrt(2 * parts_E[:, 0] / Ar_mass) * SC.c
time = (parts_pos[:, 0]) / vparts
parts_pos[:, 0] = 0
parts_time[:, 0] = time

# Advance particles to first gap
dsgn_v = np.sqrt(2 * dsgn_E[0] / Ar_mass) * SC.c
dt = (init_gap - dsgn_pos[0]) / dsgn_v
Egain = design_gap_volt * np.cos(design_omega * dt)
dsgn_E[1] = dsgn_E[0] + Egain

# Calculate additional gap centers if applicable
if Ng > 1:
    gap_dist = [init_gap]
    for i in range(1, Ng):
        dsgn_Egain = design_gap_volt * np.cos(-design_phase)
        E = dsgn_initE + i * dsgn_Egain
        this_cent = beta(E) * SC.c / 2 / design_freq
        gap_dist.append(this_cent)

    gap_centers = np.array(gap_dist).cumsum()

# ------------------------------------------------------------------------------
#    Mesh setup
# Here the mesh is setup to place a flat top field centered so that the design
# particle arrives in phase. To do this, the gap is placed an extra rf phase
# away from beta lambda / 2 since the field originally starts out positive.
# ------------------------------------------------------------------------------
z = np.linspace(0.0, gap_centers[-1] + Fcup_dist, 1000)
dz = z[1] - z[0]
Ez0 = z.copy()
if Ng > 1:
    for i, cent in enumerate(gap_centers):
        if i % 2 == 0:
            field_loc = np.where(
                (z >= cent - gap_width / 2) & (z <= cent + gap_width / 2)
            )
            Ez0[field_loc] = design_gap_volt / gap_width
        else:
            field_loc = np.where(
                (z >= cent - gap_width / 2) & (z <= cent + gap_width / 2)
            )
            Ez0[field_loc] = -design_gap_volt / gap_width

else:
    field_loc = np.where(
        (z >= init_gap - gap_width / 2) & (z <= init_gap + gap_width / 2)
    )
    Ez0[field_loc] = design_gap_volt / gap_width

# Plot field
fig, ax = plt.subplots()
ax.set_xlabel("z [mm]")
ax.set_ylabel(r"On-axis E-field $E(r=0, z)/E_{DC}$ [kV/mm]")
ax.plot(z / mm, Ez0 / E_DC)
if Ng > 1:
    for cent in gap_centers:
        ax.axvline(cent / mm, c="grey", lw=1, ls="--")

W_s = np.zeros(len(z))
W = np.zeros(shape=(Np, len(z)))
W_s[0], W[:, 0] = dsgn_initE, dsgn_initE

for i in range(1, len(z)):
    # Do design particle
    this_vs = beta(W_s[i - 1]) * SC.c
    this_dt = dz / this_vs
    dsgn_time[0] += this_dt

    Egain = Ez0[i - 1] * rf_volt(dsgn_time[0], freq=design_freq) * dz

    W_s[i] = W_s[i - 1] + Egain

    # Do other particles
    this_v = beta(W[:, i - 1]) * SC.c
    this_dt = dz / this_v
    parts_time[:, 0] += this_dt

    Egain = Ez0[i - 1] * rf_volt(parts_time[:, 0], freq=design_freq) * dz
    W[:, i] = W[:, i - 1] + Egain


fig, axes = plt.subplots(nrows=2, figsize=(10, 2), sharex=True)
axes[0].hist(W[:, -1] / keV, bins=50)


# ------------------------------------------------------------------------------
#    Warp Fields
# Here the Warp calculated fields and mesh are imported.
# ------------------------------------------------------------------------------
zwarp = np.load("zmesh.npy")
warpEz0 = np.load("field_arrays.npy")
warpphi = np.load("potential_arrays.npy")
warpdz = zwarp[1] - zwarp[0]

warp_Ws = np.zeros(len(zwarp))
warp_W = np.zeros(shape=(Np, len(zwarp)))
warp_Ws[0], warp_W[:, 0] = dsgn_initE, dsgn_initE
ts = np.zeros(len(zwarp))

# Initialize particle times by distributing particles uniformly around zero.
# Then advance particles forward and backward in time so that all particles
# advance form z=0
init_time = (0 - particle_dist) / vparts
t = np.zeros(shape=(Np, len(zwarp)))
t[:, 0] = init_time
for i in range(1, len(zwarp)):
    # Do design particle
    this_vs = beta(warp_Ws[i - 1]) * SC.c
    this_dt = warpdz / this_vs
    ts[i] = ts[i - 1] + this_dt

    Egain = warpEz0[i - 1] * rf_volt(ts[i - 1], freq=design_freq) * warpdz

    warp_Ws[i] = warp_Ws[i - 1] + Egain

    # Do other particles
    this_v = beta(warp_W[:, i - 1]) * SC.c
    this_dt = warpdz / this_v
    t[:, i] = t[:, i - 1] + this_dt

    Egain = warpEz0[i - 1] * rf_volt(t[:, i - 1], freq=design_freq) * warpdz
    warp_W[:, i] = warp_W[:, i - 1] + Egain

axes[1].hist(warp_W[:, -1] / keV, bins=50)
axes[1].set_xlabel("Final Kinetic Energy [keV]")


fig = plt.figure()
xlim = (1.15 * zwarp.min() / mm, 1.05 * zwarp.max() / mm)
ylim = (-1.05 * abs(warpEz0).max() / kV * mm, abs(warpEz0).max() * mm / kV * 1.05)
axis = plt.axes(xlim=xlim, ylim=ylim)
axis.axhline(y=0, c="k", lw=1)
axis2 = axis.twinx()
axis2.axhline(y=7, c="r", ls="--", lw=1)
axis2.set_ylim(0, Emax * 1.05 / keV)

(line,) = axis.plot([], [], lw=2, c="b")
(energy_line,) = axis2.plot([], [], lw=2, c="r")
axis.set_xlabel("z [mm]")
axis.set_ylabel(r"On-axis E-field $E(r=0, z)$ [kV/mm]")
axis2.set_ylabel("Kinetic Energy [keV]")
axis.yaxis.label.set_color("blue")
axis2.yaxis.label.set_color("red")


def init():
    line.set_data([], [])
    energy_line.set_data([], [])
    return line, energy_line


def animate(i):

    this_Ez = warpEz0 * rf_volt(ts[i], freq=design_freq)
    # this_Ez = warpEz0 * rf_volt(t[3867, i], freq=design_freq)
    line.set_data(zwarp / mm, this_Ez / kV * mm)

    energy_line.set_data(zwarp[:i] / mm, warp_Ws[:i] / keV)
    # energy_line.set_data(zwarp[:i] / mm, warp_W[3867,:i]/keV)

    return (line, energy_line)


anim = FuncAnimation(
    fig, animate, init_func=init, frames=len(zwarp), interval=2, blit=True
)
plt.show()
# Writer = animation.writers['ffmpeg'] #for saving purposes
# writer = Writer(fps=20, metadata=dict(artist='Me'), bitrate=1800) #Some video settings.

# anim.save(os.getcwd() + "/animation.mp4", writer=writer)
