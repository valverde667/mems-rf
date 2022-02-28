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
kV = 1000
keV = 1000
MHz = 1e6
cm = 1e-2
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
Ng = 12
Vg = 7.0 * kV
E_DC = Vg / gap_width
dsgn_phase = -np.pi / 2.0
gap_cent_dist = []
Einit = 7.0 * keV
rf_wave = beta(Einit) * SC.c / 2.0 / f
fcup_dist = 50.0 * mm
Egains = [Einit]

for i in range(Ng):
    this_dist = calc_pires(Einit, freq=f)
    gap_cent_dist.append(this_dist)
    Egain = Vg * np.cos(dsgn_phase)  # Max acceleration
    Egains.append(Egain)

Egains = np.array(Egains)

# Real gap positions are the cumulative sums
gap_cent_dist = np.array(gap_cent_dist)
gap_centers = gap_cent_dist.cumsum()
# Shift gaps by drift space to allow for the field to start from minimum and climb.
zs = beta(Einit) * SC.c / 2.0 / np.pi / f * (dsgn_phase + np.pi)
gap_centers += zs

# Shift all gap positions to be centered on the middle lattice period
cell1 = gap_centers[0:4]
cell2 = gap_centers[4:8]
cell3 = gap_centers[8:]
cell2_center = (cell2[1] + cell2[2]) / 2.0

print("--Gap Centers")
print(gap_centers / mm)

# class Wafer(wp.Assembly):
#     """A single RF gap comprised of two wafers time varying voltage"""
#
#     def __init__(self, zc, length=0.1*mm, rout=0.56*mm, rin=0.55*mm, Amp=7*kV, freq = 13.6 * MHz):
#         self.zc = zc
#         self.length = length
#         self.rout = rout
#         self.rin = rin
#         self.Amp = Amp
#         self.freq = freq
#
#     def generate(self):
#         cond = wp.ZAnnulus(
#             rmax = self.rout,
#             rmin = self.rin,
#             zcent = self.zc,
#             length = self.length
#         )
#         return cond
#
#     def getvolt(self, t):
#         v = self.Amp * np.cos(2 * np.pi * self.freq * t)
#         return v

# class Wafer(wp.Assembly):
#     """
#   Annulus class
#     - rmin,rmax,length: annulus size
#     - theta=0,phi=0: angle of annulus relative to z-axis
#       theta is angle in z-x plane
#       phi is angle in z-y plane
#     - voltage=0: annulus voltage
#     - xcent=0.,ycent=0.,zcent=0.: center of annulus
#     - condid='next': conductor id, must be integer, or can be 'next' in
#                      which case a unique ID is chosen
#     """
#     def __init__(self,rmin,rmax,length,theta=0.,phi=0.,
#                       voltage=0.,xcent=0.,ycent=0.,zcent=0., Amp=7*kV, freq=13.6*MHz, condid='next',**kw):
#         assert (rmin<rmax),"rmin must be less than rmax"
#         kwlist = ['rmin','rmax','length','theta','phi']
#         wp.Assembly.__init__(self,voltage,xcent,ycent,zcent,condid,kwlist,
#                                annulusconductorf,annulusconductord,
#                                annulusintercept,cylinderconductorfnew,
#                                kw=kw)
#         self.rmin   = rmin
#         self.rmax   = rmax
#         self.length = length
#         self.theta  = theta
#         self.phi    = phi
#         self.Amp = Amp
#         self.freq = freq
#
#     def getextent(self):
#         # --- This is the easiest thing to do without thinking.
#         ll = sqrt(self.rmax**2 + (self.length/2.)**2)
#         return ConductorExtent([-ll,-ll,-ll],[+ll,+ll,+ll],[self.xcent,self.ycent,self.zcent])
#
#     def gridintercepts(self,xmmin,ymmin,zmmin,dx,dy,dz,
#                        nx,ny,nz,ix,iy,iz,mglevel):
#         # --- Take advantage of the already written cylinderconductorfnew
#         kwlistsave = self.kwlist
#         self.kwlist = ['rmin','length','theta','phi']
#         cin = wp.Assembly.gridintercepts(self,xmmin,ymmin,zmmin,dx,dy,dz,
#                                       nx,ny,nz,ix,iy,iz,mglevel)
#         self.kwlist = ['rmax','length','theta','phi']
#         cout = wp.Assembly.gridintercepts(self,xmmin,ymmin,zmmin,dx,dy,dz,
#                                        nx,ny,nz,ix,iy,iz,mglevel)
#         result = cout*(-cin)
#         self.kwlist = kwlistsave
#         return result
#
#     def getvolt(self, t):
#         v = self.Amp * np.cos(2 * np.pi * self.freq * t)
#         return v


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
wp.w3d.nz = round(3 * (wp.w3d.zmmax - wp.w3d.zmmin) / length)

# Add boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.periodic

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
wp.generate()

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


def poly_fit(zdata, a1, a2, a3, a4, a5, a6):
    return (
        a1
        + a2 * zdata
        + a3 * pow(zdata, 2)
        + a4 * pow(zdata, 3)
        + a5 * pow(zdata, 4)
        + a6 * pow(zdata, 5)
    )


def gauss(x, mu, sigma, A):
    return A * np.exp(-((x - mu) ** 2) / 2 / sigma ** 2)


def bimodal(x, mu1, sigma1, A1, mu2, sigma2, A2, shift):
    return gauss(x, mu1, sigma1, A1) + gauss(x, mu2, sigma2, A2) + shift


def fit_bessel(zdata, index):
    return jv(zdata, index)


def dual_bessel(zdata, i1, i2, i3):
    return jv(zdata, i1) + jv(zdata, i2) - jv(zdata, i3)


guess = (-0.5, 0.5, 0.8, 0.5, 0.5, 0.8, -0.6)
params, cov = curve_fit(bimodal, z_array / mm, Ez_array[0, :] / E_DC, guess)
mu1, sig1, A1 = params[:3]
mu2, sig2, A2 = params[3:6]
shift = params[-1]
fit = bimodal(z / mm, mu1, sig1, A1, mu2, sig2, A2, shift)

fig, ax = plt.subplots()
ax.set_xlabel("z [mm]")
ax.set_ylabel("Normed Electric Field and Fit")
ax.plot(z / mm, Ez_array[0, :] / E_DC, c="k", label="True")
ax.plot(z / mm, fit, c="b")
ax.legend()
plt.show()
