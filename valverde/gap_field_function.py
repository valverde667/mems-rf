# This script will load the accelerating gaps and then fit a function to the
# time varying RF gap field.

import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.special import jv
import matplotlib.pyplot as plt
import os
import pdb

import warp as wp
from warp.utils.timedependentvoltage import TimeVoltage

# Useful constants to define for units and such
mm = 1e-3
kV = 1e3
keV = 1e3
MHz = 1e6
us = 1e-6
ns = 1e-9


def gap_voltage(t):
    """"Sinusoidal function of the gap voltage"""

    v = 3.5 * kV * np.cos(2 * np.pi * 13.6 * MHz * t)

    return v


def neg_gap_voltage(t):
    """"Sinusoidal function of the gap voltage"""

    v = -3.5 * kV * np.cos(2 * np.pi * 13.6 * MHz * t)

    return v


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

wp.w3d.zmmin = -2.5 * mm
wp.w3d.zmmax = 2.5 * mm
wp.w3d.nz = 300

wp.top.dt = 1 / 13.6 / MHz / 75


# Add boundary conditions
wp.w3d.bound0 = wp.dirichlet
wp.w3d.boundnz = wp.dirichlet
wp.w3d.boundxy = wp.periodic

wp.w3d.l4symtry = True
solver = wp.MRBlock3D()
wp.registersolver(solver)

rout = 0.75 * mm
rin = 0.50 * mm
length = 0.035 * mm
gap_width = 2 * mm
zcenter = abs(0.0 - gap_width / 2)
f = 13.6 * MHz


left_wafer = wp.Annulus(
    rmin=rin, rmax=rout, length=length, zcent=-zcenter, voltage=gap_voltage
)
TimeVoltage(left_wafer, voltfunc=gap_voltage)
right_wafer = wp.ZAnnulus(
    rmin=rin, rmax=rout, length=length, zcent=zcenter, voltage=neg_gap_voltage
)
TimeVoltage(right_wafer, voltfunc=neg_gap_voltage)

wp.installconductor(left_wafer)
wp.installconductor(right_wafer)
wp.generate()
z = wp.w3d.zmesh
steps = 75

# time = np.zeros(steps)
# Ez_array = np.zeros((steps, len(z)))
# Ez0 = wp.getselfe(comp='z')[0, 0, :]
# fig,ax = plt.subplots()
# phi0 = wp.getphi()[0, 0, :]
# ax.plot(z/mm, phi0, kV)
# ax.set_xlabel('z [mm]')
# ax.set_ylabel("Potential [kV]")
# ax.axvline(x=-zcenter/mm, c='g', lw=1)
# ax.axvline(x=zcenter/mm, c='g', lw=1)
# ax.axhline(y=0, c='k', lw=1)
# plt.show()

warpplots = False
if warpplots:
    wp.setup()
    wp.pfzx(fill=1, filled=1)
    wp.fma()
#
# for i in range(steps):
#     Ez = wp.getselfe(comp='z')[0, 0, :]
#     Ez_array[i, :] = Ez
#     time[i] = wp.top.time
#     wp.step()
#
# np.save('Ez_gap_field_151', Ez_array)
# np.save('zmesh', z)
# np.save(f'time_{steps}', time)

# Post process
Ez_array = np.load("Ez_gap_field_151.npy")
time_array = np.load(f"time_{steps}.npy")
z_array = np.load("zmesh.npy")


E_DC = 3.5 * kV / mm
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
