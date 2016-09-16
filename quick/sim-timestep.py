import numpy as np
import matplotlib.pyplot as plt

import scipy.constants as SC
import time

from scipy.interpolate import UnivariateSpline

# 10 us in ns intervals
E = 15e3
Vmax = 1.0e3
f = 20e6

N = 1000  # number of 1ns bins

fileidx = 0
filename = "output{:05d}.png"

class Imon():
    dt = 5e-9
    def __init__(self, z=0, minE=0):
        self.mask = None  # keep track which buckets we already counted
        self.t = []
        self.I = []
        self.z = z
        self.name = str(z)
        self.last = 0
        self.minE = minE

    def __call__(self, NE, t):
        if t < self.last + Imon.dt:
            return
        self.last = t
        newmask = NE[:, 2] > self.z
        if self.mask is not None:
            newmask = newmask ^ self.mask
        if not any(newmask):
            return
        energymask = NE[:, 0] >= self.minE
        self.t.append(t)
        self.I.append(sum(newmask*energymask)*20e-12/N)  # sum up charge in this time delta
        if self.mask is not None:
            self.mask = self.mask & newmask
        else:
            self.mask = newmask

    def getI(self):
        t, current = np.array(self.t), np.array(self.I)
        return t[:-1], np.diff(current)/np.diff(t)

I1 = Imon(z=50e-2)
I2 = Imon(z=50e-2)
I3 = Imon(z=50e-2, minE = 15.2e3)

I = [I1, I2, I3]

mass = 40*SC.physical_constants['atomic mass constant'][0]  # Argon

def v(E):
    return np.sqrt(2*E/mass*SC.e)

NE = np.zeros(shape=(N, 3))
NE[:, 0] = E                                # energy
NE[:, 1] = v(NE[:, 0])                      # velocity
NE[:, 2] = np.linspace(-v(E)*N*1e-9, 0, N)   # position

ZE = NE.copy()
# simulate RF being applied to 1 us pulse across a 500um wafer [0:500um]
# followed by a drift of 0.5m
# time steps of 1ns
t = 0
dt = 1e-10
i = 0

plt.ion()

fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
ax1.hist(NE[:, 2], 100, lw=0)
vel, = ax2.plot(NE[:, 2], NE[:, 0], ".")
vel2, = ax2.plot(ZE[:, 2], ZE[:, 0], "g.")
ax2.set(title='', xlabel='X [m]', ylabel='$E_{kin}$ [eV]')

fig.suptitle('500um accel gap + 50cm drift')

while NE[:, 2].min()<500e-6:
    i += 1
    t += dt
    V = Vmax*np.sin(2*np.pi*f*t)
    Efield = V/500e-6
    dx = NE[:, 1]*dt
    gain = Efield*dx
    NE[:, 0] = np.where((NE[:, 2] < 500e-6)*(NE[:, 2] > 0), NE[:, 0]+gain, NE[:, 0])
    NE[:, 1] = v(NE[:, 0])
    NE[:, 2] += dx
    ZE[:, 2] += ZE[:, 1]*dt
    if i%100 == 0:
        ax1.cla()
        ax1.set(title='Current pulse', xlabel='X [m]', ylabel='Charge [nC]')
        ax1.hist(NE[:, 2], 100, lw=0)
        ax1.hist(ZE[:, 2], 100, lw=0, color='green', alpha=0.4)
        vel2.set_xdata(ZE[:, 2])
        energymask = NE[:, 0] > 15.2e3
        ax1.hist(NE[energymask, 2], 100, lw=0, color='red')
        vel.set_ydata(NE[:, 0])
        vel.set_xdata(NE[:, 2])
        ax2.relim()
        ax2.autoscale_view(True, True, True)
        Xmin = NE[:, 2].min()
        ax1.set_xlim([Xmin-0.1, Xmin+0.4])
        ax2.set_xlim([Xmin-0.1, Xmin+0.4])
        ax3.cla()
        for imon, name in zip(I, ['RF', 'no RF', 'RF E>15.2keV filter']):
            T, current = imon.getI()
            ax3.plot(T*1e6, current*1e6, label=name)
            ax1.axvline(imon.z, lw=2, color="yellow")
            ax3.set(title='', xlabel='Time [us]', ylabel='Current [uA]')
            ax3.set_xlim([1.8, 3.0])
            ax3.legend(loc='upper right')
            ax3.set(xlabel="Time [us]", ylabel="Current [uA]")
        fig.canvas.draw()
        plt.pause(0.0000001)
        plt.savefig(filename.format(fileidx))
        fileidx += 1
    I1(NE, t)
    I2(ZE, t)
    I3(NE, t)

dt = 1e-9
ax1.set_ylim(0, 52)

while NE[:, 2].min()<0.5:
    i += 1
    t += dt
    NE[:, 2] += NE[:, 1]*dt
    ZE[:, 2] += ZE[:, 1]*dt
    if i%10 == 0:
        ax1.cla()
        ax1.set(title='Current pulse', xlabel='X [m]', ylabel='Charge [nC]')
        ax1.hist(NE[:, 2], 100, lw=0)
        ax1.hist(ZE[:, 2], 100, lw=0, color='green', alpha=0.4)
        vel2.set_xdata(ZE[:, 2])
        energymask = NE[:, 0] > 15.2e3
        ax1.hist(NE[energymask, 2], 100, lw=0, color='red')
        vel.set_ydata(NE[:, 0])
        vel.set_xdata(NE[:, 2])
        ax2.relim()
        ax2.autoscale_view(True, True, True)
        Xmin = NE[:, 2].min()
        ax1.set_xlim([Xmin-0.1, Xmin+0.4])
        ax2.set_xlim([Xmin-0.1, Xmin+0.4])
        ax3.cla()
        for imon, name in zip(I, ['RF', 'no RF', 'RF E>15.2keV filter']):
            T, current = imon.getI()
            ax3.plot(T*1e6, current*1e6, label=name)
            ax1.axvline(imon.z, lw=2, color="yellow")
            ax3.set_xlim([1.8, 3.0])
            ax3.legend(loc='upper right')
            ax3.set(xlabel="Time [us]", ylabel="Current [uA]")
        fig.canvas.draw()
        plt.pause(0.000001)
        plt.savefig(filename.format(fileidx))
        fileidx += 1
    I1(NE, t)
    I2(ZE, t)
    I3(NE, t)


ax1.cla()
ax1.set(title='Current pulse', xlabel='X [m]', ylabel='Charge [nC]')
ax1.hist(NE[:, 2], 100, lw=0)
ax1.hist(ZE[:, 2], 100, lw=0, color='green', alpha=0.4)
vel2.set_xdata(ZE[:, 2])
energymask = NE[:, 0] > 15.2e3
ax1.hist(NE[energymask, 2], 100, lw=0, color='red')
vel.set_ydata(NE[:, 0])
vel.set_xdata(NE[:, 2])
ax1.relim()
ax1.autoscale_view(True, True, True)
ax2.relim()
ax2.autoscale_view(True, True, True)
Xmin = NE[:, 2].min()
ax1.set_xlim([Xmin-0.1, Xmin+0.4])
ax2.set_xlim([Xmin-0.1, Xmin+0.4])
ax3.cla()
for imon, name in zip([I1, I2, I3], ['RF', 'no RF', 'RF E>15.2keV filter']):
    t, current = imon.getI()
    ax3.plot(t*1e6, current*1e6, label=name)
    ax1.axvline(imon.z, lw=2, color="yellow")
    ax3.set_xlim([1.8, 3.0])
    ax3.set(xlabel="Time [us]", ylabel="Current [uA]")
ax3.legend(loc='upper right')
fig.canvas.draw()
plt.pause(0.0001)
plt.savefig(filename.format(fileidx))
