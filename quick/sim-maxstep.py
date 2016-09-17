import numpy as np
import matplotlib.pyplot as plt

import scipy.constants as SC
import time


E = 15e3
Cutoff = 15.2e3
mass = 40*SC.physical_constants['atomic mass constant'][0]  # Argon
Vmax = 1000.0

I = 60e-6  # Amps


def RF(NE, f=20e6):
    x = NE[:, 0]
    t = NE[:, 2]
    mask = (x > 0) * (x < 500e-6)
    return Vmax*np.sin(2*np.pi*f*t)*mask


def v(E):
    return np.sqrt(2*E/mass*SC.e)


N = 5000  # 10000 for high res
NEstart = np.zeros(shape=(N, 3))
# starting conditions, packages in 1ns time intervals
dt = 1e-9
dx = v(E) * dt * 1000
dQ = I*dt

NEstart[:, 0] = np.linspace(-dx, 0, N)  # x-coordinate
NEstart[:, 1] = E  # energy
NEstart[:, 2] = 0.0  # time

# places where we have an RF cell + FC cup location
d = [500e-6/2, 0.5]


def do_sims(Vgrid):
    NE = NEstart.copy()

    # do the iteration: calc drift time, add energy and time
    for x in d:
        tau = (x-NE[:, 0])/v(NE[:, 1])
        NE[:, 0] = x
        NE[:, 2] += tau
        NE[:, 1] += RF(NE)

    mask = NE[:, 1] > Vgrid
    dt = NE[:, 2].ptp()
    del NE
    return mask.sum()*dQ/dt

X = np.linspace(13e3, 17e3, 200)
Y = np.array([do_sims(V) for V in X])

fig, ax1 = plt.subplots(1, 1)

ax2 = ax1.twinx()

ax1.plot(X*1e-3, Y*1e6)

Edist = -np.diff(Y)
Edist = Edist/Edist.sum()
ax2.plot(0.5*(X[1:]+X[:-1])*1e-3, Edist, color="r")

ax1.set_xlabel("Grid Voltage [kV]")
ax1.set_ylabel("Current [$\mu$A]")
ax2.set_ylabel("Energy profile [a.u.]", color="r")

plt.savefig("output-grid.png")
