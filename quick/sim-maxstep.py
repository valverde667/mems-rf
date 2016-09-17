import numpy as np
import matplotlib.pyplot as plt

import scipy.constants as SC
import time


E = 15e3
Cutoff = 15.2e3
mass = 40*SC.physical_constants['atomic mass constant'][0]  # Argon
Vmax = 1000.0


def RF(NE, f=20e6):
    x = NE[:, 0]
    t = NE[:, 2]
    mask = (x > 0) * (x < 500e-6)
    return Vmax*np.sin(2*np.pi*f*t)*mask


def v(E):
    return np.sqrt(2*E/mass*SC.e)


N = 1000  # 10000 for high res
NE = np.zeros(shape=(N, 3))

# starting conditions, packages in 1ns time intervals
dx = v(E) * 1e-9 * 1000

NE[:, 0] = np.linspace(-dx, 0, N)  # x-coordinate
NE[:, 1] = E  # energy
NE[:, 2] = 0.0  # time

# places where we have an RF cell + FC cup location
d = [500e-6/2, 0.5]

# do the iteration: calc drift time, add energy and time
for x in d:
    tau = (x-NE[:, 0])/v(NE[:, 1])
    NE[:, 0] = x
    NE[:, 2] += tau
    NE[:, 1] += RF(NE)

# make current plot, which is just a history over the time, since all
# x-coordinates will be at the same location of the F-cup now
plt.hist(NE[:, 2]*1e6, N//10, lw=0)

# same, but only for accelerated particles
mask = NE[:, 1] > Cutoff
plt.hist(NE[mask, 2]*1e6, N//10, lw=0)

plt.xlabel("Times [$\mu$s]")
plt.ylabel("Current [a.u.]")
plt.savefig("output-maxstep.png")
