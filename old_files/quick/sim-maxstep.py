import numpy as np
import matplotlib.pyplot as plt

import scipy.constants as SC
import time

import concurrent.futures

E = 10e3
mass = 40*SC.physical_constants['atomic mass constant'][0]  # Argon
Vmax = 700.0
f = 20e6
Ncells = 6

I = 60e-6  # Amps


def v(E):
    return np.sqrt(2*E/mass*SC.e)


N = 5000  # 10000 for high res
NEstart = np.zeros(shape=(N, 3))
# starting conditions, packages in 1ns time intervals
dt = 1e-9
dx = v(E) * 1000 * dt  # make pulse 1000 dt long
dQ = I*dt*1000/N

NEstart[:, 0] = np.linspace(-dx, 0, N)  # x-coordinate
NEstart[:, 1] = E  # energy
NEstart[:, 2] = 0.0  # time

# places where we have an RF cell + FC cup location
gap = 500e-6
tau = 1/2/f  # half period


def RF(NE, d, f=20e6):
    x = NE[:, 0]
    t = NE[:, 2]
    mask = x < d[-1]
    return Vmax*np.sin(2*np.pi*f*t)*mask


def do_sims(d):
    NE = NEstart.copy()

    # do the iteration: calc drift time, add energy and time
    for i, x in enumerate(d):
        tau = (x-NE[:, 0])/v(NE[:, 1])
        NE[:, 0] = x
        NE[:, 2] += tau
        if i % 2 == 0:
            NE[:, 1] += RF(NE, d, f)
        else:
            NE[:, 1] -= RF(NE, d, f)
    return NE


def single_grid_scan(i):
    # perfect gaps
    d = [0] + [v(E+(i+1)*Vmax)*tau for i in range(Ncells)]
    d = np.array(d)
    d = d.cumsum()
    d[-1] = 0.5  # Fcup position
    # randomize perfect gaps
    #print("b", i, ("  {:.3e} "*(Ncells+1)).format(*d))
    r = (2*np.random.random(Ncells)-1)*200e-6
    d[:-1] = d[:-1] + r
    print("a", i, ("  {:.3e} "*(Ncells+1)).format(*d))

    NE = do_sims(d)
    Y = []
    for V in X:
        mask = NE[:, 1] > V
        dt = NE[:, 2].ptp()
        Y.append(mask.sum()*dQ/dt)
    Y = np.array(Y)
    return Y

fig, (ax1, ax2) = plt.subplots(2, 1)

X = np.linspace(E-(Ncells+1)*Vmax, E+(Ncells+1)*Vmax, 200)

with concurrent.futures.ProcessPoolExecutor() as executor:
    jobs = [executor.submit(single_grid_scan, i) for i in range(20)]
    for f in concurrent.futures.as_completed(jobs):
        Y = f.result()
        ax1.plot(X*1e-3, Y*1e6)
        Edist = -np.diff(Y)
        Edist = Edist/Edist.sum()
        ax2.plot(0.5*(X[1:]+X[: -1])*1e-3, Edist)

ax1.set_xlabel("Grid Voltage [kV]")
ax1.set_ylabel("Current [$\mu$A]")
ax2.set_ylabel("Energy profile [a.u.]")

plt.show()
# plt.savefig("output-grid.png")
