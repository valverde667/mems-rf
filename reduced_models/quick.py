import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as SC


def get_ind(coord, delta, mesh):
    """ "Find index of coordinate given the mesh of values and spacing."""

    index = np.where((mesh <= coord + delta) & (mesh >= coord - delta))
    ind = index[0][0]

    return ind


mm = 1.0e-3
kV = 1.0e3
keV = 1.0e3
MHz = 1.0e6
length = 0.7 * mm
gap_width = 2.0 * mm
f = 13.6 * MHz
Ng = 2
Vg = 7.0 * kV
E_DC = 7 * kV / gap_width
dsgn_phase = -0.0 * np.pi
gap_cent_dist = []
Einit = 7.0 * keV
fcup_dist = 20.0 * mm
Energy = [Einit]

x = np.load("xmesh.npy")
y = np.load("ymesh.npy")
z = np.load("zmesh.npy")
Ez = np.load("Efield.npy")[:, 0, :]
V = np.load("potential.npy")[:, 0, :]
gap_centers = np.load("gap_centers.npy")

dx = x[1] - x[0]
dz = z[1] - z[0]

centers = np.array([0.0, 5.0, 10.0]) * mm
ind = get_ind(15.0 * mm, dx, x)

fig, ax = plt.subplots()
ax.plot(z / mm, V[ind, :] / kV)
ax.set_xlabel("z [mm]")
ax.set_ylabel("Potential [kV]")
ymin, ymax = ax.get_ylim()
for i, cent in enumerate(gap_centers):
    left = cent - gap_width / 2 - length / 2
    right = cent + gap_width / 2 + length / 2
    ax.axvline(x=left / mm, c="gray", lw=0.7)
    ax.axvline(x=right / mm, c="gray", lw=0.7)
    ax.axvspan(left / mm, right / mm, color="grey", alpha=0.5)
ax.axhline(y=0, c="k", lw=1)
plt.savefig("potential_offaxis", dpi=400)

fig, ax = plt.subplots()
ax.plot(z / mm, Ez[ind, :] / E_DC)
ax.set_xlabel("z [mm]")
ax.set_ylabel(r"Normed On-axis E-field $E(x=0, y=0, z)/E_{DC}$ [V/m]")
ymin, ymax = ax.get_ylim()
for i, cent in enumerate(gap_centers):
    left = cent - gap_width / 2 - length / 2
    right = cent + gap_width / 2 + length / 2
    ax.axvline(x=left / mm, c="gray", lw=0.7)
    ax.axvline(x=right / mm, c="gray", lw=0.7)
    ax.axvspan(left / mm, right / mm, color="grey", alpha=0.5)
ax.axhline(y=0, c="k", lw=1)
plt.savefig("Efield_offaxis", dpi=400)
plt.show()
