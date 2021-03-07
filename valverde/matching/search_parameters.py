"""Script will search the 4D parameter space for the ESQ section in the
acceleration lattice. This is done by varying the r and r' in x-y for set
voltages V1 and V2. In other words, at some fixed V1 and V2 a specified cost
fucntion is minimized and this minimum value corresponds to the set V1 and V2.
Then, for example, V1 is incremented and the minimum parameter settings are
found once more."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import parameters as params
import solver

# Useful constants
kV = 1e3
mm = 1e-3
mrad = 1e-3

# Parameters (See fodo streamlit script)
d = 9.3 * mm  # Distance between RF gap centers
g = 2 * mm  # RF gap length
lq = 0.695 * mm  # ESQ length
space = (d - g - 2 * lq) / 3  # Spacing between ESQs
Lp = 2 * d  # Lattice period
Vbrkdwn = 3 * kV / mm

# Max settings
maxBias = Vbrkdwn * space
maxR = 0.55 * mm
maxDR = maxR / Lp

x = np.linspace(-3, 3, 101)
y = np.linspace(-3, 3, 101)
z = x ** 2 + y ** 2

# Hyperparameter settings
Niter = 1000
lrng_rate = 0.0001
Vsteps = 10
threshold = 0.01  # Cost function most likely wont approach 0 exactly


# Sample gradient descent run. Keeping here for future refernce.
# init = np.array([2, 1.4])
#
# hist = np.zeros(Niter)
# for i in range(Niter):
#     this_x, this_y = init[0], init[1]
#     this_hist = this_x**2 + this_y**2
#     hist[i] = this_hist
#
#     grad = 2*this_x + 2*this_y
#     init -= rate * grad
#
# plt.plot([i for i in range(Niter)], hist)
# plt.show()

# Create meshgrid of Voltages for V1 (focusing) and V2 (defocusing).
V1range = np.linspace(0.0, maxBias, Vsteps)
V2range = np.linspace(-maxBias, 0.0, Vsteps)
V1, V2 = np.meshgrid(V1range, V2range)
