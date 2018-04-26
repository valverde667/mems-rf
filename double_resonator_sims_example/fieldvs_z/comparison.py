import numpy as np
import matplotlib.pyplot as plt

e0ez=np.loadtxt('e0ez')
erez=np.loadtxt('erez')
x = np.loadtxt('x')
z = np.loadtxt('z')

#index depends on grid
# 0 , .5mm, .8mm

r = np.array([0,.5e-3,.8e-3])
grid = list(int(round(i*100/5e-3)) for i in r)

for g in grid:
    plt.plot(z,e0ez[g], color='red',label='Free space')
    plt.plot(z,erez[g],color='blue',label='Dielectric')

plt.xlabel('Z (m)')
plt.ylabel('Ez')
plt.legend()
plt.show()
