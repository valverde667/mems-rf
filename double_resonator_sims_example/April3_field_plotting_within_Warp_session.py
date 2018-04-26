#April 3, 2018
# more python plotting of fields
#
ez0=solver.getez()[solver.ix_axis,:]
ez0
import matplotlib.pyplot as plt
z=solver.zmesh
z
plt.plot(z,ez0,label=R=0)
plt.plot(z,ez16,label=R=0.0024 m)
plt.xlabel(Z(m))
plt.ylabel(Ez (V/m))
plt.legend()
plt.show()