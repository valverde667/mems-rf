import numpy as np
import pandas as pd
from scipy.integrate import odeint # Use ODEINT to solve the differential equations defined by the vector field
import cufflinks

#defining parameters
epsilon=9.45e-7
epsilon_0=epsilon 
pi=np.pi

beamVoltage=8e3 #beam voltage
voltage=[] #voltage for each step
current=10e-6 #Ampere
linearChargeDensity=current/beamVoltage
electrodeAperture=0.55e-3

#defining initial conditions
a=0.125e-3 #horizontal 2·rms envelope coordinates of the beam envelope
aPrime=0 #horizontal divergence angle
b=0.125e-3 #vertical 2·rms envelope coordinates of the beam envelope
bPrime=0 #vertical divergence angle

stepNumber = 1000

esqVoltage = [1.120392e2, -2.939771e2, 3.601633e2, -3.477206e2, 3.254353e2, -2.933683e2, 2.94e2, -3.03e2, 3e2, -3e2]
driftDistance = 1e-3
esqDistance =1.59e-3 
beginDistance = 0    


voltageList = np.zeros(len(esqVoltage)*2+1) #voltages, of esqs or 0 for drift
conductorList=np.zeros(len(esqVoltage)*2+1) #location of conductors
for i in range(len(esqVoltage)*2+1):
        if i==0:
            conductorList[i]=driftDistance+beginDistance
            voltageList[i]=0
        elif i%2 == 0:
            conductorList[i]=conductorList[i-1]+driftDistance
            voltageList[i]=0
        else:
            conductorList[i]=conductorList[i-1]+esqDistance
            voltageList[i]=esqVoltage[int((i-1)/2)]
stoppoint=conductorList[-1]
            
def vectorfield(s,z,p):
    a, aPrime, b, bPrime = s #state variables
    voltage, beamVoltage, epsilon, linearChargeDensity, pi, epsilon_0 = p #parameters
    
    indexStep=sum(z>conductorList)
    if indexStep==len(conductorList):
        voltage=0
    else:
        voltage=voltageList[indexStep]

        
    f=[aPrime,(voltage/(beamVoltage*(electrodeAperture**2)))*a+epsilon**2/a**3+2*(linearChargeDensity/(4*pi*epsilon_0*beamVoltage))/(a+b),bPrime,(-voltage/(beamVoltage*(electrodeAperture**2)))*b+epsilon**2/b**3+2*(linearChargeDensity/(4*pi*epsilon_0*beamVoltage))/(a+b)]
    return f
abserr = 1.0e-10
relerr = 1.0e-10


z = [stoppoint * float(i) / (stepNumber - 1) for i in range(stepNumber)]
p = [voltage, beamVoltage, epsilon, linearChargeDensity, pi, epsilon_0 ]
s0 = [a, aPrime, b, bPrime]
wsol = odeint(vectorfield, s0, z, args=(p,), atol=abserr, rtol=relerr) # Call the ODE solver.
df=pd.DataFrame(np.concatenate((np.array(z).reshape((len(z),1)),wsol), 1),columns=["z","a","aPrime","b","bPrime"])
fig=df.plot(x='z',y=['a','b'])