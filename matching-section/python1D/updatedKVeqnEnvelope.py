#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd #importing Pandas 
from scipy.integrate import odeint # Use ODEINT to solve the differential equations defined by the vector field
import pandas as pd
import cufflinks as cf #importing plotly and cufflinks in offline mode
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[3]:


#defining parameters
# 0.3mm
pi=np.pi
beamEnergy=8e3 #eV
epsilon=1.42E-6 #unnormalized beam emittance
epsilon_0=epsilon 
current=0.09e-6 #Ampere
linearChargeDensity=current/beamEnergy
#defining initial conditions
a=0.3e-3 #horizontal 2·rms envelope coordinates of the beam envelope
aPrime=9e-3 #horizontal divergence angle
b=0.3e-3 #vertical 2·rms envelope coordinates of the beam envelope
bPrime=9e-3 #vertical divergence angle
electrodeAperture=.55e-3
esqVoltage=np.array([100,-250,250,-155,100,-80,70,-60,60,-60,55,-60,65,-60,55,-55,60,-65,65])


# In[4]:


saveZ=[]
saveVoltage=[]
voltage=[] #voltage for each step

stepNumber = 2000 #number of iterations

correct_distance=True #used when applying to warp. It is a better approximation if the ESQs models as slightly longer to account for fridnge fields
if correct_distance:
    correctionDistance=0.1e-3
else: 
    correctionDistance=0

driftDistance = 1e-3-correctionDistance
esqDistance =1.59e-3+correctionDistance
beginDistance = esqDistance/2 #additional distance between the beginning and the start of the first gap


voltageList = np.zeros(len(esqVoltage)*2+1) #voltages, for ESQs or 0 for gap
esqPositions=np.zeros(len(esqVoltage)*2+1) #location of conductors
for i in range(len(esqVoltage)*2+1):
        if i==0:
            esqPositions[i]=driftDistance+beginDistance
            voltageList[i]=0
        elif i%2 == 0:
            esqPositions[i]=esqPositions[i-1]+driftDistance
            voltageList[i]=0
        else:
            esqPositions[i]=esqPositions[i-1]+esqDistance
            voltageList[i]=esqVoltage[int((i-1)/2)]
stoppoint=esqPositions[-1]
    
    
def vectorfield(s,z,p):
    a, aPrime, b, bPrime = s #state variables
    voltage, beamEnergy, epsilon, linearChargeDensity, pi, epsilon_0 = p #parameters
    
    indexStep=sum(z>esqPositions)
    if indexStep==len(esqPositions):
        voltage=0
    else:
        voltage=voltageList[indexStep]
    global saveVoltage
    global saveZ
    saveZ+=[z]
    saveVoltage+=[voltage]
    f=[aPrime,(voltage/(beamEnergy*(electrodeAperture**2)))*a+epsilon**2/a**3+2*(linearChargeDensity/(4*pi*epsilon_0*beamEnergy))/(a+b),bPrime,(-voltage/(beamEnergy*(electrodeAperture**2)))*b+epsilon**2/b**3+2*(linearChargeDensity/(4*pi*epsilon_0*beamEnergy))/(a+b)]
    return f
abserr = 1.0e-12 #absolute error tolerance 
relerr = 1.0e-12 #relative error tolerance


z = [stoppoint * float(i) / (stepNumber - 1) for i in range(stepNumber)] #z position for each step
p = [voltage, beamEnergy, epsilon, linearChargeDensity, pi, epsilon_0 ]
s0 = [a, aPrime, b, bPrime]
wsol = odeint(vectorfield, s0, z, args=(p,), atol=abserr, rtol=relerr) # Call the ODE solver.
df=pd.DataFrame(np.concatenate((np.array(z).reshape((len(z),1)),wsol), 1),columns=["z","a","aPrime","b","bPrime"])
df['a']=df['a']*1000 #converting from m to mm
df['b']=df['b']*1000
df['z']=df['z']*1000


df3=pd.DataFrame()
df3['saveZ']=saveZ
df3['saveVoltage']=saveVoltage
df4=df3.sort_values(['saveZ'])
df4['saveZ']=df4['saveZ']*1000 #converting from m to mm



# Plotting Results


import plotly.graph_objects as go
from plotly.subplots import make_subplots


# Can change between true and false to plot desired results
# warp and java are imported below if neeeded
python_plot=True
warp_plot=False
java_plot=False
voltage_plot=True








# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])
# Add traces
if python_plot:
    fig.add_trace(
        go.Scatter(x=df['z'],y=df['b'], name="Python: X"),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df['z'],y=df['a'], name="Python: Y"),
        secondary_y=False,
    )


if warp_plot:
    fig.add_trace(
        go.Scatter(x=warp_data['z'],y=warp_data['x2rms'], name="Warp: X"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=warp_data['z'],y=warp_data['y2rms'], name="Warp: Y"),
        secondary_y=False,
    )
if java_plot:
    fig.add_trace(
        go.Scatter(x=df_java['Z'],y=df_java['a(m)'], name="Java: X"),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=df_java['Z'],y=df_java['b(m)'], name="Java: Y"),
        secondary_y=False,
    )

if voltage_plot:


    fig.add_trace(
        go.Scatter(x=df4['saveZ'],y=df4['saveVoltage'], name="ESQ Voltage"),
        secondary_y=True,
    )

fig.update_layout(
    xaxis=dict(
        title="Z (mm)",
   ),
    legend=dict(
    yanchor="bottom",
    y=0.2,
    xanchor="right",
    x=1.1
),
    yaxis=dict(
        domain=[0,0.7],
        title="X,Y Beam Envelope, 2•rms (mm)",
        titlefont=dict(
            color="#333333"
        ),
        tickfont=dict(
            color="#333333"
        )
    ),
    yaxis2=dict(
        domain=[0.7,1],
        zeroline=False,
        title="Voltage (V)",
        titlefont=dict(
            color="#333333"
        ),
        tickfont=dict(
            color="#333333"
        ),
        
        anchor="x",
        overlaying="y2",
        side="right",
        position=0.95
    ),
    



    )
    


# In[ ]:





# In[5]:


# for varying voltages in stability analysis

vary_voltage=False
if vary_voltage:
    variedVoltage=np.zeros(len(esqVoltage))
    for i in range(len(esqVoltage)):
        variedVoltage[i]=esqVoltage[i]*np.random.uniform(0.9,1.1)
    esqVoltage=variedVoltage

convert_warp=False
if convert_warp:
    warpVoltage=esqVoltage/1.1
    warpVoltageRounded=np.round_(warpVoltage,decimals=1)
    


# In[ ]:





# In[6]:


data_from_java=False
if data_from_java:

    data=pd.read_csv('C:/Users/Brian/Box/SUMMER 2020/LBL Java//Java_Inputs/08-02 and 08-08/35.0.out',delim_whitespace=True)

    df_java=pd.DataFrame(data)
    df_java['a(m)']=df_java['a(m)']*1000
    df_java['b(m)']=df_java['b(m)']*1000
    df_java['Z']=df_java['Z']*1000


# In[7]:


# for plotting data from warp
data_from_warp=False
if data_from_warp:
    warp_data=pd.read_csv('C:/Users/Brian/Box/SUMMER 2020/Python/35.0.csv')
    warp_data['x2rms']=2*warp_data['xrms']
    warp_data['y2rms']=2*warp_data['yrms']
    velo=196453.9
    warp_data['z']=warp_data['t']*velo
    warp_data['x2rms']=warp_data['x2rms']*1000
    warp_data['y2rms']=warp_data['y2rms']*1000
    warp_data['z']=warp_data['z']*1000


# In[ ]:




