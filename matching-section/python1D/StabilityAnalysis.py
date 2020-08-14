#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from scipy.integrate import odeint # Use ODEINT to solve the differential equations defined by the vector field
import pandas as pd

# stability analysis on radius


# In[6]:


#importing Pandas 
#importing plotly and cufflinks in offline mode
import cufflinks as cf
import plotly.offline
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[245]:


# adding zero at the end to avoid messing up graph
#directory = "C:/Users/Brian/Box/SUMMER 2020/Python/"
files = ["35.0.csv","35.1.csv","35.2.csv","35.3.csv","35.4.csv","35.5.csv","35.6.csv","35.7.csv","35.8.csv","35.9.csv","35.10.csv"]
velo = 196453.9
directory = "./"
#files = ["35.0.csv","35.1.csv","35.10.csv"]
columns = [
 't',
 'xrms',
 'yrms',
]


df2=pd.DataFrame()
for file in files:
    df = pd.read_csv(directory+file)
    t_name = file.split(".csv")[0]
    df2[t_name+'_2rms']=2000*df['xrms']
    #df2[t_name+'y2rms']=2000*df['yrms']
    df2[t_name+'z']=df['t']*velo*1000
    
df3=pd.DataFrame()
for file in files:
    df = pd.read_csv(directory+file)
    t_name = file.split(".csv")[0]
    #df3[t_name+'x2rms']=2000*df['xrms']
    df3[t_name+'_2rms']=2000*df['yrms']
    df3[t_name+'z']=df['t']*velo*1000
    
    
    


# In[246]:


result=result.rename(columns={'35.0_2rms': '0.3mm 9mrad','35.1_2rms': '0.35mm 9mrad','35.2_2rms': '0.4mm 9mrad','35.3_2rms': '0.25mm 9mrad','35.4_2rms': '0.2mm 9mrad'})


# In[247]:


result.iplot(x='35.0z',y=['0.2mm 9mrad','0.25mm 9mrad', '0.3mm 9mrad','0.35mm 9mrad','0.4mm 9mrad'],xTitle="Z (mm)",yTitle="Beam Envelope 2•rms X and Y (mm)")
       


# In[248]:


result=result.rename(columns={'35.0_2rms': '0.3mm 9mrad','35.5_2rms': '0.3mm 6mrad','35.6_2rms': '0.3mm 3mrad','35.7_2rms':'0.3mm 1.5mrad','35.8_2rms': '0.3mm 12mrad','35.9_2rms': '0.3mm 15mrad'})
result.iplot(x=['35.0z'],y=['0.3mm 1.5mrad','0.3mm 3mrad','0.3mm 6mrad','0.3mm 9mrad','0.3mm 12mrad','0.3mm 15mrad'],xTitle="Z (mm)",yTitle="Beam Envelope 2•rms X and Y (mm)")


# In[249]:


for i in range(len(df2.columns)):
    df2.iloc[0,i]=-0.1
    if i%2:
        
        df2.iloc[-1,i]=df2.iloc[-2,i]
    else: 
        df2.iloc[-1,i]=-0.1
for i in range(len(df3.columns)):
    df3.iloc[0,i]=-0.1
    if i%2:
        
        df3.iloc[-1,i]=df3.iloc[-2,i]
    else: 
        df3.iloc[-1,i]=-0.1

result = df2.append(df3)


# In[ ]:





# In[250]:


#directory = "C:/Users/Brian/Box/SUMMER 2020/Python/"
files = ["37.0.csv","37.1.csv","37.2.csv","37.3.csv","37.4.csv","37.5.csv","37.6.csv","37.7.csv","37.8.csv","37.9.csv"]
velo = 196453.9
directory = "./"
columns = [
 't',
 'xrms',
 'yrms',
]


df37x=pd.DataFrame()
for file in files:
    df = pd.read_csv(directory+file)
    t_name = file.split(".csv")[0]
    df37x[t_name+' (X and Y 2rms)']=2000*df['xrms']
    #df2[t_name+'y2rms']=2000*df['yrms']
    df37x[t_name+'z']=df['t']*velo*1000
    
df37y=pd.DataFrame()
for file in files:
    df = pd.read_csv(directory+file)
    t_name = file.split(".csv")[0]
    #df3[t_name+'x2rms']=2000*df['xrms']
    df37y[t_name+' (X and Y 2rms)']=2000*df['yrms']
    df37y[t_name+'z']=df['t']*velo*1000
    
for i in range(len(df37x.columns)):
    df37x.iloc[0,i]=-0.1
    if i%2:
        
        df37x.iloc[-1,i]=df37x.iloc[-2,i]
    else: 
        df37x.iloc[-1,i]=-0.1
for i in range(len(df37y.columns)):
    df37y.iloc[0,i]=-0.1
    if i%2:
        
        df37y.iloc[-1,i]=df37y.iloc[-2,i]
    else: 
        df37y.iloc[-1,i]=-0.1
    
df37=df37x.append(df37y)
    


# In[251]:


df37.iplot(x=['37.0z'],y=['37.0 (X and Y 2rms)','37.1 (X and Y 2rms)','37.2 (X and Y 2rms)','37.3 (X and Y 2rms)','37.4 (X and Y 2rms)'],xTitle="Z (mm)",yTitle="Beam Envelope 2•rms X and Y (mm)")


# In[252]:


df37.iplot(x=['37.0z'],y=['37.5 (X and Y 2rms)','37.6 (X and Y 2rms)','37.7 (X and Y 2rms)','37.8 (X and Y 2rms)','37.9 (X and Y 2rms)'],xTitle="Z (mm)",yTitle="Beam Envelope 2•rms X and Y (mm)")


# In[253]:



files = ["38.0.csv","38.1.csv","38.2.csv","38.3.csv","38.4.csv","38.5.csv","38.6.csv","38.7.csv","38.8.csv","38.9.csv","38.10.csv"]
velo = 196453.9
directory = "./"
columns = [
 't',
 'xrms',
 'yrms',
]


df38x=pd.DataFrame()
for file in files:
    df = pd.read_csv(directory+file)
    t_name = file.split(".csv")[0]
    df38x[t_name+'_2rms']=2000*df['xrms']
    #df2[t_name+'y2rms']=2000*df['yrms']
    df38x[t_name+'z']=df['t']*velo*1000
    
df38y=pd.DataFrame()
for file in files:
    df = pd.read_csv(directory+file)
    t_name = file.split(".csv")[0]
    #df3[t_name+'x2rms']=2000*df['xrms']
    df38y[t_name+'_2rms']=2000*df['yrms']
    df38y[t_name+'z']=df['t']*velo*1000
    
for i in range(len(df38x.columns)):
    df38x.iloc[0,i]=-0.1
    if i%2:
        
        df38x.iloc[-1,i]=df38x.iloc[-2,i]
    else: 
        df38x.iloc[-1,i]=-0.1
for i in range(len(df38y.columns)):
    df38y.iloc[0,i]=-0.1
    if i%2:
        
        df38y.iloc[-1,i]=df38y.iloc[-2,i]
    else: 
        df38y.iloc[-1,i]=-0.1
    
df38=df38x.append(df38y)
    


# In[254]:


df38['Original Voltages']=df37['37.0 (X and Y 2rms)']


# In[255]:


df38=df38.rename(columns={'38.1_2rms': '+10%','38.2_2rms': '+20%','38.3_2rms': '+30%','38.4_2rms': '+40%','38.5_2rms': '+50%','38.6_2rms': '-10%','38.7_2rms': '-20%','38.8_2rms': '-30%','38.9_2rms': '-40%','38.10_2rms': '-50%'})
df38.iplot(x=['38.0z'],y=['-50%','-40%','-30%','-20%','-10%','Original Voltages','+10%','+20%','+30%','+40%','+50%'],xTitle="Z (mm)",yTitle="Beam Envelope 2•rms X and Y (mm)")


# In[256]:



files = ["39.0.csv","39.2.csv","39.3.csv","39.4.csv","39.5.csv","39.6.csv","39.7.csv","39.8.csv","39.9.csv","39.10.csv"]
velo = 196453.9
directory = "./"
columns = [
 't',
 'xrms',
 'yrms',
]


df39x=pd.DataFrame()
for file in files:
    df = pd.read_csv(directory+file)
    t_name = file.split(".csv")[0]
    df39x[t_name+'_2rms']=2000*df['xrms']
    #df2[t_name+'y2rms']=2000*df['yrms']
    df39x[t_name+'z']=df['t']*velo*1000
    
df39y=pd.DataFrame()
for file in files:
    df = pd.read_csv(directory+file)
    t_name = file.split(".csv")[0]
    #df3[t_name+'x2rms']=2000*df['xrms']
    df39y[t_name+'_2rms']=2000*df['yrms']
    df39y[t_name+'z']=df['t']*velo*1000
    
for i in range(len(df39x.columns)):
    df39x.iloc[0,i]=-0.1
    if i%2:
        
        df39x.iloc[-1,i]=df39x.iloc[-2,i]
    else: 
        df39x.iloc[-1,i]=-0.1
for i in range(len(df39y.columns)):
    df39y.iloc[0,i]=-0.1
    if i%2:
        
        df39y.iloc[-1,i]=df39y.iloc[-2,i]
    else: 
        df39y.iloc[-1,i]=-0.1
    
df39=df39x.append(df39y)
    


# In[257]:


df39=df39.rename(columns={'39.0_2rms': 'Current: 0.09E-6 A','39.2_2rms': 'Current: 0.09E-1 A','39.3_2rms': 'Current: 0.09E-2 A','39.4_2rms': 'Current: 0.09E-3 A','39.5_2rms': 'Current: 0.09E-4 A','39.6_2rms': 'Current: 0.09E-5 A','39.7_2rms': 'Current: 0.09E-7 A','39.8_2rms': 'Current: 0.09E-8 A','39.9_2rms': 'Current: 0.09E-9 A','39.10_2rms': 'Current: 0.09E-10 A'})
df39.iplot(x=['39.0z'],y=['Current: 0.09E-2 A', 'Current: 0.09E-3 A', 'Current: 0.09E-4 A', 'Current: 0.09E-5 A','Current: 0.09E-6 A', 'Current: 0.09E-7 A','Current: 0.09E-8 A', 'Current: 0.09E-9 A','Current: 0.09E-10 A'],xTitle="Z (mm)",yTitle="Beam Envelope 2•rms X and Y (mm)")


# In[ ]:





# In[258]:



files = ["40.0.csv","40.4.csv","40.5.csv","40.6.csv","40.7.csv","40.8.csv"]
velo = 196453.9
directory = "./"
columns = [
 't',
 'xrms',
 'yrms',
]


df40x=pd.DataFrame()
for file in files:
    df = pd.read_csv(directory+file)
    t_name = file.split(".csv")[0]
    df40x[t_name+'_2rms']=2000*df['xrms']
    #df2[t_name+'y2rms']=2000*df['yrms']
    df40x[t_name+'z']=df['t']*velo*1000
    
df40y=pd.DataFrame()
for file in files:
    df = pd.read_csv(directory+file)
    t_name = file.split(".csv")[0]
    #df3[t_name+'x2rms']=2000*df['xrms']
    df40y[t_name+'_2rms']=2000*df['yrms']
    df40y[t_name+'z']=df['t']*velo*1000
    
for i in range(len(df40x.columns)):
    df40x.iloc[0,i]=-0.1
    if i%2:
        
        df40x.iloc[-1,i]=df40x.iloc[-2,i]
    else: 
        df40x.iloc[-1,i]=-0.1
for i in range(len(df40y.columns)):
    df40y.iloc[0,i]=-0.1
    if i%2:
        
        df40y.iloc[-1,i]=df40y.iloc[-2,i]
    else: 
        df40y.iloc[-1,i]=-0.1
    
df40=df40x.append(df40y)
    


# In[259]:


df40=df40.rename(columns={'40.0_2rms': 'Emittance: 1.42E-6 (m rad)','40.4_2rms': 'Emittance: 1.42E-4 (m rad)','40.5_2rms': 'Emittance: 1.42E-5 (m rad)','40.6_2rms': 'Emittance: 1.42E-7 (m rad)','40.7_2rms': 'Emittance: 1.42E-8 (m rad)','40.8_2rms': 'Emittance: 1.42E-9 (m rad)'})
df40.iplot(x=['40.0z'],y=['Emittance: 1.42E-5 (m rad)','Emittance: 1.42E-6 (m rad)','Emittance: 1.42E-7 (m rad)','Emittance: 1.42E-8 (m rad)','Emittance: 1.42E-9 (m rad)'],xTitle="Z (mm)",yTitle="Beam Envelope 2•rms X and Y (mm)")


# In[ ]:




