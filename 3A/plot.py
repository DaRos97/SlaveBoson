import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
from colorama import Fore
from scipy.interpolate import interp2d

#read data from files
D3x3 = read_csv(inp.csvfile[0],usecols=['J2','J3','Energy','L','mL'])
j2 = []
for i in range(len(D3x3)):
    J2 = D3x3['J2'][i]
    if i == 0:
        j2.append(J2)
        ind = 0
        continue
    if np.abs(J2 - j2[ind]) > 1e-10 and J2 > j2[ind]:
        j2.append(J2)
        ind += 1
j2 = np.array(j2)
j3 = np.array(j2)
Npts = len(j2)
E = np.ndarray((Npts,Npts))
for i in range(Npts):
    for j in range(Npts):
        for ii in range(Npts**2):
            if np.abs(j2[i]-D3x3['J2'][ii]) < 1e-10 and np.abs(j3[j]-D3x3['J3'][ii]) < 1e-10:
                indE = ii
        E[i,j] = D3x3['Energy'][indE]
func1 = interp2d(j2,j3,E)
#####
Dq0 = read_csv(inp.csvfile[1],usecols=['J2','J3','Energy','L','mL'])
j2 = []
for i in range(len(Dq0)):
    J2 = Dq0['J2'][i]
    if i == 0:
        j2.append(J2)
        ind = 0
        continue
    if np.abs(J2 - j2[ind]) > 1e-10 and J2 > j2[ind]:
        j2.append(J2)
        ind += 1
j2 = np.array(j2)
j3 = np.array(j2)
Npts = len(j2)
E = np.ndarray((Npts,Npts))
for i in range(Npts):
    for j in range(Npts):
        for ii in range(Npts**2):
            if np.abs(j2[i]-Dq0['J2'][ii]) < 1e-10 and np.abs(j3[j]-Dq0['J3'][ii]) < 1e-10:
                indE = ii
        E[i,j] = Dq0['Energy'][indE]
func2 = interp2d(j2,j3,E)

plt.figure(figsize=(12,8))
x = np.linspace(j2[0],j2[-1],1000)
y = np.linspace(j2[0],j2[-1],1000)
plt.subplot(1,2,1)
plt.contourf(x,y,func1(x,y))
plt.colorbar()
plt.subplot(1,2,2)
plt.contourf(x,y,func2(x,y))
plt.colorbar()

plt.show()
