import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
from pandas import read_csv
import pandas as pd
from colorama import Fore
from scipy.interpolate import interp2d


D3x3 = read_csv(inp.csvfile[0],usecols=['J2','J3','Energy','L','mL'])
Dq0 = read_csv(inp.csvfile[1],usecols=['J2','J3','Energy','L','mL'])
if len(D3x3['J2']) != len(Dq0['J2']):
    print(Fore.RED+'Error, not same points evaluated'+Fore.RESET)
    exit()
## new dict
Data = []
Npts = len(D3x3['J2'])
new_header = ['J2','J3','Ansatz','SL']  #(J2,J3) coordinates, Ansatz = 0/2 for 3x3/q0 ansatz, SL = 0,1 for SL/LRO
for j in range(Npts):
    dic = {}
    data = [D3x3['J2'][j],D3x3['J3'][j]]
    for l in range(Npts):
        if np.abs(data[0] - Dq0['J2'][l]) < inp.cutoff_pts and np.abs(data[1] - Dq0['J3'][l]) < inp.cutoff_pts:
            indq = l
            break
    if D3x3['Energy'][j] < Dq0['Energy'][indq]:
        diff = D3x3['L'][j] - D3x3['mL'][j]
        data.append(0)
    else:
        diff = Dq0['L'][indq] - Dq0['mL'][indq]
        data.append(2)
    data.append(0 if np.abs(diff) > 1e-2 else 1)
    for I,txt in enumerate(new_header):
        dic[txt] = data[I]
    Data.append(dic)
#### Figure
Color = ['r','orange','b','c']
Label = ['3x3-SL','3x3-LRO','q0-SL','q0-LRO']
step = inp.step
plt.figure(figsize=(10,8))
for i in range(0):#Npts):
    stepj2 = step
    stepj3 = stepj2
    plt.fill_between(
            np.linspace(Data[i]['J2']-stepj2/2,Data[i]['J2']+stepj2/2,10),      #x line
            Data[i]['J3']-stepj3/2,      #y line
            Data[i]['J3']+stepj3/2,                       #ref y line
            color = Color[Data[i]['Ansatz'] + Data[i]['SL']],
            label = Label[Data[i]['Ansatz'] + Data[i]['SL']]
            )
    #plt.show()
for i in range(Npts):
    plt.scatter(Data[i]['J2'],Data[i]['J3'],marker='.',color=Color[Data[i]['Ansatz'] + Data[i]['SL']])
for i in range(4):
    plt.text(0.35,0+i/10,Label[i],color=Color[i])

plt.xlabel("$J_2$",size=20)
plt.ylabel("$J_{3e}$",size=20)
if 1:
    plt.show()
