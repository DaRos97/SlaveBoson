import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
from pandas import read_csv
from colorama import Fore

#read data from files
data1 = read_csv(inp.csvfile[0],usecols=['J2','J3','Energies','L','mL'])
data2 = read_csv(inp.csvfile[1],usecols=['J2','J3','Energies','L','mL'])
if len(data1['J2']) != len(data2['J2']):
    print(Fore.RED+'Error, not same points evaluated'+Fore.RESET)
    exit()
## new dict
Data = []
Npts = len(data1['J2'])
new_header = ['J2','J3','Ansatz','SL']  #(J2,J3) coordinates, Ansatz = 0/1 for 3x3/q0 ansatz, SL = 0,1 for SL/LRO
for j2 in range(Npts):
    for j3 in range(Npts):
        dic = {}
        data = [data1['J2'][j2],data1['J3'][j3]]
        if 
        for I,txt in enumerate(new_header):
            dic[txt] = data[I]
#organize energies
minE = np.zeros((pts,pts),dtype=int)
SL = np.zeros((pts,pts),dtype=int)
for i in range(pts):
    for j in range(pts):
        if dataE[0,i,j] < dataE[1,i,j]:
            E[i,j] = dataE[0,i,j]
            Ans[i,j] = 0
            SL[i,j] = 0 if np.abs(dataL[0,0,i,j]-dataL[0,1,i,j]) < 1e-3 else 1
        else:
            E[i,j] = dataE[1,i,j]
            SL[i,j] = 0 if np.abs(dataL[1,0,i,j]-dataL[1,1,i,j]) < 1e-3 else 1
            Ans[i,j] = 2

Color = ['orange','r','c','b']
Label = ['(0,0)-LRO','(0,0)-SL','(0,pi)-LRO','(0,pi)-SL']
plt.figure(figsize=(10,8))
#grid
JM = Jmax + Jr/(pts-1)/2
Jm = Jmin - Jr/(pts-1)/2
for i in range(pts+1):
    plt.plot((Jm+Jr/(pts-1)*i,Jm+Jr/(pts-1)*i),(Jm,JM),'k')
    plt.plot((Jm,JM),(Jm+Jr/(pts-1)*i,Jm+Jr/(pts-1)*i),'k')
for i in range(pts):
    for j in range(pts):
        plt.fill_between(
                np.linspace(Jm+Jr/(pts-1)*i,Jm+Jr/(pts-1)*(i+1),10),
                np.linspace(Jm+Jr/(pts-1)*(j+1),Jm + Jr/(pts-1)*(j+1),10),
                Jm+Jr/(pts-1)*j,
                color=Color[Ans[i,j]+SL[i,j]],
                label= Label[Ans[i,j]+SL[i,j]])
for i in range(pts):
    for j in range(pts):
        plt.scatter(Jmin+Jr/(pts-1)*i,Jmin+Jr/(pts-1)*j,marker='.',color = 'k')
for i in range(4):
    plt.text(JM+Jr/(pts-1)/2,Jmax-Jr/(pts-1)*i,Label[i],color=Color[i])

plt.xlabel("$J_2$",size=20)
plt.ylabel("$J_{3e}$",size=20)
plt.show()
