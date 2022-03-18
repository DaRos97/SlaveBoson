import inputs as inp
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

txt = inp.csvfile
data = []
Nans = 4
for ans in range(Nans):
    data.append(read_csv(txt[ans]))
ln = len(data[0])
## get indices of same (j2,j3) pts
inds = []
for i in range(ln):
    j2 = data[0]['J2'][i]
    j3 = data[0]['J3'][i]
    inds.append([i])
    for ans in range(1,Nans):
        for j in range(len(data[ans])):
            dif_2 = np.abs(data[ans]['J2'][j] - j2)
            dif_3 = np.abs(data[ans]['J3'][j] - j3)
            if dif_2 < inp.cutoff_pts and dif_3 < inp.cutoff_pts:
                inds[i].append(j)
                break
## get energies and sigmas of those points
res = []
for i in range(ln):
    tempE = []
    for ans in range(Nans):
        tempE.append(data[ans]['Energy'][inds[i][ans]])
    IminE = np.argmin(np.array(tempE))
    tempS = []
    for ans in range(Nans):
        tempS.append(data[ans]['Sigma'][inds[i][ans]])
    is_good = 1 if tempS[IminE] < inp.cutoff else 0
    res.append([IminE,is_good])

Color = ['r','b','g','c']
Marker = ['*','o']
plt.figure()
for i in range(ln):
    j2 = data[0]['J2'][i]
    j3 = data[0]['J3'][i]
    plt.scatter(j2,j3,marker=Marker[res[i][1]],color=Color[res[i][0]])


plt.show()
