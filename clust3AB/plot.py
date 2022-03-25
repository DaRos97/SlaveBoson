import numpy as np
import inputs as inp
import matplotlib.pyplot as plt
import os

E = []
for file in os.listdir(inp.dirname):
    with open(inp.dirname+file, 'r') as f:
        lines = f.readlines()
    N = (len(lines)-1)//4 + 1
    tempE = []
    for i in range(N):
        data = lines[i*4+1].split(',')
        tempE.append(float(data[3]))     #ans,J2,J3,E,S
    minInd = np.argmin(np.array(tempE))
    E.append(lines[minInd*4+1].split(',')[:5])

pts = len(E)
Color = {'3x3': 'b',
         'q0':   'r',
         'cb1':  'm'}
plt.figure(figsize=(8,8))

for p in range(pts):
    conv = '^'
    if float(E[p][4]) < 1e-5:
        conv = 'o'
    plt.scatter(float(E[p][1]),float(E[p][2]),color=Color[E[p][0]],marker = conv)


plt.show()
