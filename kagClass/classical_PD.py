import numpy as np
import matplotlib.pyplot as plt
from time import time as t
import sys

Color=('k','orange','r','b','g','m','c')
txt = ('ferr along z','q=0','3x3','octahedral','cuboc1','cuboc2', 'ferr in-plane')
def energy(J1,J2,J3,J3p,t1,t3):
    k = 4*J1*(np.cos(t1)-1) + 2*J3*(np.cos(t3)-1)
    psi = 0 if k>0 else np.pi/2
    #
    ferr =  4*J1*(np.cos(psi)**2 + np.cos(t1)*np.sin(psi)**2) + 4*J2 + 2*J3p*(np.cos(psi)**2 + np.cos(t3)*np.sin(psi)**2) + 4*J3
    q0 = -2*J1*np.cos(t1)-2*J2+2*J3p*np.cos(t3) + 4*J3
    x3 = -2*J1*(np.cos(t1)+np.sqrt(3)*np.sin(t1))+4*J2-J3p*(np.cos(t3)-np.sqrt(3)*np.sin(t3)) - 2*J3
    octa =  2/3*J3p*(1+2*np.cos(t3)) - 4*J3
    cb1 = -2/3*J1*(1+2*np.cos(t1))+2*J2-2/3*J3p*(1+2*np.cos(t3))
    cb2 = 2/3*J1*(1+2*np.cos(t1))-2*J2-2/3*J3p*(1+2*np.cos(t3))
    E = (ferr,q0,x3,octa,cb1,cb2,psi)
    return np.array(E)
#################### PARAMETERS
lim = 3
if len(sys.argv) > 1:
    N = int(sys.argv[1])
else:
    N = 20
####################
pts = np.linspace(-lim,lim,N)
temp = np.zeros(6)
plt.figure(figsize=(16,8))
#################### AFM, no DM
J1 = -1
J3 = 0
DM1 = 0
DM3 = 0
plt.subplot(1,2,1)
for j in range(N):
    for l in range(N):
        j1 = J1
        j2 = pts[j]
        j3p = pts[l]
        j3 = J3
        temp = energy(j1,j2,j3,j3p,DM1,DM3)
        minE = np.argmin(temp[:-1])
        if minE == 0:
            if temp[-1] == 0:
                minE = 0
            else:
                minE = 6
        plt.scatter(j2,j3p,color=Color[minE],marker='o')
plt.hlines(0,-lim,lim,color='y')
plt.vlines(0,-lim,lim,color='y')
plt.title('No DM,$J_1=-1$')
plt.ylabel('J3')
plt.xlabel('J2')
#################### AFM, DM
J1 = -1
J3 = 0
DM1 = 4/3*np.pi
DM3 = np.pi*2/3
plt.subplot(1,2,2)
for j in range(N):
    for l in range(N):
        j1 = J1
        j2 = pts[j]
        j3p = pts[l]
        j3 = J3
        temp = energy(j1,j2,j3,j3p,DM1,DM3)
        minE = np.argmin(temp[:-1])
        if minE == 0:
            if temp[-1] == 0:
                minE = 0
            else:
                minE = 6
        plt.scatter(j2,j3p,color=Color[minE],marker='o')
plt.hlines(0,-lim,lim,color='y')
plt.vlines(0,-lim,lim,color='y')
plt.title('DM, $J_1=-1$')
plt.xlabel('J2')
#####
for i in range(len(txt)):
    plt.text(lim+0.5,lim+0.5-i/2,txt[i],color=Color[i])
plt.show()
