import numpy as np
import matplotlib.pyplot as plt
from time import time as t

def ferr(J1,J2,J3):
    return 4*J1+4*J2+2*J3
def q0(J1,J2,J3):
    return -2*J1-2*J2+2*J3
def x3(J1,J2,J3):
    return -2*J1+4*J2-J3
def octa(J1,J2,J3):
    return 2*J3
def cb1(J1,J2,J3):
    return -2*J1+2*J2-2*J3
def cb2(J1,J2,J3):
    return 2*J1-2*J2-2*J3

lim = 3
N = 31
pts = np.linspace(-lim,lim,N)
ti = t()
J1 = 1
res1 = np.zeros((N,N),dtype=int)
temp = np.zeros(6)
for j in range(N):
    for l in range(N):
        temp[0] = q0(J1,pts[j],pts[l])
        temp[1] = x3(J1,pts[j],pts[l])
        temp[2] = octa(J1,pts[j],pts[l])
        temp[3] = cb1(J1,pts[j],pts[l])
        temp[4] = cb2(J1,pts[j],pts[l])
        temp[5] = ferr(J1,pts[j],pts[l])
        res1[j,l] = np.argmin(temp)
J1 = -1
res2 = np.zeros((N,N),dtype=int)
temp = np.zeros(6)
for j in range(N):
    for l in range(N):
        temp[0] = q0(J1,pts[j],pts[l])
        temp[1] = x3(J1,pts[j],pts[l])
        temp[2] = octa(J1,pts[j],pts[l])
        temp[3] = cb1(J1,pts[j],pts[l])
        temp[4] = cb2(J1,pts[j],pts[l])
        temp[5] = ferr(J1,pts[j],pts[l])
        res2[j,l] = np.argmin(temp)

Color=('r','b','g','m','c','k')
txt = ('q0','x3','octa','cb1','cb2','ferro')
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
for i in range(N):
    for j in range(N):
        plt.scatter(pts[i],pts[j],color=Color[res1[i,j]],marker='o')
plt.title('J1=1')
plt.subplot(1,2,2)
for i in range(N):
    for j in range(N):
        plt.scatter(pts[i],pts[j],color=Color[res2[i,j]],marker='o')
plt.title('J1=-1')
for i in range(6):
    plt.text(lim+0.5,lim+0.5-i/2,txt[i],color=Color[i])
plt.show()
