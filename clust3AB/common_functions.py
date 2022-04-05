import inputs as inp
import ansatze as an
import numpy as np
from scipy import linalg as LA
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp2d, interp1d
from colorama import Fore
from pathlib import Path
import csv
from pandas import read_csv
import time
import os

import matplotlib.pyplot as plt
from matplotlib import cm
#Some parameters from inputs.py
m = inp.m
kp = inp.sum_pts
S = inp.S
J1 = inp.J1
#grid points
grid_pts = inp.grid_pts
####
J = np.zeros((2*m,2*m))
for i in range(m):
    J[i,i] = -1
    J[i+m,i+m] = 1
####
def sumEigs(P,L,args):
    N = an.Nk(P,L,args)
    res = np.zeros((m,grid_pts,grid_pts))
    for i in range(grid_pts):
        for j in range(grid_pts):
            Nk = N[:,:,i,j]
            try:
                K = LA.cholesky(Nk)
            except LA.LinAlgError:
                return 0
            except:
                print("Unexpected error")
                exit()
            temp = np.dot(K,J)
            temp = np.dot(temp,np.conjugate(K.T))
            res[:,i,j] = np.tensordot(J,LA.eigvalsh(temp),1)[:m]
    r2 = 0
    for i in range(m):
        func = interp2d(inp.kg[0],inp.kg[1],res[i],kind='cubic')
        temp = func(inp.Kp[0],inp.Kp[1])
        r2 += temp.ravel().sum()
    r2 /= (m*kp**2)
    return r2

####
def totE(P,args):
    res = minimize_scalar(lambda l: -totEl(tuple(P)+(l,),args),
            method = 'bounded',
            bounds = (0.4,1.5),
            options={'xatol':inp.prec_L}
            )
    L = res.x
    minE = -res.fun
    return minE, L
####
def totEl(P,args):
    L = P[-1]
    P = tuple(P[:-1])
    J1,J2,J3,ans = args
    J = (J1,J2,J3)
    j2 = np.sign(int(np.abs(J2)*1e8))
    j3 = np.sign(int(np.abs(J3)*1e8))
    res = 0
    if ans == '3x3':
        Pp = (P[0],0.,P[1]*j3,P[2*j3]*j3+P[1]*(1-j3),P[3*j2*j3]*j2*j3+P[2*j2*(1-j3)]*(1-j3)*j2,P[4*j3*j2]*j3*j2+P[3*j3*(1-j2)]*j3*(1-j2))
    elif ans == 'q0':
        Pp = (P[0],P[1]*j2,0.,P[2*j2]*j2+P[1]*(1-j2),P[3*j2]*j2,P[4*j3*j2]*j3*j2+P[2*j3*(1-j2)]*j3*(1-j2))
    elif ans == 'cb1':
        Pp = (P[0],
                P[1*j2]*j2,
                P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2),
                P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3),
                P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3),
                0)
    for i in range(3):
        res += inp.z[i]*(Pp[i]**2-Pp[i+3]**2)*J[i]/2
    res -= L*(2*inp.S+1)
    res += sumEigs(P,L,args)
    return res
####
def Sigma(P,args):
    t = time.time()
    J1,J2,J3,ans = args
    res = 0
    temp = []
    for i in range(len(P)):
        Ps = (P[i] + inp.der_range[i], P[i])
        e = np.ndarray(2)
        pp = np.array(P)
        for j in range(2):
            pp[i] = Ps[j]
            e[j] = totE(pp,args)[0]
        temp.append(((e[0]-e[1])/inp.der_range[i])**2)
    print(P)
    print(temp)
    res = np.array(temp).sum()
    return res

#################################################################
def CheckCsv(csvf):
    my_file = Path(csvf)
    ans = []
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
            N = (len(lines)-1)//4 +1
            for i in range(N):
                if float(lines[i*4+1].split(',')[4]) < 1e-8:
                    ans.append(lines[i*4+1].split(',')[0])
    res = []
    for a in inp.text_ans:
        t = 1
        for b in ans:
            if a == b:
                t *= 0
        if t:
            res.append(a)
    return res

def checkHessian(P,args):
    res = []
    for i in range(len(P)):
        pp = np.array(P)
        Der = []
        der = []
        ptsP = np.linspace(P[i]-inp.der_range[i],P[i]+inp.der_range[i],3)
        for j in range(3):
            pp[i] = ptsP[j]
            der.append(totE(pp,args)[0])        #uses at each energy evaluation the best lambda
        for l in range(2):
            de = np.gradient(der[l:l+2])
            dx = np.gradient(ptsP[l:l+2])
            derivative = de/dx
            f = interp1d(ptsP[l:l+2],derivative)
            Der.append(f((ptsP[l]+ptsP[l+1])/2))
        ptsPP = [(ptsP[l]+ptsP[l+1])/2 for l in range(2)]
        dde = np.gradient(Der)
        ddx = np.gradient(ptsPP)
        dderivative = dde/ddx
        f = interp1d(ptsPP,dderivative)
        res.append(f(P[i]))
    return np.array(res)

def checkInitial(J2,J3,ansatze):
    P = {}
    for file in os.listdir(inp.refDirname):     #find file in dir
        j2 = float(file[7:-5].split('_')[0])/10000
        j3 = float(file[7:-5].split('_')[1])/10000
        if j2 == J2 and j3 == J3:               #once found read it
            with open(inp.refDirname+file, 'r') as f:
                lines = f.readlines()
            N = (len(lines)-1)//4 + 1
            for i in range(N):
                data = lines[i*4+1].split(',')
                if float(data[4]) < 1e-8:              #if sigma small enough
                    P[data[0]] = data[6:]
                    for j in range(len(P[data[0]])):    #cast to float
                        P[data[0]][j] = float(P[data[0]][j])
    j2 = np.abs(J2) > inp.cutoff_pts    #bool for j2 not 0
    j3 = np.abs(J3) > inp.cutoff_pts
    if len(P) == 0:
        for ans in ansatze:
            P[ans] = [0.51]             #A1
            if j2 and (ans == 'q0' or ans == 'cb1'):
                P[ans].append(0.2)      #A2
            if j3 and (ans =='3x3' or ans == 'cb1'):
                P[ans].append(0.28)      #A3
            P[ans].append(0.176)         #B1
            if j2:
                P[ans].append(0.1)      #B2
            if j3 and ans != 'cb1':
                P[ans].append(0.1)      #B3
            if ans == 'cb1':
                P[ans].append(1.95)      #phiA1
    else:
        nP = {}
        for ans in P.keys():
            nP[ans] = []
            for i in np.nonzero(P[ans])[0]:
                nP[ans].append(P[ans][i])
        P = nP
    return P

def findBounds(J2,J3,ansatze):
    P = {}
    j2 = np.abs(J2) > inp.cutoff_pts
    j3 = np.abs(J3) > inp.cutoff_pts
    for ans in ansatze:
        for ans in ansatze:
            P[ans] = ((0,1),)             #A1
            if j2 and (ans == 'q0' or ans == 'cb1'):
                P[ans] = P[ans] + ((0,1),)      #A2
            if j3 and (ans =='3x3' or ans == 'cb1'):
                P[ans] = P[ans] + ((0,1),)      #A3
            P[ans] = P[ans] + ((0,0.5),)      #B1
            if j2:
                P[ans] = P[ans] + ((0,0.5),)      #B2
            if j3 and ans != 'cb1':
                P[ans] = P[ans] + ((0,0.5),)      #B3
            if ans == 'cb1':
                P[ans] = P[ans] + ((-np.pi,np.pi),)      #phiA1
    return P

def arrangeP(P,ans,J2,J3):
    j2 = np.sign(int(np.abs(J2)*1e8))
    j3 = np.sign(int(np.abs(J3)*1e8))
    newP = [P[0]]
    if ans == '3x3':
        newP.append(P[1]*j3)
        newP.append(P[2*j3]*j3+P[1]*(1-j3))
        newP.append(P[3*j2*j3]*j2*j3+P[2*j2*(1-j3)]*(1-j3)*j2)
        newP.append(P[4*j3*j2]*j3*j2+P[3*j3*(1-j2)]*j3*(1-j2))
    elif ans == 'q0':
        newP.append(P[1]*j2)
        newP.append(P[2*j2]*j2+P[1]*(1-j2))
        newP.append(P[3*j2]*j2)
        newP.append(P[4*j3*j2]*j3*j2+P[2*j3*(1-j2)]*j3*(1-j2))
    elif ans == 'cb1':
        newP.append(P[2*j2]*j2)
        newP.append(P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2))
        newP.append(P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3))
        newP.append(P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3))
        newP.append(P[-1])
    return tuple(newP)
