import numpy as np
from scipy import linalg as LA
from scipy.optimize import minimize_scalar
from scipy.interpolate import interp1d,interp2d
import inputs as inp
import ansatze as an
from colorama import Fore
from pathlib import Path
import csv
from pandas import read_csv
import time
import os

#Some parameters from inputs.py
kp = inp.sum_pts
S = inp.S
J1 = inp.J1
#grid points
grid_pts = inp.grid_pts
####
def newSumEigs(P,L,args):
    m = 6
    N = an.Nk(P,L,args,'h')
    J = np.zeros((2*m,2*m))
    for i in range(m):
        J[i,i] = -1
        J[i+m,i+m] = 1
    res = np.zeros((2*m,grid_pts,grid_pts))
    rr = 0
    for i in range(grid_pts):
        for j in range(grid_pts):
            Nk = N[:,:,i,j]
            e = LA.eigvalsh(Nk)
            if np.amin(e) < 0:
                return 0
            K = LA.cholesky(Nk)
            K_ = np.conjugate(K.T)
            temp = np.dot(K,J)
            temp = np.dot(temp,K_)
            res[:,i,j] = np.tensordot(J,LA.eigvalsh(temp),1)
            rr += res[:,i,j].ravel().sum()
    rr /= (2*m*grid_pts**2)
    return rr

####
def sumEigs(P,L,args):
    m = 6
    N = an.Nk(P,L,args,'z')
    res = np.zeros((m,grid_pts,grid_pts))
    for i in range(grid_pts):
        for j in range(grid_pts):
            temp = LA.eigvals(N[:,:,i,j])
            if np.amax(np.abs(np.imag(temp))) > inp.complex_cutoff:   #not cool
                return 0
            res[:,i,j] = np.sort(np.real(temp))[m:] #also imaginary part if not carefuli
    result = 0
    for i in range(m):      #look difference without interpolation
        func = interp2d(inp.kg[0],inp.kg[1],res[i])
        temp = func(inp.Kp[0],inp.Kp[1])
        result += temp.ravel().sum()
    result /= (m*kp**2)
    return result
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
    res = 0
    if ans == '3x3':
        Pp = (P[0],0,P[1],P[2],P[3],P[4])
    elif ans == 'q0':
        Pp = (P[0],P[1],0,P[2],P[3],P[4])
    elif ans == '0-pi':
        Pp = (P[0],P[1],P[2],P[3],P[4],0)
    elif ans == 'pi-pi':
        Pp = (P[0],0,0,P[1],P[2],0)
    elif ans == 'cb1':
        Pp = (P[0],0,0,P[1],P[2],P[3])
    for i in range(3):
        res += inp.z[i]*(Pp[i]**2-Pp[i+3]**2)*J[i]/2
    res -= L*(2*inp.S+1)
    res += newSumEigs(P,L,args)
    return res
####
def Sigma(P,args):
    #t=time.time()
    J1,J2,J3,ans = args
    res = 0
    ran = inp.der_range
    for i in range(len(P)):
        e = np.ndarray(inp.der_pts)
        rangeP = np.linspace(P[i]-ran[i],P[i]+ran[i],inp.der_pts)
        pp = np.array(P)
        for j in range(inp.der_pts):
            pp[i] = rangeP[j]
            e[j] = totE(pp,args)[0]        #uses at each energy evaluation the best lambda
        de = np.gradient(e)
        dx = np.gradient(rangeP)
        der = de/dx
        f = interp1d(rangeP,der)
        res += f(P[i])**2
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
#                if lines[i*4+1].split(',')[4] < inp.cutoff:
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

def checkInitial(J2,J3):
    for file in os.listdir(inp.refDirname):
        j2 = float(file[7:-5].split('_')[0])/10000
        j3 = float(file[7:-5].split('_')[1])/10000
        if j2 == J2 and j3 == J3:
            with open(inp.refDirname+file, 'r') as f:
                lines = f.readlines()
            N = (len(lines)-1)//4 + 1
            P = {}
            for i in range(N):
                data = lines[i*4+1].split(',')
                P[data[0]] = data[6:]
                for j in range(len(P[data[0]])):
                    P[data[0]][j] = float(P[data[0]][j])
            return P
