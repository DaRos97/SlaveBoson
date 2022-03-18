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

#Some parameters from inputs.py
kp = inp.sum_pts
S = inp.S
J1 = inp.J1
#grid points
grid_pts = inp.grid_pts
####
def sumEigs(P,L,args):
    m = 6
    N = an.Nk(P,L,args)
    res = np.zeros((m,grid_pts,grid_pts))
    for i in range(grid_pts):
        for j in range(grid_pts):
            temp = LA.eigvals(N[:,:,i,j])
            if np.amax(np.abs(np.imag(temp))) > inp.complex_cutoff:   #not cool
                return 0
            res[:,i,j] = np.sort(np.real(temp))[m:] #also imaginary part if not careful
    result = 0
    for i in range(m):      #look difference without interpolation
        func = interp2d(inp.kg[0],inp.kg[1],res[i])
        temp = func(inp.Kp[0],inp.Kp[1])
        result += temp.ravel().sum()
    return result/(m*kp**2)
####
def totE(P,args):
    res = minimize_scalar(lambda l: -totEl(P,l,args),
            method = 'bounded',
            bounds = (0.4,1.5),
            options={'xatol':inp.prec_L}
            )
    L = res.x
    minE = -res.fun
    return minE, L
####
def totEl(P,L,args):
    J1,J2,J3,ans = args
    J = (J1,J2,J3)
    res = 0
    if ans == 0:    #3x3
        Pp = (P[0],0,P[1],P[2],P[3],P[4])
    elif ans == 1:  #q0
        Pp = (P[0],P[1],0,P[2],P[3],P[4])
    elif ans == 2:  #0,pi
        Pp = (P[0],P[1],P[2],P[3],P[4],0)
    elif ans == 3:  #pi,pi
        Pp = (P[0],0,0,P[1],P[2],0)
    elif ans == 4:
        Pp = (P[0],P[1],P[2],P[3],P[4],P[5])
    for i in range(3):
        res += inp.z[i]*(Pp[i]**2-Pp[i+3]**2)*J[i]/2
    res -= L*(2*inp.S+1)
    res += sumEigs(P,L,args)
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
    #print(P)
    #print(res,"time: ",time.time()-t,"\n")
    return res

#################################################################
def CheckCsv(ans):
    my_file = Path(inp.csvfile[ans])
    if my_file.is_file():
        with open(my_file,'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames
            a = 1
            for ind in range(len(headers)):
                if inp.header[ans][ind] != headers[ind]:
                    a *= 0
        if a:
            return 0
    with open(my_file,'w') as f:
        writer = csv.DictWriter(f, fieldnames = inp.header[ans])
        writer.writeheader()
    return 0

####
def is_new(J2,J3,ans):
    my_file = Path(inp.csvfile[ans])
    data = read_csv(my_file)
    for ind2,j2 in enumerate(data['J2']):
        dif2 = np.abs(J2-j2)
        dif3 = np.abs(J3-data['J3'][ind2])
        if dif2 < inp.cutoff_pts and dif3 < inp.cutoff_pts:
            S = data['Sigma'][ind2]
            L = data['L'][ind2]
            P = []
            for txt in inp.header[ans][5:]:
                P.append(data[txt][ind2])
            P = tuple(P)
            if S < inp.cutoff and L + np.abs(P).sum() > 0.1:
                return False,P,False
            else:
                return True,P,True
    return True,[0],False

def modify_csv(J2,J3,ans,dic):
    my_file = Path(inp.csvfile[ans])
    data = read_csv(my_file)
    for ind2, j2 in enumerate(data['J2']):
        dif2 = np.abs(J2-j2)
        dif3 = np.abs(J3-data['J3'][ind2])
        if dif2 < inp.cutoff_pts and dif3 < inp.cutoff_pts:
            ind = ind2
            break
    P = []
    for txt in inp.header[ans][5:]:
        P.append(dic[txt])
    P = tuple(P)
    if dic['Sigma'] < data['Sigma'][ind] and dic['L'] + np.abs(P).sum() > 0.1:
        for txt in inp.header[ans][2:]:
            data[txt][ind2] = dic[txt]
        data.to_csv(inp.csvfile[ans],index = False)



