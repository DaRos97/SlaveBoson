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
from time import time as t
import os

####
J = np.zeros((2*inp.m,2*inp.m))
for i in range(inp.m):
    J[i,i] = -1
    J[i+inp.m,i+inp.m] = 1
# Check also Hessians on the way --> more time. Calls only the totE func
def Sigma(P,*Args):
    #ti = t()
#    print("\nP = ",P)
    J1,J2,J3,ans,der_range,mi = Args
    j2 = int(np.sign(J2)*np.sign(int(np.abs(J2)*1e8)) + 1)   #j < 0 --> 0, j == 0 --> 1, j > 0 --> 2
    j3 = int(np.sign(J3)*np.sign(int(np.abs(J3)*1e8)) + 1)
    args = (J1,J2,J3,ans)
    init = totE(P,args)         #check initial point
    if init[2][0] == inp.shame1 or np.abs(init[1]-inp.L_bounds[0]) < 1e-3: #if not pos def H or converging to L_min
#        print("initial P not good")
        return inp.shame2
#    print("initial Params are good")
    res = 0
    temp = []
    final_Hess = []
    for i in range(len(P)): #for each parameter
        pp = np.array(P)
        dP = der_range[i]
        pp[i] = P[i] + dP
        init_plus = totE(pp,args)   #compute derivative
        der1 = (init_plus[0]-init[0])/dP
        #compute Hessian to see if it is of correct sign
#        if ans == 'cb1' and i == len(P)-1:
#            temp.append(der1**2)
#            continue
        pp[i] = P[i] + 2*dP
        init_2plus = totE(pp,args)
        der2 = (init_2plus[0]-init_plus[0])/dP
        final_Hess.append((der2-der1)/dP)
        hess = int(np.sign(final_Hess[-1]))    #order is important!!
        sign = inp.HS[ans][j2][j3][i]
        if sign == hess:
            temp.append(der1**2)     #add it to the sum
        else:
#            print("Wrong Hess")
            return inp.shame3
    res += np.array(temp).sum()
    #print(P,temp)
    #print("time: ",t()-ti)
    #print(Fore.YELLOW+"res for P = ",P," is ",res,' with L = ',test[1],Fore.RESET)
    final_E = init[0]
    final_L = init[1]
    final_gap = init[2][1]
#    print("Exiting Sigma with S = ",res," \nL = ",final_L," \ngap = ",final_gap," \nE = ",final_E)
    if mi:
        return res
    else:
        return res, final_Hess, final_E, final_L, final_gap

#### Computes the part of the energy given by the Bogoliubov eigen-modes
def sumEigs(P,L,args):
    N = an.Nk(P,L,args) #compute Hermitian matrix
    res = np.zeros((inp.m,inp.grid_pts,inp.grid_pts))
    for i in range(inp.grid_pts):
        for j in range(inp.grid_pts):
            Nk = N[:,:,i,j]
            try:
                K = LA.cholesky(Nk)     #not always the case since for some parameters of Lambda the eigenmodes are negative
            except LA.LinAlgError:      #matrix not pos def for that specific kx,ky
                return inp.shame1, 10      #if that's the case even for a single k in the grid, return a defined value
            temp = np.dot(np.dot(K,J),np.conjugate(K.T))    #we need the eigenvalues of M=KJK^+ (also Hermitian)
            res[:,i,j] = np.sort(np.tensordot(J,LA.eigvalsh(temp),1)[:inp.m])    #only diagonalization
    r2 = 0
    for i in range(inp.m):
        func = interp2d(inp.kg[0],inp.kg[1],res[i],kind='quintic')    #Interpolate the 2D surface
        temp = func(inp.Kp[0],inp.Kp[1])                            #sum over more points to increase the precision
        if i == 0:
            gap = np.amin(temp.ravel())
        r2 += temp.ravel().sum()
    r2 /= (inp.m*inp.sum_pts**2)
    return r2, gap

#### Computes Energy from Parameters P, by maximizing it wrt the Lagrange multiplier L. Calls only totEl function
def totE(P,args):
    res = minimize_scalar(lambda l: -totEl(P,l,args)[0],  #maximize energy wrt L with fixed P
            method = inp.L_method,
            bounds = inp.L_bounds,
            options={'xatol':inp.prec_L}
            )
    L = res.x
    minE = -res.fun
    temp = totEl(P,L,args)[1]
    return minE, L, temp

#### Computes the Energy given the paramters P and the Lagrange multiplier L
def totEl(P,L,args):
    J1,J2,J3,ans = args
    J = (J1,J2,J3)
    j2 = np.sign(int(np.abs(J2)*1e8))   #check if it is 0 or 1 --> problem for VERY small J2,J3 points
    j3 = np.sign(int(np.abs(J3)*1e8))
    res = 0
    if ans == '3x3':
        Pp = (P[0],0.,P[1]*j3,P[2*j3]*j3+P[1]*(1-j3),P[3*j2*j3]*j2*j3+P[2*j2*(1-j3)]*(1-j3)*j2,P[4*j3*j2]*j3*j2+P[3*j3*(1-j2)]*j3*(1-j2))
    elif ans == 'q0':
        Pp = (P[0],P[1]*j2,0.,P[2*j2]*j2+P[1]*(1-j2),P[3*j2]*j2,P[4*j3*j2]*j3*j2+P[2*j3*(1-j2)]*j3*(1-j2))
    elif ans == '0-pi' or ans == 'cb1' or ans == 'cb2':
        Pp = (  P[0],
                P[1*j2]*j2,
                P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2),
                P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3),
                P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3),
                0)
    elif ans == 'octa':
        Pp = (  P[0],
                P[1*j2]*j2,
                0,
                P[2*j2]*j2 + P[1*(1-j2)]*(1-j2),
                P[3*j2]*j2,
                P[4*j2*j2]*j2*j3 + P[2*j3*(1-j2)]*j3*(1-j2))
    for i in range(3):
        res += inp.z[i]*(Pp[i]**2-Pp[i+3]**2)*J[i]/2
    res -= L*(2*inp.S+1)
    eigEn = sumEigs(P,L,args)
    res += eigEn[0]
    return res, eigEn

#Computes the Hessian values of the energy, i.e. the second derivatives wrt the variational paramters. In this way
#we can check that the energy is a max in As and min in Bs (for J>0).
def Hessian(P,Args):
    res = []
    der_range = Args[-1]
    args = Args[:-1]
    for i in range(len(P)):
        pp = np.array(P)
        Der = []
        der = []
        ptsP = np.linspace(P[i]-der_range[i],P[i]+der_range[i],3)
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


#################################################################
#checks if the file exists and if it does, reads which ansatze have been computed and returns the remaining ones
#from the list of ansatze in inputs.py
def CheckCsv(csvf):
    my_file = Path(csvf)
    ans = []
    if my_file.is_file():
        with open(my_file,'r') as f:
            lines = f.readlines()
        N = (len(lines)-1)//4 +1        #4 lines per ansatz
        for i in range(N):
            data = lines[i*4+1].split(',')
            if float(data[4]) < inp.cutoff:# and np.abs(float(data[6])) > 0.5:    #if Sigma accurate enough and Lambda not equal to the lower bound
                ans.append(lines[i*4+1].split(',')[0])
    res = []
    for a in inp.list_ans:
        if a not in ans:
            res.append(a)
    return res

#Extracts the initial point for the minimization from a file in a reference directory specified in inputs.py
#If the file matching the j2,j3 point is not found initialize the initial point with default parameters defined in inputs.py
def FindInitialPoint(J2,J3,ansatze):
    P = {}  #parameters
    if Path(inp.ReferenceDir).is_dir():
        for file in os.listdir(inp.ReferenceDir):     #find file in dir
            j2 = float(file[7:-5].split('_')[0])/10000  #specific for the name of the file
            j3 = float(file[7:-5].split('_')[1])/10000
            if np.abs(j2-J2) < inp.cutoff_pts and np.abs(j3 - J3) < inp.cutoff_pts:         #once found read it
                with open(inp.ReferenceDir+file, 'r') as f:
                    lines = f.readlines()
                N = (len(lines)-1)//4 + 1
                for Ans in ansatze:
                    for i in range(N):
                        data = lines[i*4+1].split(',')
                        if data[0] == Ans:              #correct ansatz
                            P[data[0]] = data[7:]
                            for j in range(len(P[data[0]])):    #cast to float
                                P[data[0]][j] = float(P[data[0]][j])
    j2 = np.abs(J2) > inp.cutoff_pts    #bool for j2 not 0
    j3 = np.abs(J3) > inp.cutoff_pts
    #remove eventual 0 values
    nP = {}
    for ans in P.keys():
        nP[ans] = []
        for i in np.nonzero(P[ans])[0]:
            nP[ans].append(P[ans][i])
    P = nP
    #check eventual missing ansatze from the reference fileand initialize with default values
    for ans in ansatze:
        if ans in list(P.keys()):
            continue
        P[ans] = []
        P[ans] = [inp.Pi[ans]['A1']]             #A1
        if j2 and ans in inp.list_A2:
            P[ans].append(inp.Pi[ans]['A2'])      #A2
        if j3 and ans in inp.list_A3:
            P[ans].append(inp.Pi[ans]['A3'])      #A3
        P[ans].append(inp.Pi[ans]['B1'])         #B1
        if j2:
            P[ans].append(inp.Pi[ans]['B2'])      #B2
        if j3 and ans in inp.list_B3:
            P[ans].append(inp.Pi[ans]['B3'])      #B3
        if ans == 'cb1':
            P[ans].append(inp.Pi[ans]['phiA1'])      #phiA1
        if ans == 'cb12':
            P[ans].append(inp.Pi[ans]['phiA1'])      #phiA1
            if j2:
                P[ans].append(inp.Pi[ans]['phiB2'])      #phiA1
        if ans == 'cb2' or ans == 'octa':
            P[ans].append(inp.Pi[ans]['phiB1'])      #phiA1
    return P

#Constructs the bounds of the specific ansatz depending on the number and type of parameters involved in the minimization
def FindBounds(J2,J3,ansatze):
    B = {}
    j2 = np.abs(J2) > inp.cutoff_pts
    j3 = np.abs(J3) > inp.cutoff_pts
    for ans in ansatze:
        B[ans] = (inp.bounds['A1'],)             #A1
        if j2 and ans in inp.list_A2:
            B[ans] = B[ans] + (inp.bounds['A2'],)      #A2
        if j3 and ans in inp.list_A3:
            B[ans] = B[ans] + (inp.bounds['A3'],)      #A3
        B[ans] = B[ans] + (inp.bounds['B1'],)      #B1
        if j2:
            B[ans] = B[ans] + (inp.bounds['B2'],)      #B2
        if j3 and ans in inp.list_B3:
            B[ans] = B[ans] + (inp.bounds['B3'],)      #B3
        if ans == 'cb1':# or ans == 'cb2' or ans == 'octa':
            B[ans] = B[ans] + (inp.bounds['phiA1'],)      #phiB1
    return B

#Compute the derivative ranges for the various parameters of the minimization
def ComputeDerRanges(J2,J3,ansatze):
    R = {}
    j2 = np.abs(J2) > inp.cutoff_pts
    j3 = np.abs(J3) > inp.cutoff_pts
    for ans in ansatze:
        Npar = 2
        if j2:
            Npar +=1
            if ans in inp.list_A2:
                Npar +=1
        if j3 and ans in inp.list_A3:
            Npar +=1
        if j3 and ans in inp.list_B3:
            Npar +=1
        R[ans] = [inp.der_par for i in range(Npar)]
        if ans in inp.list_chiral:
            R[ans].append(inp.der_phi)
            if ans == 'cb12' and j2:
                R[ans].append(inp.der_phi)
    return R
#From the list of parameters obtained after the minimization constructs an array containing them and eventually 
#some 0 parameters which may be omitted because j2 or j3 are equal to 0.
def arangeP(P,ans,J2,J3):
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
    elif ans == '0-pi':
        newP.append(P[1*j2]*j2)
        newP.append(P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2))
        newP.append(P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3))
        newP.append(P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3))
    elif ans == 'cb1' or ans == 'cb2':
        newP.append(P[1*j2]*j2)
        newP.append(P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2))
        newP.append(P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3))
        newP.append(P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3))
        newP.append(P[-1])
    elif ans == 'cb12':
        newP.append(P[1*j2]*j2)
        newP.append(P[2*j3*j2]*j2*j3 + P[1*j3*(1-j2)]*j3*(1-j2))
        newP.append(P[3*j2*j3]*j2*j3 + P[2*j2*(1-j3)]*j2*(1-j3) + P[2*j3*(1-j2)]*j3*(1-j2) + P[1*(1-j2)*(1-j3)]*(1-j2)*(1-j3))
        newP.append(P[4*j3*j2]*j2*j3 + P[3*j2*(1-j3)]*j2*(1-j3))
        newP.append(P[5*j3*j2]*j3*j2 + P[4*j2*(1-j3)]*j2*(1-j3) + P[3*j3*(1-j2)]*j3*(1-j2) + P[-1]*(1-j2)*(1-j3))
        newP.append(P[-1]*j2)
    elif ans == 'octa':
        newP.append(P[1*j2]*j2)
        newP.append(P[2*j2]*j2 + P[1*(1-j2)]*(1-j2))
        newP.append(P[3*j2]*j2)
        newP.append(P[4*j3*j2]*j2*j3 + P[2*j3*(1-j2)]*j3*(1-j2))
        newP.append(P[-1])
    return tuple(newP)

#Save the dictionaries in the file given, rewriting the already existing data if precision is better
def SaveToCsv(Data,Hess,csvfile):
    N_ = 0
    if Path(csvfile).is_file():
        with open(csvfile,'r') as f:
            init = f.readlines()
    else:
        init = []
    ans = Data['ans']       #computed ansatz
    N = (len(init)-1)//4+1
    ac = False      #ac i.e. already-computed
    for i in range(N):
        D = init[i*4+1].split(',')
        if D[0] == ans:
            ac = True
            if (float(D[4]) > Data['Sigma'] and float(Data['L']) > inp.L_bounds[0]+1e-3) or (np.abs(float(D[7])) < 0.5 and np.abs(Data['A1']) > 0.5):
                N_ = i+1
    ###
    header = inp.header[ans]
    if N_:
        with open(csvfile,'w') as f:
            for i in range(4*N_-4):
                f.write(init[i])
        with open(csvfile,'a') as f:
            writer = csv.DictWriter(f, fieldnames = header)
            writer.writeheader()
            writer.writerow(Data)
            writer = csv.DictWriter(f, fieldnames = header[7:])
            writer.writeheader()
            writer.writerow(Hess)
        with open(csvfile,'a') as f:
            for l in range(4*N_,len(init)):
                f.write(init[l])
    elif not ac:
        with open(csvfile,'a') as f:
            writer = csv.DictWriter(f, fieldnames = header)
            writer.writeheader()
            writer.writerow(Data)
            writer = csv.DictWriter(f, fieldnames = header[7:])
            writer.writeheader()
            writer.writerow(Hess)

#### OLD SIGMA
#### Sum of the square of the derivatives of the energy wrt the mean field parameters (not Lambda)
def SigmaOld(P,*Args):
    #ti = t()
    J1,J2,J3,ans,der_range = Args
    args = (J1,J2,J3,ans)
    test = totE(P,args)         #check initial point
    if test[2] == inp.shame_value or np.abs(test[1]-inp.L_bounds[0]) < 1e-3:
        return inp.shame2
    res = 0
    temp = []
    for i in range(len(P)):
        pp = np.array(P)
        dP = der_range[i]
        pp[i] = P[i] + dP
        tempE = totE(pp,args)
        der = (tempE[0]-test[0])/dP
        if tempE[2] == inp.shame_value or np.abs(tempE[1]-inp.L_bounds[0]) < 1e-3:  #try in the other direction
            return inp.shame2
            pp[i] = P[i] - dP
            tempE = totE(pp,args)
            if tempE[2] == inp.shame_value or np.abs(tempE[1]-inp.L_bounds[0]) < 1e-3:
                return inp.shame2
        temp.append(der**2)
        if ans == 'cb1' and i == len(P) - 1:
            pp2 = np.array(P)
            pp2[i] = P[i] - dP
            tempE2 = totE(pp2,args)
            der2 = (test[0]-tempE2[0])/dP
            hess = (der-der2)/dP
            if hess < 0:
                res += inp.shame2
    res += np.array(temp).sum()
    #print(P,temp)
    #print("time: ",t()-ti)
    #print(Fore.YELLOW+"res for P = ",P," is ",res,' with L = ',test[1],Fore.RESET)
    return res

