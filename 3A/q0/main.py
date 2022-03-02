import numpy as np
import functions as fs
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import inputs1 as inp
from time import time as t
from colorama import Fore, Style
from scipy.optimize import minimize, minimize_scalar
import csv 
#######
ans = 1
print(Fore.GREEN+"\nUsing ansatz ",inp.text_ans[ans])
#######
Ti = t()
J1 = inp.J1
Bnds = ((0,1),(0,1))      #A1,A2  --> sure? Put neg values possible
Pinitial = (0.5,0.1)       #initial guess of A1,A2 from classical values?? see art...
Pi = Pinitial
non_converging_points = 0
reps = 3
header = inp.header
csvfile = inp.csvfile[ans]

fs.CheckCsv(csvfile)       #checks if the file exists and if not initializes it with the header
rJ = fs.ComputeRanges(csvfile)
J3 = 0.
rJ3 = np.linspace(inp.Ji,inp.Jf,inp.Jpts)

Data = []
for j2,J2 in enumerate(rJ):
    Args = (J1,J2,J3)
    Tti = t()
    print(Fore.RED+"\nEvaluating energy of (J2,J3) = (",J2,",",J3,")",Style.RESET_ALL)
    Stay = True
    rd = 0
    tempE = []; tempS = []; tempP = []; tempL = []; tempmL = [];
    DataDic = {}
    while Stay:
        ti = t()
        print("Initial guess: (A1,A2) = ",Pi)
        result = minimize(lambda x:fs.Sigma(x,Args),
            Pi,
            method = 'Nelder-Mead',
            bounds = Bnds,
            options = {
                'adaptive':True}
            )
        Pf = result.x
        #checks
        if abs(J2) < 1e-15:
            Pf[1] = 0.
        S = fs.Sigma(Pf,Args)
        E,L,mL = fs.totE(Pf,Args)
        print("After minimization:\n\tparams = ",Pf,"\n\tL,mL = ",L,mL,"\n\tSigma = ",S,"\n\tEnergy = ",E)
        if S<inp.cutoff:
            print("exiting minimization")
            Stay = False
            #save values
            Pi = Pf
            data = [J2,J3,E,S,Pf[0],Pf[1],0.,L,mL]
            for ind in range(len(data)):
                DataDic[header[ind]] = data[ind]
        elif rd <= reps:
            tempE += [E]
            tempS += [S]
            tempP += [Pf]
            tempL += [L]
            tempmL += [mL]
            rd += 1
            Pi = (Pinitial[0]+0.05*rd,Pinitial[1]+0.05*rd)
            print(Fore.BLUE+"Changing initial parameters to ",Pi,Fore.RESET)
        else:
            print(Fore.GREEN+"It's taking too long, pass to other point")
            Pi = Pinitial
            Stay = False
            arg = np.argmin(tempE)
            data = [J2,J3,tempE[arg],tempS[arg],tempP[arg][0],tempP[arg][1],0.,tempL[arg],tempmL[arg]]
            for ind in range(len(data)):
                DataDic[header[ind]] = data[ind]
            print("Keeping the best result:\n\tparams = ",data[4],data[5],data[6],"\n\tL,mL = ",data[7],data[8],"\n\tSigma = ",data[3],"\n\tEnergy = ",data[2],Fore.RESET)
            non_converging_points += 1
    Data.append(DataDic)
    #compute other points
    for Jj3 in rJ3:
        tempDD = dict(DataDic)
        tempDD['Energy'] = DataDic['Energy'] + Jj3*inp.z[2]*inp.S**2/2
        tempDD['J3'] = Jj3
        Data.append(tempDD)
    print(Fore.YELLOW+"time of (j2,j3) point: ",t()-Tti,Fore.RESET)
#####   save externally to .csv
print('\nSaving externally to ',csvfile)
for l in range(len(Data)):      #probably easier way of doing this
    with open(csvfile,'a') as f:
        writer = csv.DictWriter(f, fieldnames = header)
        writer.writerow(Data[l])
print(Fore.GREEN+"Non converging points: ",non_converging_points,Fore.RESET)
print(Fore.YELLOW+"Total time: ",t()-Ti,Style.RESET_ALL)
