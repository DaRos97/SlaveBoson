import numpy as np
import inputs as inp
import common_functions as cf
from time import time as t
from colorama import Fore, Style
from scipy.optimize import minimize, minimize_scalar
from pandas import read_csv
import csv
from pathlib import Path
#######
ans = int(input("Which ansatz(int 0/1)? [0 for 3x3 and 1 for q0]\n\t"))
print(Fore.GREEN+"\nUsing ansatz ",inp.text_ans[ans],Fore.RESET)
#######
Ti = t()
J1 = inp.J1
Bnds = ((0,1),(0,1))      #A1,A3  --> sure? Put neg values possible
Pinitial = (0.5,0.1)       #initial guess of A1,A3 from classical values?? see art...
Pi = Pinitial
non_converging_points = 0
reps = 3
header = inp.header
csvfile1 = inp.csvfile1[ans]
csvfile = inp.csvfile[ans]

cf.CheckCsv(csvfile1)       #checks if the file exists and if not initializes it with the header
rJ = cf.ComputeRanges(csvfile1,ans)
Jm = 0.

Data1 = []
for J in rJ:
    J2 = Jm if ans == 0 else J
    J3 = J if ans == 0 else Jm
    Args = (J1,J2,J3,ans)
    Tti = t()
    print(Fore.RED+"\nEvaluating energy of (J2,J3) = (",J2,",",J3,")",Style.RESET_ALL)
    Stay = True
    rd = 0
    tempE = []; tempS = []; tempP = []; tempL = []; tempmL = [];
    DataDic = {}
    while Stay:
        ti = t()
        print("Initial guess: (A1,A3) = ",Pi)
        result = minimize(lambda x:cf.Sigma(x,Args),
            Pi,
            method = 'Nelder-Mead',
            bounds = Bnds,
            options = {
                'adaptive':True}
            )
        Pf = result.x
        #checks
        S = result.fun
        E,L,mL = cf.totE(Pf,Args)
        print("After minimization:\n\tparams = ",Pf,"\n\tL,mL = ",L,mL,"\n\tSigma = ",S,"\n\tEnergy = ",E)
        if S<inp.cutoff:
            print("exiting minimization")
            Stay = False
            #save values
            Pi = Pf
            data = [J2,J3,E,S,Pf[0],0.,0.,L,mL]
            data[6-ans] = Pf[1]
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
            data = [J2,J3,tempE[arg],tempS[arg],tempP[arg][0],0.,0.,tempL[arg],tempmL[arg]]
            data[6-ans] = tempP[arg][1]
            for ind in range(len(data)):
                DataDic[header[ind]] = data[ind]
            print("Keeping the best result:\n\tparams = ",data[4],data[5],data[6],"\n\tL,mL = ",data[7],data[8],"\n\tSigma = ",data[3],"\n\tEnergy = ",data[2],Fore.RESET)
            non_converging_points += 1
    Data1.append(DataDic)
    print(Fore.YELLOW+"time of (j2,j3) point: ",t()-Tti,Fore.RESET)
#####   save externally to csvfile1
for l in range(len(Data1)):      #probably easier way of doing this
    with open(csvfile1,'a') as f:
        writer = csv.DictWriter(f, fieldnames = header)
        writer.writerow(Data1[l])
#####   extract csvfile1 dictionaries
data = read_csv(Path(csvfile1))
Data2 = data.to_dict('records')
#####   compute other points
rJm = np.arange(inp.Ji,inp.Jf,inp.step)
Data = []
for l in range(len(Data2)):
    for jm in rJm:
        tempDD = dict(Data2[l])
        txt = 'J2' if ans == 0 else 'J3'
        tempDD['Energy'] = Data2[l]['Energy'] + jm*inp.z[ans+1]*inp.S**2/2
        tempDD[txt] = jm
        Data.append(tempDD)
#####   save externally to csvfile
with open(Path(csvfile),'w') as f:
    writer = csv.DictWriter(f, fieldnames = inp.header)
    writer.writeheader()
for l in range(len(Data)):
    with open(csvfile,'a') as f:
        writer = csv.DictWriter(f, fieldnames = header)
        writer.writerow(Data[l])
print(Fore.GREEN+"Non converging points: ",non_converging_points,Fore.RESET)
print(Fore.YELLOW+"Total time: ",t()-Ti,Style.RESET_ALL)
