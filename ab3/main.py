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
print(Fore.GREEN+"\nUsing ansatz ",inp.text_ans[ans])
#######
Ti = t()
J1 = inp.J1
Bnds = ((0,1),(-1,1),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5))      #A1,A3,B1,B2,B3  --> sure? Put neg values possible
Pinitial = (0.5,0.1,0.05,0.05,0.05)       #initial guess of A1,A3,B1,B2,B3 from classical values?? see art...
Pi = Pinitial
non_converging_points = 0
reps = 3
header = inp.header
csvfile = inp.csvfile[ans]

Data = []
for j2,J2 in enumerate(inp.rJ2):
    for j3,J3 in enumerate(inp.rJ3):
        Args = (J1,J2,J3,ans)
        Tti = t()
        print(Fore.RED+"\nEvaluating energy of (J2,J3) = (",J2,",",J3,")",Style.RESET_ALL)
        Stay = True
        rd = 0
        tempE = []; tempS = []; tempP = []; tempL = []; tempmL = [];
        DataDic = {}
        while Stay:
            ti = t()
            print("Initial guess: (A1,A3,B1,B2,B3) = ",Pi)
            result = minimize(lambda x:cf.Sigma(x,Args),
                Pi,
                method = 'Nelder-Mead',
                bounds = Bnds,
                options = {
                    'adaptive':True}
                )
            Pf = result.x
            S = result.fun
            E,L,mL = cf.totE(Pf,Args)
            print("After minimization:\n\tparams = ",Pf,"\n\tL,mL = ",L,mL,"\n\tSigma = ",S,"\n\tEnergy = ",E)
            if S<inp.cutoff:
                print("exiting minimization")
                Stay = False
                #save values
                Pi = Pf
                data = [J2,J3,E,S,Pf[0],0.,0.,Pf[2],Pf[3],Pf[4],L,mL]
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
                data = [J2,J3,tempE[arg],tempS[arg],tempP[arg][0],0.,0.,tempP[arg][2],tempP[arg][3],tempP[arg][4],tempL[arg],tempmL[arg]]
                data[6-ans] = tempP[arg][1]
                for ind in range(len(data)):
                    DataDic[header[ind]] = data[ind]
                print("Keeping the best result:\n\tparams = ",data[4],data[5],data[6],"\n\tL,mL = ",data[7],data[8],"\n\tSigma = ",data[3],"\n\tEnergy = ",data[2],Fore.RESET)
                non_converging_points += 1
        Data.append(DataDic)
        print(Fore.YELLOW+"time of (j2,j3) point: ",t()-Tti,Fore.RESET)
#####   save externally to csvfile1
for l in range(len(Data)):      #probably easier way of doing this
    with open(csvfile,'w') as f:
        writer = csv.DictWriter(f, fieldnames = header)
        writer.writerow(Data[l])
print(Fore.GREEN+"Non converging points: ",non_converging_points,Fore.RESET)
print(Fore.YELLOW+"Total time: ",t()-Ti,Style.RESET_ALL)
