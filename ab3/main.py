import numpy as np
import inputs as inp
import common_functions as cf
from time import time as t
from colorama import Fore
from scipy.optimize import minimize
from pandas import read_csv
import csv
from pathlib import Path
import sys
#######
if len(sys.argv) == 2:
    ans = int(sys.argv[1])
else:
    ans = int(input("Which ansatz(int 0/1/2/3/4)? \n\t0 for 3x3, \n\t1 for q0, \n\t2 for (0,pi), \n\t3 for (pi,pi), \n\t4 for cuboc1(not working)\n\n\t"))
print(Fore.GREEN+'Computing ansatz ',inp.text_ans[ans],Fore.RESET)
#######
Ti = t()
J1 = inp.J1
Bnds = inp.Bnds[ans]
Pinitial = (0.52,0.14,0.11,0.24,2e-4) if ans < 3 else (0.52,0.12,0.05)
Pi = tuple(Pinitial)
non_converging_points = 0
reps = inp.repetitions
header = inp.header[ans]
csvfile = inp.csvfile[ans]
cf.CheckCsv(ans)

for j2,J2 in enumerate([-0.3]):#inp.rJ2):
    for j3,J3 in enumerate([-0.15]):#inp.rJ3):
        #check if this point has already been computed successfully
        is_n, P, rep = cf.is_new(J2,J3,ans)
        if not is_n:
            Pi = P
            continue
        if P[0] != 0:
            Pi = P
        Args = (J1,J2,J3,ans)
        a = t()
        print(Pi,'\nSigma:',cf.Sigma(Pi,Args),'\ntime:',t()-a)
        Tti = t()
        print(Fore.RED+"\nEvaluating energy of (J2,J3) = (",J2,",",J3,")",Fore.RESET)
        Stay = True
        rd = 0
        tempE = []; tempS = []; tempP = []; tempL = [];
        DataDic = {}
        while Stay:
            ti = t()
            print("Initial guess: ",header[5:]," = ",Pi)
            result = minimize(lambda x:cf.Sigma(x,Args),
                Pi,
                method = 'Nelder-Mead',
                bounds = Bnds,
                tol = 1e-6,
                options = {
                    'xatol':1e-6,
                    'fatol':1e-6,
                    'adaptive':True}
                )
            print(result.nfev,result.nit)
            Pf = result.x
            S = result.fun
            E,L = cf.totE(Pf,Args)
            tempE += [E]
            tempS += [S]
            tempP += [Pf]
            tempL += [L]
            print("After minimization:\n\tparams ",header[5:]," = ",Pf,"\n\tL = ",L,"\n\tSigma = ",S,"\n\tEnergy = ",E)
            if S<inp.cutoff:
                print("exiting minimization")
                Stay = False
                #save values
                Pi = Pf
                data = [J2,J3,E,S,L]
                for ind in range(len(data)):
                    DataDic[header[ind]] = data[ind]
                for ind2 in range(len(Pf)):
                    DataDic[header[5+ind2]] = Pf[ind2]
            elif rd < reps:
                rd += 1
                Pi = Pf
                print(Fore.BLUE+"Changing initial parameters to ",Pi,Fore.RESET)
            else:
                print(Fore.GREEN+"It's taking too long, pass to other point")
                Pi = Pinitial
                Stay = False
                arg = np.argmin(tempE)
                data = [J2,J3,tempE[arg],tempS[arg],tempL[arg]]
                for ind in range(len(data)):
                    DataDic[header[ind]] = data[ind]
                for ind2 in range(len(Pf)):
                    DataDic[header[5+ind2]] = tempP[arg][ind2]
                non_converging_points += 1
        if rep:
            cf.modify_csv(J2,J3,ans,DataDic)
        else:
            with open(csvfile,'a') as f:
                writer = csv.DictWriter(f, fieldnames = header)
                writer.writerow(DataDic)
        print(Fore.YELLOW+"time of (j2,j3) point: ",(t()-Tti)/60,Fore.RESET)

print(Fore.GREEN+"Non converging points: ",non_converging_points,' of ',len(inp.rJ2)*len(inp.rJ3),Fore.RESET)
print(Fore.YELLOW+"Total time: ",(t()-Ti)/60,Fore.RESET)
