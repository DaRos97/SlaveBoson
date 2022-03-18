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
Pinitial = [(0.52,0.31,0.21,0.39,0.18),     #3x3
            (0.51,0.19,0.16,0.2,0.24),      #q0
            (0.5,0.07,0.03,0.13,0.07),    #0,pi
            (0.5,0.14,0.07),                    #pi,pi
            (0.5,0.,0.,0.,0.,0.,0.)]  #cuboc2
Pi = tuple(Pinitial[ans])
non_converging_points = 0
header = inp.header[ans]
csvfile = inp.csvfile[ans]
cf.CheckCsv(ans)

for j2,J2 in enumerate(inp.rJ2):
    for j3,J3 in enumerate(inp.rJ3):
        print(Fore.RED+"\nEvaluating energy of (J2,J3) = (",J2,",",J3,")",Fore.RESET)
        is_n, P, rep = cf.is_new(J2,J3,ans)
        if not is_n:
            print(Fore.GREEN+"already computed point"+Fore.RESET)
            Pi = P
            continue
        if P[0] != 0:
            print(Fore.RED+"Trying again this point"+Fore.RESET)
            Pi = P
        Args = (J1,J2,J3,ans)
        Tti = t()
        Stay = True
        rd = 0
        DataDic = {}
        result = minimize(lambda x:cf.Sigma(x,Args),
            Pi,
            method = 'Nelder-Mead',
            bounds = Bnds,
            #tol = 1e-6,
            options = {
                'maxiter':100*len(Pi),
                #'xatol':1e-6,
                'fatol':inp.cutoff,
                'adaptive':True}
            )
        Pf = result.x
        S = result.fun
        E,L = cf.totE(Pf,Args)
        print("After minimization:\n\tparams ",header[5:]," = ",Pf,"\n\tL = ",L,"\n\tSigma = ",S,"\n\tEnergy = ",E)
        #save values
        Pi = Pf
        data = [J2,J3,E,S,L]
        for ind in range(len(data)):
            DataDic[header[ind]] = data[ind]
        for ind2 in range(len(Pf)):
            DataDic[header[5+ind2]] = Pf[ind2]
        if rep:
            cf.modify_csv(J2,J3,ans,DataDic)
        else:
            with open(csvfile,'a') as f:
                writer = csv.DictWriter(f, fieldnames = header)
                writer.writerow(DataDic)
        if S<inp.cutoff:
            print("successful")
        else:
            print("did not converge")
            non_converging_points += 1
        print(Fore.YELLOW+"time of (j2,j3) point: ",(t()-Tti)/60,Fore.RESET)

print(Fore.GREEN+"Non converging points: ",non_converging_points,' of ',len(inp.rJ2)*len(inp.rJ3),Fore.RESET)
print(Fore.YELLOW+"Total time: ",(t()-Ti)/60,Fore.RESET)
