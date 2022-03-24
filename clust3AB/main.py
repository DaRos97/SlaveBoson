import numpy as np
import inputs as inp
import common_functions as cf
from time import time as t
from colorama import Fore
from scipy.optimize import minimize
from pandas import read_csv
import csv
import sys
####### inputs
J1 = inp.J1
J2, J3 = inp.J[int(sys.argv[1])]
#######
csvfile = inp.dirname+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
ansatze = cf.CheckCsv(csvfile)
Ti = t()
for ans in ansatze:
    print("using ansatz ",ans)
    header = inp.header[ans]
    Args = (J1,J2,J3,ans)
    Tti = t()
    Pi = inp.initialPoint[ans]
    DataDic = {}
    HessDic = {}
    result = minimize(lambda x:cf.Sigma(x,Args),
       Pi,
       method = 'Nelder-Mead',
       bounds = inp.Bnds[ans],
       options = {
           'maxiter':50*len(Pi),
           'fatol':inp.cutoff,
           'adaptive':True}
       )
    Pf = tuple(result.x)
    E,L = cf.totE(Pf,Args)
    S = result.fun
    ########check
    hessian = cf.checkResult(Pf,Args)
    for i in range(len(Pf)):
        HessDic[header[6+i]] = hessian[i]
    #save values
    data = [ans,J2,J3,E,S,L]
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(Pf)):
        DataDic[header[6+ind2]] = Pf[ind2]
    with open(csvfile,'a') as f:
        writer = csv.DictWriter(f, fieldnames = header)
        writer.writeheader()
        writer.writerow(DataDic)
    with open(csvfile,'a') as f:
        writer = csv.DictWriter(f, fieldnames = header[5:])
        writer.writeheader()
        writer.writerow(HessDic)
    print("Time of ans: ",(t()-Tti)/60,'\n')              ################

print("Total time: ",(t()-Ti)/60)                           ################
