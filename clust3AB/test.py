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
J1 = 1
J2, J3 = (0.0,0.0)
print("J2,J3=",J2,J3)
cutoff = inp.cutoff
#######
Ti = t()
ansatze = ['q0','cb1']
for ans in ansatze:
    header = inp.header[ans]
    print("Using ansatz: ",ans)
    Args = (J1,J2,J3,ans)
    Tti = t()
    Pi = (0.5,0.2,0.1,0.1,0.05)
    print("Initial guess: ",Pi)
    DataDic = {}
    HessDic = {}
    result = minimize(lambda x:cf.Sigma(x,Args),
       Pi,
       method = 'Nelder-Mead',
       bounds = inp.Bnds[ans],
       options = {
           'maxiter':100*len(Pi),
           'fatol':cutoff,
#           'ftol':cutoff#,
           'adaptive':True
           }
       )
    print(result.message,' with iterations: ',result.nit)
    Pf = tuple(result.x)
    E,L = cf.totE(Pf,Args)
    S = cf.Sigma(Pf,Args)
    print("Final parameters:",result.x)
    print("E,S,L:",E,S,L)
    ########check
    hessian = cf.checkHessian(Pf,Args)
    for i in range(len(Pf)):
        HessDic[header[6+i]] = hessian[i]
    #save values
    data = [ans,J2,J3,E,S,L]
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(Pf)):
        DataDic[header[6+ind2]] = Pf[ind2]
    print("Hessian values:",HessDic)
    print("Time of ans",ans,": ",(t()-Tti)/60,'\n')              ################

print("Total time: ",(t()-Ti)/60)                           ################
