import numpy as np
import inputs as inp
import common_functions as cf
from time import time as t
from colorama import Fore
from scipy.optimize import minimize
from pandas import read_csv
import csv
import sys
import os
####### inputs
J1 = inp.J1
J2, J3 = inp.J[int(sys.argv[1])]
print('(J2,J3) = ('+'{:5.4f}'.format(J2)+',{:5.4f}'.format(J3)+')\n')
#######
csvfile = inp.dirname+inp.dataDir+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
ansatze = cf.CheckCsv(csvfile)
Ti = t()
Pinitial = cf.checkInitial(J2,J3,ansatze)
Bnds = cf.findBounds(J2,J3,ansatze)
#ansatze = ['cb1']
for ans in ansatze:
    print("Using ansatz: ",ans)
    header = inp.header[ans]
    Args = (J1,J2,J3,ans)
    Tti = t()
    Pi = Pinitial[ans]
    DataDic = {}
    HessDic = {}
    result = minimize(lambda x:cf.Sigma(x,Args),
       Pi,
       method = inp.method,
       bounds = Bnds[ans],
       options = {
#           'maxiter':100*len(Pi),
           'ftol':inp.cutoff}
#           'adaptive':True}
       )
    Pf = tuple(result.x)
    E,L = cf.totE(Pf,Args)
    S = result.fun
    #Add 0 values
    newP = cf.arrangeP(Pf,ans,J2,J3)
    ########Check of Hessian values
    hessian = cf.arrangeP(cf.checkHessian(Pf,Args),ans,J2,J3)
    for i in range(len(hessian)):
        HessDic[header[6+i]] = hessian[i]
    #save values
    data = [ans,J2,J3,E,S,L]
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(newP)):
        DataDic[header[6+ind2]] = newP[ind2]
    cf.saveValues(DataDic,HessDic,csvfile)
    print(DataDic)
    print(HessDic)
    print("Time of ans",ans,": ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################
