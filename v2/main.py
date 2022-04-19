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
print('(J2,J3) = ('+'{:5.4f}'.format(J2)+',{:5.4f}'.format(J3)+')\n')
#######
csvfile = inp.dirname+inp.dataDir+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
ansatze = cf.CheckCsv(csvfile)
Ti = t()
Pinitial = cf.checkInitial(J2,J3,ansatze)
Bnds = cf.findBounds(J2,J3,ansatze)
for ans in ansatze:
    print("Using ansatz: ",ans)
    header = inp.header[ans]
    Args = (J1,J2,J3,ans)
    Tti = t()
    Pi = Pinitial[ans]
    DataDic = {}
    HessDic = {}
    print(Pi,Bnds[ans])
    result = minimize(lambda x:cf.Sigma(x,Args),
       Pi,
       method = inp.method,
       bounds = Bnds[ans],
       options = {
           #           'maxiter':100*len(Pi),
           #           'adaptive':True,
           #           'xatol':inp.cutoff,
           'ftol':inp.cutoff}
       )
    Pf = tuple(result.x)
    L = Pf[0]
    P = Pf[1:]
    E = cf.totE(Pf,Args)
    S = result.fun
    #Add 0 values
    newP = cf.arrangeP(P,L,ans,J2,J3)
    ########Check of Hessian values
    a = cf.checkHessian(Pf,Args)
    hessian = cf.arrangeP(a[1:],a[0],ans,J2,J3)
    for i in range(len(hessian)):
        HessDic[header[5+i]] = hessian[i]
    #save values
    data = [ans,J2,J3,E,S]
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(newP)):
        DataDic[header[5+ind2]] = newP[ind2]
    print(DataDic)
    print(HessDic)
    cf.saveValues(DataDic,HessDic,csvfile)
    print("Time of ans",ans,": ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################
