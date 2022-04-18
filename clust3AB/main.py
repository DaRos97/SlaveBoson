import numpy as np
import inputs as inp
import common_functions as cf
from time import time as t
from colorama import Fore
from scipy.optimize import minimize
from scipy.optimize import differential_evolution as d_e
from pandas import read_csv
import csv
import sys
####### inputs
J1 = inp.J1
J2, J3 = inp.J[int(sys.argv[1])]
print('(J2,J3) = ('+'{:5.4f}'.format(J2)+',{:5.4f}'.format(J3)+')\n')
#######
csvfile = inp.DataDir+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
ansatze = cf.CheckCsv(csvfile)
ansatze = ['3x3']
Ti = t()
Pinitial = cf.FindInitialPoint(J2,J3,ansatze)
Bnds = cf.FindBounds(J2,J3,ansatze)
for ans in ansatze:
    Tti = t()
    print("Using ansatz: ",ans)
    header = inp.header[ans]
    Pi = Pinitial[ans]
    bnds = Bnds[ans]
    Args = (J1,J2,J3,ans)
    DataDic = {}
    HessDic = {}
    print("Initial point and bounds: \n",Pi,'\n',bnds,'\n')
    result = d_e(lambda x: cf.Sigma(x,Args),
            x0 = Pi,
            bounds = bnds,
            disp = True,
            tol = inp.cutoff,
            atol = inp.cutoff,
            maxiter = inp.MaxIter*len(Pi),
            workers = 1
            )
    print(result.success,result.message)
    Pf = tuple(result.x)
    E,L = cf.totE(Pf,Args)[:2]
    S = result.fun
    #Add 0 values
    newP = cf.arangeP(Pf,ans,J2,J3)
    ########Check of Hessian values
    hessian = cf.arangeP(cf.Hessian(Pf,Args),ans,J2,J3)
    for i in range(len(hessian)):
        HessDic[header[6+i]] = hessian[i]
    #save values
    data = [ans,J2,J3,E,S,L]
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(newP)):
        DataDic[header[6+ind2]] = newP[ind2]
    cf.SaveToCsv(DataDic,HessDic,csvfile)
    print(DataDic)
    print(HessDic)
    print("Time of ans",ans,": ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################
