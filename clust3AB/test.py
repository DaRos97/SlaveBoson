import numpy as np
import inputs as inp
import common_functions as cf
from time import time as t
from scipy.optimize import differential_evolution as d_e
from pandas import read_csv
import csv
import sys
####### inputs
J1 = inp.J1
J2, J3 = (0,0)
print('(J2,J3) = ('+'{:5.4f}'.format(J2)+',{:5.4f}'.format(J3)+')\n')
#######
csvfile = inp.DataDir+'testDM_3x3.csv'
ansatze = ['cb1']
Ti = t()
Pinitial = cf.FindInitialPoint(J2,J3,ansatze)
Bnds = cf.FindBounds(J2,J3,ansatze)
ans = ansatze[0]
#range_phi = np.arange(0,2*np.pi+np.pi/3,np.pi/3)
range_phi = np.arange(0,2.4,0.2)
print("Using ansatz: ",ans)
for phi in range_phi:
    print("Using phi = ",phi)
    inp.DM1 = phi
    print(inp.DM1)
    Tti = t()
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
            popsize = 15,
            maxiter = inp.MaxIter,
            disp = True,
            tol = inp.cutoff,
            atol = inp.cutoff,
            workers = 1     #parallelization --> see if necessary
            )
    Pf = tuple(result.x)
    S = result.fun
    E,L = cf.totE(Pf,Args)[:2]
    #Add 0 values
    newP = cf.arangeP(Pf,ans,J2,J3)
    ########Compute Hessian values
    hessian = cf.arangeP(cf.Hessian(Pf,Args),ans,J2,J3)
    for i in range(len(hessian)):
        HessDic[header[6+i]] = hessian[i]
    #save values
    data = [ans,J2,J3,E,S,L]
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(newP)):
        DataDic[header[6+ind2]] = newP[ind2]
    #cf.SaveToCsv(DataDic,HessDic,csvfile)
    print(DataDic)
    print(HessDic)
    print("Time of phi point : ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################
