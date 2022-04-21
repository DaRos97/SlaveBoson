import numpy as np
import inputs as inp
import common_functions as cf
from time import time as t
from scipy.optimize import differential_evolution as d_e
from pandas import read_csv
import csv
import sys
import os
print("There are ",os.cpu_count()," CPUs available")
####### inputs
J1 = inp.J1
J2, J3 = (0,0)
S,DM = inp.DM_PD[int(sys.argv[1])]
DataDir = '../Data/test/dm_5/'
#######
filename = inp.DataDir+'testDM_S-DM='+str("{:4.4f}".format(DM)).replace('.','')+'-'+str("{:4.4f}".format(DM)).replace('.','')+'.csv'
ansatze = ['3x3','q0','cb1']
Ti = t()
Pinitial = {'3x3':(0.51,0.17), 'q0':(0.51,0.18), 'cb1':(0.51,0.17,1.95)}
Bnds = {'3x3':((-1,1),(-0.5,0.5)), 'q0':((-1,1),(-0.5,0.5)), 'cb1':((-1,1),(-0.5,0.5),(0,2*np.pi))}
Header = {'3x3':['ans','J2','J3','Energy','Sigma','L','A1','B1'],'q0':['ans','J2','J3','Energy','Sigma','L','A1','B1'],'cb1':['ans','J2','J3','Energy','Sigma','L','A1','B1','phiA1']}
DerRange = cf.ComputeDerRanges(J2,J3,ansatze)
print("Using S-DM = ",S,DM)
for ans in ansatze:
    print("Using ansatz: ",ans)
    inp.DM1 = DM
    inp.S = S
    Tti = t()
    der_range = DerRange[ans]
    header = Header[ans]
    Args1 = (J1,J2,J3,ans,der_range)
    Args = (J1,J2,J3,ans)
    result = d_e(cf.Sigma,
        args = Args1,
        x0 = Pinitial[ans],
        bounds = Bnds[ans],
        popsize = 15,
        maxiter = inp.MaxIter,
        disp = True,
        tol = inp.cutoff,
        atol = inp.cutoff,
        updating='deferred' if inp.mp_cpu == -1 else 'immediate',
        workers = inp.mp_cpu     #parallelization
        )
    Pf = tuple(result.x)
    S = result.fun
    E,L = cf.totE(Pf,Args)[:2]
    #
    DataDic = {}
    data = [ans,J2,J3,E,S,L]
    for ind in range(len(data)):
        DataDic[header[ind]] = data[ind]
    for ind2 in range(len(Pf)):
        DataDic[header[6+ind2]] = Pf[ind2]
    #
    HessDic = {}
    hessian = cf.Hessian(Pf,Args1)
    for i in range(len(hessian)):
        HessDic[header[6+i]] = hessian[i]
    #save values
    with open(filename, 'a') as f:
        writer = csv.DictWriter(f, fieldnames = header)
        writer.writeheader()
        writer.writerow(DataDic)
        writer = csv.DictWriter(f, fieldnames = header[6:])
        writer.writeheader()
        writer.writerow(HessDic)
    print("Time of point : ",'{:5.2f}'.format((t()-Tti)/60),' minutes\n')              ################

print("Total time: ",'{:5.2f}'.format((t()-Ti)/60),' minutes.')                           ################
