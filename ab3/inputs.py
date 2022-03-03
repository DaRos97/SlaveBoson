import numpy as np

S = 0.5
#derivative
der_pts = 2
der_range = [0.001,0.001,0.001]
Jpts = 13
step = 0.1
sum_pts = 101
grid_pts = 11
#fixed
J1 = 1
z1 = 4
z2 = 4
z3 = 2
z = (z1,z2,z3)
#minimization
cutoff = 1e-6
#phase diagram
#ans = 0
Ji = -0.3
Jf = 0.3+step
rJ2 = np.arange(Ji,Jf,step)
rJ3 = np.arange(Ji,Jf,step)

#summation over BZ
maxK1 = np.pi
maxK2 = 2*np.pi/np.sqrt(3)
K1 = np.linspace(0,maxK1,sum_pts)  #Kx in BZ
K23 = np.linspace(0,maxK2,sum_pts)  #Ky in BZ
#text
text_ans = ['3x3','q0','(pi,pi)','(0,pi)']
dirname = 'Data/'
#csv
header = ['J2','J3','Energy','Sigma','A1','A2','A3','B1','B2','B3','L','mL']
csvfile = [dirname+'S'+str(S).replace('.','')+'-'+text_ans[ans]+'.csv' for ans in range(2)]
csvfile1 = [dirname+'S'+str(S).replace('.','')+'-'+text_ans[ans]+'_1.csv' for ans in range(2)]

cutoff_pts = 1e-12
