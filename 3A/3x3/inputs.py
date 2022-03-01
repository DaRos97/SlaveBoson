import numpy as np

S = 0.5
#derivative
der_pts = 2
der_range = [0.001,0.001,0.001]
Jpts = 5
J2pts = 100
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
Jf = 0.3
rJ = np.linspace(Ji,Jf,Jpts)
#summation over BZ
maxK1 = np.pi
maxK2 = 2*np.pi/np.sqrt(3)
K1 = np.linspace(0,maxK1,sum_pts)  #Kx in BZ
K23 = np.linspace(0,maxK2,sum_pts)  #Ky in BZ
K26 = np.linspace(0,maxK2/2,sum_pts)
#text
text_ans = ['(0,0)','(pi,0)','(pi,pi)','(0,pi)']
text_params = ['Energies','Sigmas','Params','Ls']
Save = True
#csv
header = ['J2','J3','Energy','Sigma','A1','A2','A3','L','mL']
csvfile = ['S'+str(S).replace('.','')+text_ans[ans]+'.csv' for ans in range(2)]


