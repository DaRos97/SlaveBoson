import numpy as np

S = 0.5
#derivative
der_pts = 2
der_range = [0.001,0.001,0.001,0.001]
PD_pts = 11
sum_pts = 101
grid_pts = 11
#fixed
J1 = 1
z1 = 4
z2 = 4
z3 = 2
z = (z1,z2,z3)
#minimization
cutoff = 1e-8
#phase diagram
iJ2 = -0.6
fJ2 = 0#0.3
iJ3 = -0.3
fJ3 = 0.3
rJ2 = [[0],np.linspace(iJ2,fJ2,PD_pts)]
rJ3 = [np.linspace(iJ3,fJ3,PD_pts),[0]]
rJ = (rJ2,rJ3)
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
dirname = 'DataS'+str(S).replace('.','')+'/'
text_file = [[dirname+text_params[i]+'-'+text_ans[ans]+'PDpts='+str(int(PD_pts))+'a.npy' for i in range(len(text_params))] for ans in range(2)]

