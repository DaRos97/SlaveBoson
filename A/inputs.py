import numpy as np

S = 0.2
#derivative
der_pts = 2
der_range3 = [0.01,0.01,0.01]
der_range6 = [0.01,0.01,0.01,0.01]
PD_pts = 5
sum_pts = 51
#fixed
J1 = 1
z1 = 4
z2 = 4
z3 = 2
z = [z3,z2]
#minimization
cutoff = 1e-4
#phase diagram
iJ2 = 0
fJ2 = 0.3
iJ3 = 0
fJ3 = 0.3
rJ2 = [[0],np.linspace(iJ2,fJ2,PD_pts),np.linspace(iJ2,fJ2,PD_pts)]
rJ3 = [np.linspace(iJ3,fJ3,PD_pts),[0],np.linspace(iJ3,fJ3,PD_pts)]
#summation over BZ
maxK1 = np.pi
maxK2 = 2*np.pi/np.sqrt(3)
K1 = np.linspace(0,maxK1,sum_pts)  #Kx in BZ
K23 = np.linspace(0,maxK2,sum_pts)  #Ky in BZ
K26 = np.linspace(0,maxK2/2,sum_pts)
#text
text_ans = ['(0,0)','(pi,0)','(pi,pi)','(0,pi)']
