import numpy as np

S = 0.5
#fixed
J1 = 1
z1 = 4
z2 = 4
z3 = 2
#minimization
cutoff = 1e-4
#derivative
der_pts = 2
der_range = [0.001,0.001,0.001,0.001]       #derivative range for A1 and L
#phase diagram
PD_pts = 5
iJ2 = 0
fJ2 = 0.3
iJ3 = 0
fJ3 = 0.3
rJ2 = np.linspace(iJ2,fJ2,PD_pts)
rJ3 = np.linspace(iJ3,fJ3,PD_pts)
#summation over BZ
sum_pts = 51
maxK1 = np.pi
maxK2 = 2*np.pi/np.sqrt(3)
K1 = np.linspace(0,maxK1,sum_pts)  #Kx in BZ
K2 = np.linspace(0,maxK2,sum_pts)  #Ky in BZ
#text
text_ans = ['(0,0)','(pi,0)','(pi,pi)','(0,pi)']
