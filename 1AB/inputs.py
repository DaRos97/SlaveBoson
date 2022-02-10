import numpy as np

#fixed
z1 = 4
J1 = 1
m = [3,3,6,6]
maxK1 = [np.pi,np.pi,np.pi,np.pi]
maxK2 = [2*np.pi/np.sqrt(3),2*np.pi/np.sqrt(3),np.pi/np.sqrt(3),np.pi/np.sqrt(3)]
text_ans = ['(0,0)','(pi,0)','(pi,pi)','(0,pi)']

#variable
S = 0.2
cutoff = 1e-6
sum_pts = 51
derPts = 5

K1 = np.linspace(0,maxK1[0],sum_pts)  #Kx in BZ
K2 = np.linspace(0,maxK2[0],sum_pts)  #Ky in BZ
