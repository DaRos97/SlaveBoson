import numpy as np

#fixed
S = 0.5
z1 = 4
z2 = 4
J1 = 1
m = [3,3,6,6]
maxK1 = [np.pi,np.pi,np.pi,np.pi]
maxK2 = [2*np.pi/np.sqrt(3),2*np.pi/np.sqrt(3),np.pi/np.sqrt(3),np.pi/np.sqrt(3)]
text_ans = ['(0,0)','(pi,0)','(pi,pi)','(0,pi)']

phi1 = np.pi/3*2
phi2 = np.pi
#variable
J2 = 0.2
sum_pts = 200

phi_pts = 20
phi_max = np.pi/3

