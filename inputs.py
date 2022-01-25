import numpy as np

#fixed
z1 = 4
z2 = 4
J1 = 1
J2 = 0
m = [3,3,6,6]
maxK1 = [2*np.pi,2*np.pi,2*np.pi,2*np.pi] ## sure?
maxK2 = [4*np.pi/np.sqrt(3),4*np.pi/np.sqrt(3),2*np.pi/np.sqrt(3),2*np.pi/np.sqrt(3)] ## sure?
text_ans = ['(0,0)','(pi,0)','(pi,pi)','(0,pi)']
#variable
S = 0.5
sum_pts = 101
phi_pts = 10
phi_max = 0.5*np.pi

