import numpy as np

#fixed
z1 = 4
z2 = 4
J1 = 1
J2 = 0
m = [3,3,6,6]
maxK1 = [np.pi,np.pi,np.pi,np.pi] ## sure?
maxK2 = [2*np.pi/np.sqrt(3),2*np.pi/np.sqrt(3),np.pi/np.sqrt(3),np.pi/np.sqrt(3)] ## sure?
text_ans = ['(0,0)','(pi,0)','(pi,pi)','(0,pi)']
#variable
S = 0.025
sum_pts = 300
phi_pts = 20
phi_max = 0.52#*np.pi

