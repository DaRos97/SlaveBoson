import numpy as np

#fixed
S = 0.5
z1 = 4
z2 = 4
z3 = 2
J1 = 1
m = [3,3,6,6]
maxK1 = [np.pi,np.pi,np.pi,np.pi]
maxK2 = [2*np.pi/np.sqrt(3),2*np.pi/np.sqrt(3),np.pi/np.sqrt(3),np.pi/np.sqrt(3)]
text_ans = ['(0,0)','(pi,0)','(pi,pi)','(0,pi)']
#values of DM phases fixed by symmetries
phi1 = 0#2*np.pi/3
phi2 = 0#np.pi    #sure?
phi3 = 0#2*np.pi/3        ## what is it?????
#variable
J2_max = 0.25
J2_pts = 11
J3_max = 0.5
J3_pts = 3

sum_pts = 100

alpha_min = -0.1
alpha_max = np.pi/2+0.1
alpha_pts = 50
