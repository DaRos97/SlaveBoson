import numpy as np

#fixed
z1 = 4
z2 = 4
J1 = 1
J2 = J1/10
m = [3,3,6,6]
maxK1 = [2*np.pi,2*np.pi,2*np.pi,2*np.pi] ## sure?
maxK2 = [4*np.pi/np.sqrt(3),4*np.pi/np.sqrt(3),2*np.pi/np.sqrt(3),2*np.pi/np.sqrt(3)] ## sure?
text_ans = ['(0,0)','(pi,0)','(pi,pi)','(0,pi)']
#variable
S = 0.2
#ansatz = 3  ### 0 == (0,0), 1 == (pi,0), 2 == (pi,pi), 3 == (0,pi) ansatz
sum_pts = 51
phi_pts = 11
phi_max = np.pi/2

#text_data = 'data2/' + text_ans[ansatz] + 'E_gs-' + str(S).replace('.',',') + '.npy'

