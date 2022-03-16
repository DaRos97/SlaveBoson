import numpy as np

S = 0.5
#derivative
der_pts = 5
der_range = [0.0001,0.0001,0.0001,0.0001,1e-4]
step = 0.15
sum_pts = 101
grid_pts = 7
#fixed
J1 = 1
z1 = 4
z2 = 4
z3 = 2
z = (z1,z2,z3)
#minimization
Bnds = [((0,1),(-1,1),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5)),  #3x3 -> A1,A3,B1,B2,B3
        ((0,1),(-1,1),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5)),  #q0 -> A1,A2,B1,B2,B3
        ((0,1),(-1,1),(-1,1),(-0.5,0.5),(-0.5,0.5)),  #(0,pi) -> A1,A2,A3,B1,B2
        ((0,1),(-0.5,0.5),(-0.5,0.5)),  #(pi,pi) -> A1,B1,B2
        ((0,1),(0,1),(0,0.05),(0,0.05),(0,0.05))]  #cuboc1 -> A1,...????
cutoff = 1e-6       #accettable value of Sigma to exit minimization
repetitions = 0
complex_cutoff = 1e-4       #max value of complex terms in diagonalization
cutoff_pts = 1e-12      #min difference b/w phase diagram points to be considered the same
#phase diagram
Ji = -0.3
Jf = 0.3+step
rJ2 = np.arange(Ji,Jf,step)
rJ3 = np.arange(Ji,Jf,step)
#summation over BZ
maxK1 = np.pi
maxK26 = np.pi/np.sqrt(3)
K1 = np.linspace(0,maxK1,sum_pts)  #Kx in BZ
K26 = np.linspace(0,maxK26,sum_pts)  #Ky in BZ
Kp = (K1,K26)
kg = (np.linspace(0,maxK1,grid_pts),np.linspace(0,maxK26,grid_pts))
#text
text_ans = ['3x3','q0','(0,pi)','(pi,pi)','cuboc1']
dirname = 'Data/'
#csv
header = [['J2','J3','Energy','Sigma','L','A1','A3','B1','B2','B3'],  #3x3
          ['J2','J3','Energy','Sigma','L','A1','A2','B1','B2','B3'],  #q0
          ['J2','J3','Energy','Sigma','L','A1','A2','A3','B1','B2'],  #(0,pi)
          ['J2','J3','Energy','Sigma','L','A1','B1','B2'],  #(pi,pi)
          ['J2','J3','Energy','Sigma','L','A1','A2','A3','B1','B2','B3','phiA1']]  #cuboc1
csvfile = [dirname+'S'+str(S).replace('.','')+'-'+text_ans[ans]+'.csv' for ans in range(4)]

