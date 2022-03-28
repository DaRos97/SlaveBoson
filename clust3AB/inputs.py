import numpy as np

text_ans = ['3x3','q0','cb1']
dirname = '/home/users/r/rossid/git/clust3AB/Data/'    ###########
#dirname = 'TempData/'                                       ###########
####
S = 0.5
#derivative
der_pts = 2
der_range = [1e-8 for i in range(8)]
Jpts = 11
sum_pts = 101
grid_pts = 9    ############
cutoff = 1e-8   ############      #accettable value of Sigma to accept result as converged
#fixed
J1 = 1
z = (4,4,2)
#minimization
initialPoint = {'3x3':  (0.52,0.26,0.18,0.31,0.12),
                'q0':  (0.51,0.2,0.18,0.19,0.043),
                'cb1':  (0.51,0.1,0.01,0.06,1.95)}
Bnds = {'3x3':  ((0,1),(0,1),(0.,0.5),(0.,0.5),(0.,0.5)),    #3x3 -> A1,A3,B1,B2,B3
        'q0':   ((0,1),(0,1),(0.,0.5),(0.,0.5),(0.,0.5)),    #q0 -> A1,A2,B1,B2,B3
        '0-pi': ((0,1),(-1,1),(-1,1),(-0.5,0.5),(-0.5,0.5)),        #0-pi -> A1,A2,A3,B1,B2
        'pi-pi':((0,1),(-0.5,0.5),(-0.5,0.5)),                      #pi-pi -> A1,B1,B2
        'oct':  ((0,1),(-0.5,0.5),(-0.5,0.5)),                      #octa ->
        'cb1':  ((0,1),(-0.5,0.5),(-0.5,0.5),(-0.5,0.5),(-np.pi,np.pi)),  #cb1 -> A1,A2,A3,B1,B2,B3,phiA,phiB
        'cb2':  ((0,1),(-1,1),(-1,1),(-0.5,0.5),(-0.5,0.5),(-np.pi,np.pi),(-np.pi,np.pi))}  #cb2 -> A1,A2,A3,B1,B2,phiB1,phiA2(?)
prec_L = 1e-8       #precision required in L maximization
complex_cutoff = 1e-8       #max value of complex terms in diagonalization
cutoff_pts = 1e-12      #min difference b/w phase diagram points to be considered the same
#phase diagram
Ji = -0.03
Jf = 0.03
J = []
for i in range(Jpts):
    for j in range(Jpts):
        J.append((Ji+(Jf-Ji)/(Jpts-1)*i,Ji+(Jf-Ji)/(Jpts-1)*j))
#summation over BZ
maxK1 = 2*np.pi
maxK2 = 2*np.pi/np.sqrt(3)
K1 = np.linspace(0,maxK1,sum_pts)  #Kx in BZ
K2 = np.linspace(0,maxK2,sum_pts)  #Ky in BZ
Kp = (K1,K2)
kg = (np.linspace(0,maxK1,grid_pts),np.linspace(0,maxK2,grid_pts))
#matrices
Mkg = np.zeros((2,grid_pts,grid_pts),dtype=complex)
for i in range(grid_pts):
    Mkg[0,i,:] = kg[0]
    Mkg[1,:,i] = kg[1]

#text
#csv
header = {'3x3':    ['ans','J2','J3','Energy','Sigma','L','A1','A3','B1','B2','B3'],  #3x3
          'q0':     ['ans','J2','J3','Energy','Sigma','L','A1','A2','B1','B2','B3'],  #q0
          'cb1':    ['ans','J2','J3','Energy','Sigma','L','A1','B1','B2','B3','phiA1']}  #cuboc1
csvfile = [dirname+text_ans[ans]+'.csv' for ans in range(len(text_ans))]


print("min prec:",cutoff)
print("grid pts:",grid_pts)
print("Complex cutoff:",complex_cutoff)
print("Der pts and distance:",der_pts,der_range[0])
print("L precision:",prec_L)
