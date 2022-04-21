import numpy as np
from colorama import Fore
#
m = 6
S = 0.5
####
DM1 = 0#4/3*np.pi
DM3 = 0#2/3*np.pi
tDM = '_DM' if (DM1 != 0 or DM3 != 0) else ''
####
grid_pts = 11
mp_cpu = -1
list_ans = ['cb1','3x3','q0']#,'0-pi','cb2']#,'octa']
DirName = '/home/users/r/rossid/Test/noDM/'
#DirName = '../Data/test/'
DataDir = DirName + 'Data_'+str(grid_pts)+tDM+'/'
ReferenceDir = 'none'#DirName + 'Data_11/'
#derivative
der_par = 1e-6
der_phi = 1e-4
Jpts = 11
sum_pts = 101
cutoff = 1e-10   ############      #accettable value of Sigma to accept result as converged
MaxIter = 500
prec_L = 1e-10       #precision required in L maximization
cutoff_pts = 1e-12      #min difference b/w phase diagram points to be considered the same
L_method = 'bounded'
#phase diagram
J1 = 1
z = (4,4,2)
J2i = -0.02; J2f = 0.03
J3i = -0.04; J3f = 0.01
J = []
for i in range(Jpts):
    for j in range(Jpts):
        J.append((J2i+(J2f-J2i)/(Jpts-1)*i,J3i+(J3f-J3i)/(Jpts-1)*j))
#summation over BZ
maxK1 = 2*np.pi
maxK2 = 2*np.pi/np.sqrt(3)
K1 = np.linspace(0,maxK1,sum_pts)  #Kx in BZ
K2 = np.linspace(0,maxK2,sum_pts)  #Ky in BZ
Kp = (K1,K2)
kg = (np.linspace(0,maxK1,grid_pts),np.linspace(0,maxK2,grid_pts))
Mkg = np.zeros((2,grid_pts,grid_pts),dtype=complex)
for i in range(grid_pts):
    Mkg[0,i,:] = kg[0]
    Mkg[1,:,i] = kg[1]
#initial point
Pi = {  '3x3':{'A1':0.51, 'A3':0.1, 'B1':0.17, 'B2': 0.36, 'B3': 0.0},
        'q0':{'A1':0.51, 'A2':0.4, 'B1':0.18, 'B2': 0.2, 'B3': 0.2},
        'cb1':{'A1':0.51, 'A2':0.4, 'A3':0.3, 'B1':0.17, 'B2': 0.2, 'phiA1':1.95},
        'cb12':{'A1':0.51, 'A2':0.4, 'A3':0.3, 'B1':0.17, 'B2': 0.2, 'phiA1':1.95, 'phiB2': 3.14},
        '0-pi':{'A1':0.5, 'A2':0.0, 'A3':0.0, 'B1':0.2, 'B2': 0.0},
        'cb2':{'A1':0.5, 'A2':0.0, 'A3':0.0, 'B1':0.0, 'B2': 0.0, 'phiB1':np.pi}
        }
#bounds
bounds = {  'A1':(-1,1),
            'A2':(-1,1),
            'A3':(-1,1),
            'B1':(-0.5,0.5),
            'B2':(-0.5,0.5),
            'B3':(-0.5,0.5)}
L_bounds = (0.1,5)
shame_value = -1
shame2 = 5
#csv
header = {'3x3':    ['ans','J2','J3','Energy','Sigma','L','A1','A3','B1','B2','B3'],  #3x3
          'q0':     ['ans','J2','J3','Energy','Sigma','L','A1','A2','B1','B2','B3'],  #q0
          'cb1':    ['ans','J2','J3','Energy','Sigma','L','A1','A2','A3','B1','B2','phiA1'],  #cuboc1
          'cb12':    ['ans','J2','J3','Energy','Sigma','L','A1','A2','A3','B1','B2','phiA1','phiB2'],  #cuboc1
          '0-pi':   ['ans','J2','J3','Energy','Sigma','L','A1','A2','A3','B1','B2'],  #0-pi
          'cb2':    ['ans','J2','J3','Energy','Sigma','L','A1','A2','A3','B1','B2','phiB1'],  #cuboc2
          'octa':   ['ans','J2','J3','Energy','Sigma','L','A1','A2','B1','B2','B3','phiB1']}  #octa
list_A2 = ['q0','0-pi','octa','cb1','cb2','cb12']
list_A3 = ['3x3','0-pi','cb1','cb2','cb12']
list_B3 = ['3x3','q0','octa']
list_chiral = ['cb1','cb2','cb12','octa']

print("Minimization precision (both tol and atol):",cutoff)
print("Grid / Summation pts:",grid_pts,'/',sum_pts)
print("Derivative distance (par / phi):",der_par,'/',der_phi)
print("Lagrange multiplier maximization precision:",prec_L)
print("Dzyaloshinskii-Moriya angles:",DM1,"  ",DM3)

####################################
DM_PD = []
Si = 0.0
Sf = 0.5
DM1i = 0
DM1f = 1
PDpts = 11
for i in range(Jpts):
    for j in range(Jpts):
        DM_PD.append((Si+(Sf-Si)/(PDpts-1)*i,DM1i+(DM1f-DM1i)/(PDpts-1)*j))
