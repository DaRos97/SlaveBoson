import numpy as np
from colorama import Fore
#
m = 6
S = 0.5
####
DM1 = 0#4/3*np.pi
DM3 = 0#2/3*np.pi
####
Nx = 13
Ny = 13
mp_cpu = 1
list_ans = ['cb1']#,'q0','cb1']#,'q0']#,'0-pi','cb2']#,'octa']
DirName = '/home/users/r/rossid/Data/noDMbig/'
#DirName = '../Data/test/'
DataDir = DirName + 'Data_'+str(Nx)+'-'+str(Ny)+'/'
ReferenceDir = DirName + 'Data_12-12/'
#derivative
der_par = 1e-6
der_phi = 1e-5
der_lim = 1e-2  #limit under which compute the Hessian for that parameter
cutoff = 1e-10   ############      #accettable value of Sigma to accept result as converged
MaxIter = 200
prec_L = 1e-10       #precision required in L maximization
cutoff_pts = 1e-12      #min difference b/w phase diagram points to be considered the same
L_method = 'bounded'
#phase diagram
J1 = 1
z = (4,4,2)
#small
#J2i = -0.02; J2f = 0.03; J3i = -0.04; J3f = 0.01; Jpts = 11
#big
J2i = -0.3; J2f = 0.3; J3i = -0.3; J3f = 0.3; Jpts = 9
J= []
for i in range(Jpts):
    for j in range(Jpts):
        J.append((J2i+(J2f-J2i)/(Jpts-1)*i,J3i+(J3f-J3i)/(Jpts-1)*j))
#summation over BZ
kxg = np.linspace(0,1,Nx)
kyg = np.linspace(0,1,Ny)
kkg = np.ndarray((2,Nx,Ny),dtype=complex)
kkgp = np.ndarray((2,Nx,Ny))
for i in range(Nx):
    for j in range(Ny):
        kkg[0,i,j] = kxg[i]*2*np.pi
        kkg[1,i,j] = (kxg[i]+kyg[j])*2*np.pi/np.sqrt(3)
        kkgp[0,i,j] = kxg[i]*2*np.pi
        kkgp[1,i,j] = (kxg[i]+kyg[j])*2*np.pi/np.sqrt(3)
#initial point
Pi = {  '3x3':{'A1':0.51706, 'A3':0.1, 'B1':0.17790, 'B2': 0.36, 'B3': 0.1},
        'q0':{'A1':0.51624, 'A2':0.1, 'B1':0.18036, 'B2': 0.17, 'B3': 0.13},
        'cb1':{'A1':0.51660, 'A2':0.1, 'A3':0.1, 'B1':0.17616, 'B2': 0.1, 'phiA1':1.9525},
        '0-pi':{'A1':0.5, 'A2':0.0, 'A3':0.0, 'B1':0.2, 'B2': 0.0},
        'cb2':{'A1':0.5, 'A2':0.0, 'A3':0.0, 'B1':0.0, 'B2': 0.0, 'phiB1':np.pi}
        }
#bounds
bounds = {  '3x3':{ 'A1':(0.4,0.6),
                    'A3':(0,1),
                    'B1':(0,0.5),
                    'B2':(0,0.5),
                    'B3':(0,0.5)},
            'q0': { 'A1':(0.4,0.6),
                    'A2':(0,0.6),
                    'B1':(0,0.5),
                    'B2':(0,0.5),
                    'B3':(0,0.5)},
            'cb1':{ 'A1':(0.5,0.7),
                    'A2':(0,1),
                    'A3':(0,1),
                    'B1':(0,0.5),
                    'B2':(0,0.5),
                    'phiA1':(0,2*np.pi)}
         }
L_bounds = (0.1,10)
shame1 = -1
shame2 = 5
shame3 = 2
#csv
header = {'3x3':    ['ans','J2','J3','Energy','Sigma','gap','L','A1','A3','B1','B2','B3'],  #3x3
          'q0':     ['ans','J2','J3','Energy','Sigma','gap','L','A1','A2','B1','B2','B3'],  #q0
          'cb1':    ['ans','J2','J3','Energy','Sigma','gap','L','A1','A2','A3','B1','B2','phiA1'],  #cuboc1
          '0-pi':   ['ans','J2','J3','Energy','Sigma','gap','L','A1','A2','A3','B1','B2'],  #0-pi
          'cb2':    ['ans','J2','J3','Energy','Sigma','gap','L','A1','A2','A3','B1','B2','phiB1'],  #cuboc2
          'octa':   ['ans','J2','J3','Energy','Sigma','gap','L','A1','A2','B1','B2','B3','phiB1']}  #octa
list_A2 = ['q0','0-pi','octa','cb1','cb2']
list_A3 = ['3x3','0-pi','cb1','cb2']
list_B3 = ['3x3','q0','octa']
list_chiral = ['cb1','cb2','octa']
#Hess signs
HS = {'3x3':[[[1,-1,-1,1,1], [1,-1,1], [1,1,-1,1,-1]  ],
             [[1,-1,-1,1],   [1,-1],   [1,1,-1,-1]     ],
             [[1,-1,-1,-1,1],[1,-1,-1],[1,1,-1,-1,-1]]],
      'q0' :[[[1,-1,-1,1,1],[1,-1,-1,1],[1,-1,-1,1,-1] ],
             [[1,-1,1],     [1,-1],     [1,-1,-1]      ],
             [[1,1,-1,-1,1],[1,1,-1,-1],[1,1,-1,-1,-1]]],
      'cb1':[[[1,-1,-1,-1,1,1],[1,-1,-1,1,1],[1,-1,1,-1,1,1] ],
             [[1,-1,-1,1],     [1,-1,1],     [1,1,-1,1]      ],
             [[1,1,-1,-1,-1,1],[1,1,-1,-1,1],[1,1,1,-1,-1,1]]]
      }

min_S = 10

print("Minimization precision (both tol and atol):",cutoff)
print("Grid pts:",Nx,'*',Ny)
print("Derivative distance (par / phi):",der_par,'/',der_phi)
print("Lagrange multiplier maximization precision:",prec_L)
print("Dzyaloshinskii-Moriya angles:",DM1,"  ",DM3)
print("Number of CPUs used: ",mp_cpu)

