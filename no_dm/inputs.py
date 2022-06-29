import numpy as np
from colorama import Fore
#
m = 6
S = 0.5
#S = (np.sqrt(3)-1)/2#0.5
####
DM1 = 0
DM3 = 0
####
Nx = 13
Ny = 13
mp_cpu = 1#6
list_ans = ['cb2']#'3x3_1','q0_1','cb1','cb2','oct']#,'3x3_2','3x3_3','3x3_4','q0_1','q0_2','q0_3','q0_4','cb1','cb2','oct']
#DirName = '/home/users/r/rossid/Data/'
DirName = '../Data/test/'
DataDir = DirName# + 'no_DM_13/'
ReferenceDir = 'none'#DirName + 'Data_13-13/'
#derivative
s_b = 0.01
der_par = 1e-6
der_phi = 1e-5
der_lim = 1  #limit under which compute the Hessian for that parameter
cutoff = 1e-8   ############      #accettable value of Sigma to accept result as converged
MaxIter = 200
prec_L = 1e-10       #precision required in L maximization
cutoff_pts = 1e-12      #min difference b/w phase diagram points to be considered the same
L_method = 'Brent'
L_bounds = (0.3,3)
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
header = {'3x3_1':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A3','B1','B2','B3','phiA3'],  #3x3
          '3x3_2':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A3','B1','B2','B3','phiB1','phiB2'],  #3x3
          '3x3_3':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A3','B1','B2','B3','phiB1','phiB3'],  #3x3
          '3x3_4':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A3','B1','B2','B3','phiA1','phiB2'],  #3x3
          'q0_1':     ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','B1','B2','B3','phiA2'],  #q0
          'q0_2':     ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','B1','B2','B3','phiB1','phiB2'],  #q0
          'q0_3':     ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','B1','B2','B3','phiB1','phiA2','phiB3'],  #q0
          'q0_4':     ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','B1','B2','B3','phiA1','phiB2'],  #q0
          'cb1':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','A3','B1','B2','phiA1','phiB2'],  #cuboc1
          'cb2':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','A3','B1','B2','phiB1','phiA2'],  #cuboc2
          'oct':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','B1','B2','B3','phiB1','phiB2']}  #octahedral
t_0 = np.arctan(np.sqrt(2))
Pi = {  '3x3_1':{'A1':0.51, 'A3':0.433, 'B1':0.17, 'B2': 0.35, 'B3': 0.25, 'phiA3': np.pi},
        '3x3_2':{'A1':0.51, 'A3':0.433, 'B1':0.17, 'B2': 0.35, 'B3': 0.25, 'phiB1': np.pi, 'phiB2': 0},
        '3x3_3':{'A1':0.51, 'A3':0.433, 'B1':0.17, 'B2': 0.35, 'B3': 0.25, 'phiB1': np.pi, 'phiB3': np.pi},
        '3x3_4':{'A1':0.51, 'A3':0.433, 'B1':0.17, 'B2': 0.35, 'B3': 0.25, 'phiA1': np.pi, 'phiB2': 0},
        'q0_1':{'A1':0.51, 'A2':0.433, 'B1':0.18, 'B2': 0.25, 'B3': 0.35, 'phiA2': np.pi},
        'q0_2':{'A1':0.51, 'A2':0.433, 'B1':0.18, 'B2': 0.25, 'B3': 0.35, 'phiB1': np.pi, 'phiB2': np.pi},
        'q0_3':{'A1':0.51, 'A2':0.433, 'B1':0.18, 'B2': 0.25, 'B3': 0.35, 'phiB1': np.pi, 'phiA2': np.pi, 'phiB3': 0},
        'q0_4':{'A1':0.51, 'A2':0.433, 'B1':0.18, 'B2': 0.25, 'B3': 0.35, 'phiA1': 0, 'phiB2': np.pi},
        'cb1':{'A1':0.51, 'A2':0.25, 'A3':0.3, 'B1':0.17, 'B2': 0.3, 'phiA1':np.pi, 'phiB2':np.pi},
        'cb2':{'A1':0.51, 'A2':0.433, 'A3':0.3, 'B1':0.3, 'B2': 0.25, 'phiB1':np.pi, 'phiA2':np.pi},
        'oct':{'A1':0.51, 'A2':0.35, 'B1':0.35, 'B2': 0.35, 'B3':0.35 ,'phiB1':np.pi, 'phiB2':np.pi}
        }
lAns = header.keys()
bounds = {}
num_phi = {}
list_A2 = []
list_A3 = []
list_B3 = []
for ans in lAns:
    bounds[ans] = {}
    num_phi[ans] = 0
    lPar = Pi[ans].keys()
    if 'A2' in lPar:
        list_A2.append(ans)
    if 'A3' in lPar:
        list_A3.append(ans)
    if 'B3' in lPar:
        list_B3.append(ans)
    #bounds
    if ans == '3x3_1':
        bounds[ans]['A1'] = (0.4,0.6)
        bounds[ans]['A3'] = (0.001,0.45)
        bounds[ans]['B1'] = (0.001,0.45)
        bounds[ans]['B2'] = (0.001,0.45)
        bounds[ans]['B3'] = (0,0.5)
        bounds[ans]['phiA3'] = (np.pi-0.2,np.pi+0.2)
        num_phi[ans] = 1
    elif ans == 'q0_1':
        bounds[ans]['A1'] = (0.4,0.6)
        bounds[ans]['A2'] = (0.001,0.45)
        bounds[ans]['B1'] = (0.001,0.45)
        bounds[ans]['B2'] = (0.001,0.45)
        bounds[ans]['B3'] = (0,0.5)
        bounds[ans]['phiA2'] = (np.pi-0.2,np.pi+0.2)
        num_phi[ans] = 1
    elif ans == 'cb1':
        bounds[ans]['A1'] = (0.4,0.6)
        bounds[ans]['A2'] = (0.001,0.45)
        bounds[ans]['A3'] = (0.001,0.45)
        bounds[ans]['B1'] = (0.001,0.45)
        bounds[ans]['B2'] = (0.001,0.45)
        bounds[ans]['phiA1'] = (np.pi-0.2,np.pi+0.2)
        bounds[ans]['phiB2'] = (np.pi-0.2,np.pi+0.2)
        num_phi[ans] = 2
    elif ans == 'cb2':
        bounds[ans]['A1'] = (0.4,0.6)
        bounds[ans]['A2'] = (0.001,0.45)
        bounds[ans]['A3'] = (0.001,0.45)
        bounds[ans]['B1'] = (0.001,0.45)
        bounds[ans]['B2'] = (0.001,0.45)
        bounds[ans]['phiB1'] = (np.pi-0.2,np.pi+0.2)
        bounds[ans]['phiA2'] = (np.pi-0.2,np.pi+0.2)
        num_phi[ans] = 2
    elif ans == 'oct':
        bounds[ans]['A1'] = (0.4,0.6)
        bounds[ans]['A2'] = (0.001,0.45)
        bounds[ans]['B1'] = (0.001,0.45)
        bounds[ans]['B2'] = (0.001,0.45)
        bounds[ans]['B3'] = (0,0.5)
        bounds[ans]['phiB1'] = (np.pi-0.2,np.pi+0.2)
        bounds[ans]['phiB2'] = (np.pi-0.2,np.pi+0.2)
        num_phi[ans] = 2
shame2 = 5

print("Minimization precision (both tol and atol):",cutoff)
print("Grid pts:",Nx,'*',Ny)
print("Derivative distance (par / phi):",der_par,'/',der_phi)
print("Spin: ",S)
print("Bound on values given by smaller grids: ",s_b)
print("Number of CPUs used: ",mp_cpu)
