import numpy as np
#
m = 6
S = 0.5
#S = (np.sqrt(3)-1)/2#0.5
####
phi = 0.00
DM1 = phi
DM2 = 0
DM3 = phi*2
####
Nx = 13
Ny = 13
mp_cpu = 32
list_ans = ['cb1']
DirName = '/home/users/r/rossid/Data/phi'+"{:3.2f}".format(phi).replace('.','')+"/"
#DirName = '../Data/test/'
DataDir = DirName + '13/'
ReferenceDir = 'none'#DirName + 'fullDM_13/'
#derivative
s_b = 0.01 #bound on values given by smaller grids
der_par = 1e-6
der_phi = 1e-5
der_lim = 1  #limit under which compute the Hessian for that parameter
cutoff = 1e-8   ############      #accettable value of Sigma to accept result as converged
MaxIter = 100
prec_L = 1e-10       #precision required in L maximization
cutoff_pts = 1e-12      #min difference b/w phase diagram points to be considered the same
L_method = 'Brent'
L_bounds = (0.4,0.75)
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
header = {'cb1':    ['ans','J2','J3','Converge','Energy','Sigma','gap','L','A1','A2','A3','B1','B2','phiA1','phiB1','phiA2','phiB2']}  #cuboc1
t_0 = np.arctan(np.sqrt(2))
Pi = {'cb1':{'A1':0.51, 'A2':0.1, 'A3':0.43, 'B1':0.17, 'B2': 0.17, 'phiA1': 2, 'phiB1': np.pi, 'phiA2': 4.1, 'phiB2': 5.1}
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
    if ans == 'cb1':
        bounds[ans]['A1'] = (0.4,0.6)
        bounds[ans]['A2'] = (0.01,0.31)
        bounds[ans]['A3'] = (0.03,0.55)
        bounds[ans]['B1'] = (0.08,0.27)
        bounds[ans]['B2'] = (0.1,0.35)
        bounds[ans]['phiA1'] = (1.4,2.35)
        bounds[ans]['phiB1'] = (3.12,3.2)
        bounds[ans]['phiA2'] = (3.7,4.3)
        bounds[ans]['phiB2'] = (4.7,5.3)
        num_phi[ans] = 4
shame2 = 5

print("Minimization precision (both tol and atol):",cutoff)
print("Grid pts:",Nx,'*',Ny)
print("Derivative distance (par / phi):",der_par,'/',der_phi)
print("Spin: ",S)
print("Bound on values given by smaller grids: ",s_b)
print("Number of CPUs used: ",mp_cpu)
