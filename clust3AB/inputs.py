import numpy as np

m = 6
S = 0.5
####
DM1 = np.pi/3
DM3 = 0.#DM1/2#np.pi/3
tDM = 'DM' if (DM1 != 0 or DM3 != 0) else ''
####
grid_pts = 5
text_ans = ['3x3','q0','0-pi','cb1','cb2']#,'octa']
#dirname = '/home/users/r/rossid/git/Data/'
dirname = '../Data/test/'
dataDir = 'Data_'+str(grid_pts)+tDM+'/'
refDirname = dirname+dataDir#dirname+'Data_7/'          ########ATTENZIONE
#derivative
method = 'Powell'
der_range = [1e-8 for i in range(8)]
Jpts = 21
sum_pts = 101
cutoff = 1e-10   ############      #accettable value of Sigma to accept result as converged
prec_L = 1e-10       #precision required in L maximization
cutoff_pts = 1e-12      #min difference b/w phase diagram points to be considered the same
#phase diagram
J1 = 1
z = (4,4,2)
Ji = -0.3
Jf = 0.3
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
#csv
header = {'3x3':    ['ans','J2','J3','Energy','Sigma','L','A1','A3','B1','B2','B3'],  #3x3
          'q0':     ['ans','J2','J3','Energy','Sigma','L','A1','A2','B1','B2','B3'],  #q0
          '0-pi':   ['ans','J2','J3','Energy','Sigma','L','A1','A2','A3','B1','B2'],  #0-pi
          'cb1':    ['ans','J2','J3','Energy','Sigma','L','A1','A2','A3','B1','B2','phiA1'],  #cuboc1
          'cb2':    ['ans','J2','J3','Energy','Sigma','L','A1','A2','A3','B1','B2','phiB1'],  #cuboc2
          'octa':   ['ans','J2','J3','Energy','Sigma','L','A1','A2','B1','B2','B3','phiB1']}  #octa
list_A2 = ['q0','0-pi','octa','cb1','cb2']
list_A3 = ['3x3','0-pi','cb1','cb2']
list_B3 = ['3x3','q0','octa']

print("Method used: ",method)
print("Minimization precision:",cutoff)
print("Grid / Summation pts:",grid_pts,'/',sum_pts)
print("Derivative distance:",der_range[0])
print("L minimization precision:",prec_L)
