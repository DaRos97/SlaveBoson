import numpy as np
import functions as fs

#import parameters from file
ans = '3x3_1'
J1, J2, J3 = (1,0,0)
S = 0.5
DM = False
pts = '13'

txt_S = '05' if S == 0.5 else '03'
txt_DM = 'DM' if DM else 'no_DM'
filename = '../Data/'+pts+'/'+txt_S+txt_DM+'/'+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
params = fs.import_data(ans,filename)
args = (J1,J2,J3,ans,DM)
#compute the Ks of the minimum band
Nx = 97     #points for looking at minima in BZ
Ny = 97
K_,is_LRO = fs.find_minima(params,args,Nx,Ny)
if not is_LRO:
    print("Not LRO, there is a gap")
    exit()
print("Found Ks: ",K_)
#Compute the M in those K and extract the relative columns
V = fs.get_V(K_,params,args)
#construct the spin matrix for each sublattice and compute the coordinates
#of the spin at each lattice site
UC = 10
S = np.zeros((3,6,UC,UC)) #3 spin components, 6 sites in UC, ij coordinates of UC
sigma = np.zeros((3,2,2),dtype = complex)
sigma[0] = np.array([[0,1],[1,0]])
sigma[1] = np.array([[0,-1j],[1j,0]])
sigma[2] = np.array([[1,0],[0,-1]])
a1 = np.array([1,0])
a2 = np.array([-1,np.sqrt(3)])
k1 = K_[0]
v1 = V[0]
m = 6
c1 = (1+1j)/np.sqrt(2); c1_ = np.conjugate(c1)      #what is this??
c2 = (1-1j)/np.sqrt(2); c2_ = np.conjugate(c2)      #what is this??
for i in range(UC):
    for j in range(UC):
        r = i*a1 + j*a2
        for s in range(6):
            cond = np.zeros(2,dtype=complex)
            cond[0] = c1*v1[s]*np.exp(1j*np.dot(k1,r)) + c2*np.conjugate(v1[s])*np.exp(-1j*np.dot(k1,r))
            cond[1] = c2_*v1[m+s]*np.exp(1j*np.dot(k1,r)) + c1_*np.conjugate(v1[m+s])*np.exp(-1j*np.dot(k1,r))
            for x in range(3):
                S[x,s,i,j] = np.real(1/2*np.dot(np.conjugate(cond.T),np.einsum('ij,j->i',sigma[x],cond)))
savenameS = "SpinOrientations/S_"+ans+'_'+txt_DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
np.save(savenameS,S)
#
#Now compute the SSF
UC = 6
Kx = 17     #point to compute the SSF in the EBZ
Ky = 17
kxg = np.linspace(-8*np.pi/3,8*np.pi/3,Kx)
kyg = np.linspace(-4*np.pi/np.sqrt(3),4*np.pi/np.sqrt(3),Ky)
K = np.zeros((2,Kx,Ky))
SFzz = np.zeros((Kx,Ky))
SFxy = np.zeros((Kx,Ky))
for i in range(Kx):
    for j in range(Ky):
        K[:,i,j] = np.array([kxg[i],kyg[j]])
        SFxy[i,j], SFzz[i,j] = fs.SpinStructureFactor(K[:,i,j],S,UC)

savenameSSFzz = "SSF/SSFzz_"+ans+'_'+txt_DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
savenameSSFxy = "SSF/SSFxy_"+ans+'_'+txt_DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
np.save(savenameSSFzz,SFzz)
np.save(savenameSSFxy,SFxy)
