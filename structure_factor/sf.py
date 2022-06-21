import numpy as np
import functions as fs
from time import time as T

#structure factor of ansatz ans at (J2,J3) from data in filename
ans = '3x3_1'
J1, J2, J3 = (1,0.225,0)
S = 0.5
DM = False

txt_S = '05' if S == 0.5 else '03'
txt_DM = 'DM' if DM else 'no_DM'
filename = '../Data/S'+txt_S+'/'+txt_DM+'_13/'+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
savename = "SF_"+ans+'_'+txt_DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
Kx = 13     #points to compute in the SF BZ
Ky = 13
kxg = np.linspace(0,1,Kx)
kyg = np.linspace(0,1,Ky)
##
Nx = 19     #points for summation over BZ
Ny = 19
nxg = np.linspace(0,1,Nx)
nyg = np.linspace(0,1,Ny)

params = fs.import_data(ans,filename)

args = (J1,J2,J3,ans,DM)

SF = np.zeros((Kx,Ky))
for i in range(Kx):
    for j in range(Ky):
        K = np.array([kxg[i]*2*np.pi,(kxg[i]+kyg[j])*2*np.pi/np.sqrt(3)])
        res = 0
        Ti = T()
        for ii in range(Nx):
            for ij in range(Ny):
                Q = np.array([nxg[ii]*2*np.pi,(nxg[ii]+nyg[ij])*2*np.pi/np.sqrt(3)])
                U1,X1,V1,Y1 = fs.M(Q,params,args)
                U2,X2,V2,Y2 = fs.M(-Q,params,args)
                U3,X3,V3,Y3 = fs.M(K-Q,params,args)
                U4,X4,V4,Y4 = fs.M(Q-K,params,args)
                #1
                A1 = np.einsum('sn,sm->nm',np.conjugate(X2),U3)
                B1 = np.einsum('nm,rm->nr',A1,np.conjugate(U3))
                C1 = np.einsum('nr,ri->ni',B1,X2)
                res += np.einsum('ii',C1)
                D1 = np.einsum('nm,rn->mr',A1,Y2)
                E1 = np.einsum('mr,ri->mi',D1,np.conjugate(V3))
                res -= np.einsum('ii',E1)
                #2
                A1 = np.einsum('sn,sm->nm',np.conjugate(X2),np.conjugate(Y4))
                B1 = np.einsum('nm,rm->nr',A1,Y4)
                C1 = np.einsum('nr,ri->ni',B1,X2)
                res += 2*np.einsum('ii',C1)
                D1 = np.einsum('nm,rn->mr',A1,Y2)
                E1 = np.einsum('mr,ri->mi',D1,X4)
                res += 2*np.einsum('ii',E1)
                #3
                A1 = np.einsum('sn,sm->nm',V1,U3)
                B1 = np.einsum('nm,rm->nr',A1,np.conjugate(U3))
                C1 = np.einsum('nr,ri->ni',B1,np.conjugate(V1))
                res += 2*np.einsum('ii',C1)
                D1 = np.einsum('nm,rn->mr',A1,np.conjugate(U1))
                E1 = np.einsum('mr,ri->mi',D1,np.conjugate(V3))
                res += 2*np.einsum('ii',E1)
                #4
                A1 = np.einsum('sn,sm->nm',V1,np.conjugate(V4))
                B1 = np.einsum('nm,rm->nr',A1,V4)
                C1 = np.einsum('nr,ri->ni',B1,np.conjugate(V1))
                res += np.einsum('ii',C1)
                D1 = np.einsum('nm,rn->mr',A1,np.conjugate(U1))
                E1 = np.einsum('mr,ri->mi',D1,X4)
                res -= np.einsum('ii',E1)
        SF[i,j] = 3/2*np.real(res)/(Nx*Ny)

np.save(savename,SF)
