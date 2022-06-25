import numpy as np
import functions as fs
from time import time as T

#structure factor of ansatz ans at (J2,J3) from data in filename
ans = 'cb1'
J1, J2, J3 = (1,0,0)
S = 0.366
DM = True

txt_S = '05' if S == 0.5 else '03'
txt_DM = 'DM' if DM else 'no_DM'
filename = '../Data/S'+txt_S+'/'+txt_DM+'_13/'+'J2_J3=('+'{:5.4f}'.format(J2).replace('.','')+'_'+'{:5.4f}'.format(J3).replace('.','')+').csv'
savenameZZ = "SFs/SFzz_"+ans+'_'+txt_DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
savenameXY = "SFs/SFxy_"+ans+'_'+txt_DM+'_'+txt_S+'_J2_J3=('+'{:5.3f}'.format(J2).replace('.','')+'_'+'{:5.3f}'.format(J3).replace('.','')+').npy'
Kx = 17     #points to compute in the SF BZ
Ky = 17
#kxg = np.linspace(-1,1,Kx)
#kyg = np.linspace(-1,1,Ky)
kxg = np.linspace(-8*np.pi/3,8*np.pi/3,Kx)
kyg = np.linspace(-4*np.pi/np.sqrt(3),4*np.pi/np.sqrt(3),Ky)
##
Nx = 19     #points for summation over BZ
Ny = 19
nxg = np.linspace(0,1,Nx)
nyg = np.linspace(0,1,Ny)

params = fs.import_data(ans,filename)

args = (J1,J2,J3,ans,DM)

SFzz = np.zeros((Kx,Ky))
SFxy = np.zeros((Kx,Ky))
#
for i in range(Kx):
    for j in range(Ky):
        #K = np.array([kxg[i]*2*np.pi,(kxg[i]+2*kyg[j])*2*np.pi/np.sqrt(3)])
        K = np.array([kxg[i],kyg[j]])
        if not fs.EBZ(K):
            SFzz[i,j] = 'nan'
            SFxy[i,j] = 'nan'
            continue
        reszz = 0
        resxy = 0
        Ti = T()
        for ii in range(Nx):
            for ij in range(Ny):
                Q = np.array([nxg[ii]*2*np.pi,(nxg[ii]+nyg[ij])*2*np.pi/np.sqrt(3)])
                U1,X1,V1,Y1 = fs.M(Q,params,args)
                U2,X2,V2,Y2 = fs.M(-Q,params,args)
                U3,X3,V3,Y3 = fs.M(K-Q,params,args)
                U4,X4,V4,Y4 = fs.M(Q-K,params,args)
                #zz1
                A1 = np.einsum('ba,bc->ac',np.conjugate(X1),U4)
                B1 = np.einsum('ac,dc->ad',A1,np.conjugate(U4))
                C1 = np.einsum('ad,de->ae',B1,X1)
                reszz += np.einsum('aa',C1)
                D1 = np.einsum('ac,cd->ad',A1,Y1)
                E1 = np.einsum('ad,ed->ae',D1,np.conjugate(V4))
                reszz -= np.einsum('aa',E1)
                #zz2
                A1 = np.einsum('ab,cb->ac',V2,np.conjugate(Y3))
                B1 = np.einsum('ac,cd->ad',A1,Y3)
                C1 = np.einsum('ad,ed->ae',B1,np.conjugate(V2))
                reszz += np.einsum('aa',C1)
                D1 = np.einsum('ac,dc->ad',A1,np.conjugate(U2))
                E1 = np.einsum('ad,de->ae',D1,X3)
                reszz -= np.einsum('aa',E1)
                #xx+yy1
                A1 = np.einsum('ba,cb->ac',np.conjugate(X1),np.conjugate(Y3))
                B1 = np.einsum('ac,cd->ad',A1,Y3)
                C1 = np.einsum('ad,de->ae',B1,X1)
                resxy += 2*np.einsum('aa',C1)
                D1 = np.einsum('ac,cd->ad',A1,Y1)
                E1 = np.einsum('ad,de->ae',D1,X3)
                resxy += 2*np.einsum('aa',E1)
                #xx+yy2
                A1 = np.einsum('ab,bc->ac',V2,U4)
                B1 = np.einsum('ac,dc->ad',A1,np.conjugate(U4))
                C1 = np.einsum('ad,ed->ae',B1,np.conjugate(V2))
                resxy += 2*np.einsum('aa',C1)
                D1 = np.einsum('ac,dc->ad',A1,np.conjugate(U2))
                E1 = np.einsum('ad,ed->ae',D1,np.conjugate(V4))
                resxy += 2*np.einsum('aa',E1)
        SFzz[i,j] = 3/2*np.real(reszz)/(Nx*Ny)
        SFxy[i,j] = 3/2*np.real(resxy)/(Nx*Ny)
        print("Step ",i*Kx+j,"/",Kx*Ky)#," took ",T()-Ti)
np.save(savenameZZ,SFzz)
np.save(savenameXY,SFxy)
