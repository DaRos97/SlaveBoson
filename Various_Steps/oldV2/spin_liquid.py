import inputs as inp
import ansatze as an
import numpy as np
from scipy import linalg as LA
from scipy.interpolate import interp2d
import csv
import matplotlib.pyplot as plt
from matplotlib import cm
import os

#Some parameters from inputs.py
m = inp.m
kp = inp.sum_pts
J1 = inp.J1
#grid points
grid_pts = inp.grid_pts         #IMPORTANT
minPts = 6000
####
J = np.zeros((2*m,2*m))
for i in range(m):
    J[i,i] = -1
    J[i+m,i+m] = 1
#
dirname = '../Data/noDMbig/copy/'
# main code
for filename in os.listdir(dirname):
    print("Doing file ",dirname+filename)
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()
    N = (len(lines)-1)//4 + 1
    for I in range(N):
        head = inp.header[lines[I*4+1].split(',')[0]]
        data = lines[I*4+1].split(',')
        for n in range(1,len(data)):
            data[n] = float(data[n])
        ans = data[0]
        print("ansatz :",ans)
        J2 = data[1]
        J3 = data[2]
        gapI = data[5]
        L = data[6]
        P = data[7:]
#        res = findPhase(P,L,J2,J3,ans)      #returns either 'SL' or 'LRO'
        args = (inp.J1,J2,J3,ans)
        N = an.Nk(P,L,args) #compute Hermitian matrix
        res = np.zeros((m,grid_pts,grid_pts))
        for i in range(grid_pts):
            for j in range(grid_pts):
                Nk = N[:,:,i,j]
                try:
                    K = LA.cholesky(Nk)
                except LA.LinAlgError:
                    #print("not good for ",i,j,ans)
                    ff = False
                    continue
                temp = np.dot(np.dot(K,J),np.conjugate(K.T))
                res[:,i,j] = np.sort(np.tensordot(J,LA.eigvalsh(temp),1)[:m])
        #
        if ff:
        #fig = plt.figure(figsize=(8,8))
        #ax = fig.add_subplot(111, projection='3d')
            func = interp2d(inp.kg[0],inp.kg[1],res[0],kind='cubic')    #Interpolate the 2D surface
            Kp = (np.linspace(0,inp.maxK1,minPts),np.linspace(0,inp.maxK2,minPts))
            tempB = func(Kp[0],Kp[1])
            gapF = np.amin(tempB.ravel())
        else:
            print("LRO")
            gapF = gapI
        #fig
        #X,Y = np.meshgrid(Kp[0],Kp[1])
        #Z = tempB
        #ax.plot_surface(X,Y,Z,cmap=cm.coolwarm)
        #plt.show()
        print("Initial/Final  gap: ",gapI,'\t',gapF)
        input()
        continue
        #saveData(data,i,filename,res)
        with open (dirname+filename, 'r') as f:
            lines = f.readlines()
        Head = list(inp.header[data[0]])
        Head.append('phase')
        Data = {'ans':data[0]}
        for i in range(1,len(data)):
            Data[Head[i]] = float(data[i])
        Data[Head[-1]] = res
        #write to file
        with open(dirname+filename,'w') as f:
            for l in range(4*i):
                f.write(lines[l])
        with open(dirname+filename, 'a') as f:
            writer = csv.DictWriter(f, fieldnames = Head)
            writer.writeheader()
            writer.writerow(Data)
        with open(dirname+filename,'a') as f:
            for l in range(4*i+2,len(lines)):
                f.write(lines[l])
        with open(dirname+filename, 'r') as f:
            lines = f.readlines()



#
def findPhase(P,L,J2,J3,ans):
    args = (inp.J1,J2,J3,ans)
    N = an.Nk(P,L,args) #compute Hermitian matrix
    res = np.zeros((m,grid_pts,grid_pts))
    for i in range(grid_pts):
        for j in range(grid_pts):
            Nk = N[:,:,i,j]
            try:
                K = LA.cholesky(Nk)
            except LA.LinAlgError:
                print("not good for ",J2,J3,ans)
                return 100
            temp = np.dot(np.dot(K,J),np.conjugate(K.T))
            res[:,i,j] = np.sort(np.tensordot(J,LA.eigvalsh(temp),1)[:m])
    #
    #fig = plt.figure(figsize=(8,8))
    #ax = fig.add_subplot(111, projection='3d')
    func = interp2d(inp.kg[0],inp.kg[1],res[0],kind='cubic')    #Interpolate the 2D surface
    Kp = (np.linspace(0,inp.maxK1,minPts),np.linspace(0,inp.maxK2,minPts))
    tempB = func(Kp[0],Kp[1])
    Min = np.amin(tempB.ravel())
    #fig
    #X,Y = np.meshgrid(Kp[0],Kp[1])
    #Z = tempB
    #ax.plot_surface(X,Y,Z,cmap=cm.coolwarm)
    #plt.show()
    return Min
#






def saveData(data,I,filename,res):
    with open (dirname+filename, 'r') as f:
        lines = f.readlines()
    Head = list(inp.header[data[0]])
    Head.append('phase')
    Data = {'ans':data[0]}
    for i in range(1,len(data)):
        Data[Head[i]] = float(data[i])
    Data[Head[-1]] = res
    #write to file
    with open(dirname+filename,'w') as f:
        for l in range(4*I):
            f.write(lines[l])
    with open(dirname+filename, 'a') as f:
        writer = csv.DictWriter(f, fieldnames = Head)
        writer.writeheader()
        writer.writerow(Data)
    with open(dirname+filename,'a') as f:
        for l in range(4*I+2,len(lines)):
            f.write(lines[l])
    with open(dirname+filename, 'r') as f:
        lines = f.readlines()

