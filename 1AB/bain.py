import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
from scipy.interpolate import interp1d
import functions as fs
import inputs as inp
import time
from colorama import Style,Fore
from tqdm import tqdm
start_time = time.time()

S = inp.S

minA = 0
maxA = (2*S+1)/2
minB = 0
maxB = S
minL = 0
maxL = maxA
bn = [[minA,maxA],[minB,maxB],[minL,maxL]]

pts = inp.pts
m = inp.m

ti = time.time()
s = fs.Sigmaa(0.1,0.01,0.3)
tt = time.time()-ti
print("Time of 1 evaluation is : ",tt)
print("Expected time per cicle : ",tt*pts**3/60," mins")

dim = pts-m+1
cicle = 1
Stay = True
while Stay:
    ti = time.time()
    print(Fore.RED,"Initiating cicle ",cicle,Style.RESET_ALL)
    print("Range of parameters: \n",
            "A: ",bn[0][0]," to ",bn[0][1],'\n',
            "B: ",bn[1][0]," to ",bn[1][1],'\n',
            "L: ",bn[2][0]," to ",bn[2][1])
    a = np.linspace(bn[0][0],bn[0][1],pts)
    b = np.linspace(bn[1][0],bn[1][1],pts)
    l = np.linspace(bn[2][0],bn[2][1],pts)
    e = np.ndarray((pts,pts,pts))
    for i,ai in tqdm(enumerate(a)):
        for j,bi in enumerate(b):
            for k,li in enumerate(l):
                e[i,j,k] = fs.Sigmaa(ai,bi,li)
    #decompose in mean values
    em = np.zeros((dim,dim,dim))
    for i in range(dim):
        for j in range(dim):
            for k in range(dim):
                for m1 in range(m):
                    for m2 in range(m):
                        for m3 in range(m):
                            em[i,j,k] += e[i+m1,j+m2,k+m3]
                em[i,j,k] /= m**3
    earr = em.ravel()
    emin = np.amin(earr)
    indMin = np.argmin(earr)
    indA = indMin//dim**2
    indB = (indMin-indA*dim**2)//dim
    indL = indMin-indA*dim**2-indB*dim
    if emin != em[indA,indB,indL]:
        print("error")
        exit()
    
    Emin = np.amin(e[indA:indA+m,indB:indB+m,indL:indL+m].ravel())
    Ergmin = np.argmin(e[indA:indA+m,indB:indB+m,indL:indL+m].ravel())
    Aind = Ergmin//m**2
    Bind = (Ergmin-Aind*m**2)//m
    Lind = Ergmin-Aind*m**2-Bind*m
    
    TotA = indA+Aind
    TotB = indB+Bind
    TotL = indL+Lind
    print("Min medium energy is: ",emin)
    print("and min energy in the mean is ",Emin)
    print(indA,indB,indL)
    print(Aind,Bind,Lind)
    
    bnAl = a[TotA-m//2] if TotA-m//2 >= 0 else a[0]
    bnBl = a[TotB-m//2] if TotB-m//2 >= 0 else b[0]
    bnLl = a[TotL-m//2] if TotL-m//2 >= 0 else l[0]
    bnAu = a[TotA+m//2] if TotA+m//2 < pts else a[pts]
    bnBu = a[TotB+m//2] if TotB+m//2 < pts else b[pts]
    bnLu = a[TotL+m//2] if TotL+m//2 < pts else l[pts]
    bn[0] = [bnAl,bnAu]
    bn[1] = [bnBl,bnBu]
    bn[2] = [bnLl,bnLu]

    tf = time.time()
    print("Time of cicle ",cicle," is ",(tf-ti)/60," mins")
    A = (bn[0][0]+bn[0][1])/2
    B = (bn[1][0]+bn[1][1])/2
    L = (bn[2][0]+bn[2][1])/2
    s = fs.Sigmaa(A,B,L)
    print("Sigma is : ",s)
    if s < inp.cutoff:
        Stay = False
    cicle += 1

print("Results:")
print("Min values found:",A,B,L)
print("Sigma : ",s)
e = fs.tot_E([A,B,L])
print(e)

print("Tima taken: ",time.time()-start_time)
