import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import functions as fs
import inputs as inp
from tqdm import tqdm
import time
start_time = time.time()

dirname = 'Data3/'

#Fixed parameters
z1 = inp.z1
J1 = inp.J1
S = inp.S
phi_pts = inp.phi_pts
phi_max = inp.phi_max
range_phi = np.linspace(0,phi_max,phi_pts)
kp = inp.sum_pts

for ans in range(2):    #sum over the ansatze
    print('Using ansatz: ',inp.text_ans[ans])
    m = inp.m[ans]  #unit cell of that ansatz
    E_gs = np.zeros((2,phi_pts))    #array for phi and 
                                    #corresponding energy value
    K1 = np.linspace(0,inp.maxK1[ans],kp)  #Kx in BZ
    K2 = np.linspace(0,inp.maxK2[ans],kp)  #Ky in BZ
    for p,phi in tqdm(enumerate(range_phi)):    #sum over phi pts
        params = (J1, phi, ans)
        g2 = fs.eigG2_arr(K1,K2,params)
        minRatio = -np.sqrt(max(g2.ravel()))
        #Evaluate ratio from the stationary condition on lam
        func1 = lambda ratio: np.absolute(2*S+1+fs.sum_lam(ratio,ans,g2))
        res = minimize_scalar( func1,
                method='bounded',
                bounds=(-10000,minRatio),
                options={
                    'xatol':1e-15}
                )
        ratio = res.x
        if not(res.success):
            print("Minimization error in ratio")
        #Evaluate xi by minimizing the energy knowing the ratio
        en_rt = fs.sum_mf(ratio,ans,g2)
        E_mf = lambda xi: (xi*(en_rt + 2*z1*J1*xi + ratio*(2*S+1)) + J1*z1*(S**2))/(S*(S+1))
        res2 = minimize_scalar( E_mf, method='brent')
        if not(res2.success):
            print("Minimization error in xi")
        xi = res2.x
        lam = ratio*xi
        E_gs[0,p] = phi
        E_gs[1,p] = E_mf(res2.x)
    #Save energy values and phi values externally
    text_data = dirname + inp.text_ans[ans] + 'E_gs-' + str(S).replace('.',',') + '.npy'
    np.save(text_data,E_gs)

print("Time taken: ",time.time()-start_time)
