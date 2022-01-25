import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import functions as fs
import input_variables as inp
from tqdm import tqdm
import time
start_time = time.time()

z1 = inp.z1
z2 = inp.z1
#parameters
J1 = inp.J1
J2 = 0
S = inp.S
#ans = inp.ansatz
phi_pts = inp.phi_pts
phi_max = inp.phi_max
range_phi = np.linspace(0,phi_max,phi_pts)

for ans in range(2):
    print('Using ansatz: ',inp.text_ans[ans])
    m = inp.m[ans]
    E_gs = np.zeros((2,phi_pts))
    K1 = np.linspace(0,inp.maxK1[ans],inp.sum_pts)
    K2 = np.linspace(0,inp.maxK2[ans],inp.sum_pts)
    for p,phi in tqdm(enumerate(range_phi)):
        #print("Step : ",p+1,"/",phi_pts)
        params = (J1, J2, phi, ans)

        minRatio = -np.sqrt(max(fs.eigG2_arr(K1,K2,params).ravel()))
        ###### evaluate ratio
        func1 = lambda ratio: np.absolute(2*S+1+fs.sum_lam(ratio,params))
        res = minimize_scalar( func1, #function to be minimized
                method='bounded',
                bounds=(-10000,minRatio),
                options={
                    'maxiter':100,
                    'xatol':1e-7}
                    #'disp':True}
                )
#        if func1(res.x) < 1e-4:
#            print("At phi=","{:10.4f}".format(phi)," SL phase.")
#        else:
#            print("At phi=","{:10.4f}".format(phi)," LRO phase.")
        ratio = res.x
        ##### evaluate xi
        en_rt = fs.sum_mf(ratio,params)
        def E_mf(xi):
            return (xi*(en_rt + (z1*J1+z2*J2)*xi + ratio*(2*S+1)) + (J1*z1+J2*z2)*S**2/2)/(S*(S+1))
        res2 = minimize_scalar( E_mf, #function to be minimized
                method='bounded',
                bounds=(0,1000),
                options={
                    'maxiter':100,
                    'xatol':1e-7}
        #            'disp':True}
                )
        xi = res2.x
        lam = ratio*xi
        E_gs[0,p] = phi
        E_gs[1,p] = E_mf(res2.x)

    text_data = 'data2/' + inp.text_ans[ans] + 'E_gs-' + str(S).replace('.',',') + '.npy'
    np.save(text_data,E_gs)

print("Time taken: ",time.time()-start_time)
