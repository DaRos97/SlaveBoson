import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import functions_2nn as fs
import inputs_2nn as inp
from tqdm import tqdm
from pathlib import Path
from scipy.interpolate import interp1d
import time
start_time = time.time()
temp = time.time()

dirname = 'data2nn/'
do_plot = False
#Fixed parameters
ans = 1
z1 = inp.z1
z2 = inp.z1
J1 = inp.J1
J2 = inp.J2
S = inp.S

convergence_tol = 1e-3
kp = inp.sum_pts
dp = 200
print('Using ansatz: ',inp.text_ans[ans])

K1 = np.linspace(0,inp.maxK1[ans],kp)  #Kx in BZ
K2 = np.linspace(0,inp.maxK2[ans],kp)  #Ky in BZ

#bounds function
path_alpha = Path('alphaD/alphas_kp='+str(kp)+'dp='+str(dp)+'J2='+str(J2).replace('.',',')+'.npy')
if not(path_alpha.is_file()):
    print("Evaluating alpha pts for (",kp,"",dp,"",J2,")...")
    fs.generate_alpha(params,kp,dp)
data_alpha = np.load(path_alpha)
interp = interp1d(data_alpha[0],data_alpha[1],'cubic')

#initial guess
minAlpha = np.pi/4
maxAlpha = np.pi/2

phi_pts = inp.phi_pts
phi_max = inp.phi_max
range_phi = np.linspace(0,phi_max,phi_pts)
E_gs = np.ndarray((2,phi_pts))
for p,phi in enumerate(range_phi):
    params = (J1, J2, phi, ans)
    alpha = maxAlpha
    step = 0
    stay = True
    en_arr = [0]
    alpha_arr = [alpha]
    while stay:
        step += 1
        print("Step ",step,":\t alpha = ",alpha)
        g2 = fs.eigG2_arr(K1,K2,params,alpha)
        minRatio = -np.sqrt(max(g2.ravel()))
        #Evaluate ratio from the stationary condition on lam
        condensate = lambda Ratio: np.absolute(2*S+1+fs.sum_lam(Ratio,ans,g2))
        res = minimize_scalar( condensate,
                method='bounded',
                bounds=(-10000,minRatio),
                options={
                    'xatol':1e-15}
                )
        ratio = res.x
        cond = condensate(ratio)
        if not(res.success):
            print("Error in minimizing ratio")
################
        print("\t\t ratio = ",ratio," and condensate = ",cond)
        #Evaluate xi by minimizing the energy knowing the ratio
        en_rt = fs.sum_mf(ratio,ans,g2)
        E_mf = lambda Xi: (Xi*(en_rt + (z1*J1*np.sin(alpha)**2+z2*J2*np.cos(alpha)**2)*Xi + ratio*(2*S+1)) + (J1*z1+J2*z2)*(S**2)/2)/(S*(S+1))
        res2 = minimize_scalar( E_mf, method='brent')
        if not(res2.success):
            print("Error in minimizing xi")
        xi = res2.x
        E_mff = E_mf(xi)
################
        print("\t\t xi = ",xi)
        print("\t\t E_mf = ",E_mff)
        #Find alpha minimizing E_mf2
        maxAlpha = fs.bounds_alpha(ratio,interp)
        if do_plot:
            plt.figure()
            alla = np.linspace(0,np.pi,100)
            plt.plot(alla,interp(alla),'r-')
            plt.hlines(-ratio,alla[0],alla[-1],'b')
            plt.vlines(minAlpha,0,4,'g')
            plt.vlines(maxAlpha,0,4,'g')
            plt.show()
        en_ = lambda Alpha: fs.sum_mf(ratio,ans,fs.eigG2_arr(K1,K2,params,Alpha))
        E_mf2 = lambda Alpha: (xi*(en_(Alpha) + (z1*J1*np.sin(Alpha)**2+z2*J2*np.cos(Alpha)**2)*xi + ratio*(2*S+1)) + (J1*z1+J2*z2)*(S**2)/2)/(S*(S+1))
        if np.abs(alpha-minAlpha) > 1e-4:
            res3 = minimize_scalar( E_mf2,
                method='bounded',
                bounds=(minAlpha,maxAlpha),
                options={
                    'xatol':1e-15}
                )
            if not(res3.success):
                print("Error in minimizing alpha")
    #Update alpha value
            alpha = res3.x
            alpha_arr = alpha_arr + [alpha]
        en_arr = en_arr + [E_mff]
################
        print("\t\t After minimization, alpha = ",alpha," and E = ",en_arr[step])
        if step>1:
            if np.abs(en_arr[step]-en_arr[step-1]) < convergence_tol:
                stay = False
    
    print("Minimization of phi = ",phi," took ",step," steps and time: ",time.time()-temp)
    temp = time.time()
    E_gs[0,p] = phi
    E_gs[1,p] = en_arr[-1]


#Save energy values and phi values externally
text_data = dirname + inp.text_ans[ans] + 'E_2nn-J2=' + str(J2).replace('.',',') + 'phi1.npy'
np.save(text_data,E_gs)

print("Time taken: ",time.time()-start_time)
