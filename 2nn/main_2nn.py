import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar
import functions_2nn as fs
import inputs_2nn as inp
from tqdm import tqdm
from pathlib import Path
from scipy.interpolate import interp1d
from colorama import Fore, Style
import time
start_time = time.time()
temp = time.time()

dirname = 'Data/'
do_plot = True
#Fixed parameters
ans = 1
S = inp.S
z1 = inp.z1
z2 = inp.z2
z3 = inp.z3
J1 = inp.J1

J2_max = inp.J2_max     #function of J2
J2_pts = inp.J2_pts
J2_range = np.linspace(0,J2_max,J2_pts)
J3_max = inp.J3_max     #function of J2
J3_pts = inp.J3_pts
J3_range = np.linspace(0,J3_max,J3_pts)

convergence_tol = 1e-3
kp = inp.sum_pts
dp = inp.alpha_pts

K1 = np.linspace(0,inp.maxK1[ans],kp)  #Kx in BZ
K2 = np.linspace(0,inp.maxK2[ans],kp)  #Ky in BZ

E_gs = np.ndarray((3,J2_pts,J3_pts))
#bounds function
for ans in range(2):
    print('Using ansatz: ',inp.text_ans[ans])
    for j3,J3 in enumerate(J3_range):       #iterate over J3 values
        for j2,J2 in enumerate(J2_range):       #iterate over J2 values
            params = (J1, J2, J3, ans)
            const = (J1*z1+J2*z2+J3*z3)*S**2
            #check if values of alpha for given J2,J3 have already been computed, if not do it
            if ans == 0:
                path_alpha = Path('alphaD/alphas-'+inp.text_ans[ans]+'_J3='+"{:.4f}".format(J3).replace('.',',')+'.npy')
            elif ans ==1:
                path_alpha = Path('alphaD/alphas-'+inp.text_ans[ans]+'_J2='+"{:.4f}".format(J2).replace('.',',')+'.npy')
            if not(path_alpha.is_file()):
                print("Evaluating alpha pts for ans ",inp.text_ans[ans])
                fs.generate_alpha(params,kp,dp,path_alpha)
            data_alpha = np.load(path_alpha)
            #interpolate pts of maxG2 as a function of alpha
            interp = interp1d(data_alpha[0],data_alpha[1],'cubic')
            xlin = np.linspace(min(data_alpha[0]),max(data_alpha[0]),1000)
            #initial guess, so that MF hopping moduli are positive
            minAlpha = 0  #maybe 0
            maxAlpha = np.pi/4

            alpha = np.pi/8
            step = 0
            stay = True
            en_arr = [0]
            print("Initiating loop with J3 = ","{:.4f}".format(J3)," and J2 = ","{:.4f}".format(J3))
            while stay:
                step += 1
                print(Fore.GREEN+"Step ",step,":\t initial alpha = ",alpha,Fore.RED)
                g2 = fs.eigG2(K1,K2,params,alpha)
                minRatio = -interp(alpha)
                #Evaluate ratio from the stationary condition on lam
                condensate = lambda Ratio: np.absolute(2*S+1+fs.sum_lam(Ratio,ans,g2))
                res = minimize_scalar( condensate,
                        method='bounded',
                        bounds=(-10000,minRatio),
                        options={
                            'xatol':1e-15}
                        )
                if not(res.success):
                    print("Error in minimizing ratio")
                ratio = res.x
                cond = condensate(ratio)
                if np.abs(cond) > 1e-4:
                    print("\t LRO phase")
                else:
                    print("SL phase")
                print("\t ratio = ",ratio, "and minRatio = ",minRatio)
        #############Evaluate xi by the second constraint, and consider it has to be diff from 0
                xi = fs.sum_xi(params,alpha,ratio,g2)
                print("\t xi found = ",xi)
                #en_rt = fs.sum_mf(ratio,ans,g2)
                #E_mf = lambda Xi: Xi*(en_rt + 2*(z1*J1*np.cos(alpha)**2+ans*z2*J2*np.sin(alpha)**2+(1-ans)*z3*J3*np.sin(alpha)**2)*Xi
                #    + ratio*(2*S+1)) + const
                #res2 = minimize_scalar(E_mf, method='brent')
                #if not(res2.success):
                #    print("Error in minimizing xi")
                #xi = res2.x
                #E_mff = E_mf(xi)
                #print("\t after second minimization, xi = ",xi, "and E_mf = ",E_mff)
        #############Find alpha minimizing E_mf2
                minAlpha = fs.bnd_alpha(data_alpha,ratio)
                if do_plot:
                    plt.figure()
                    plt.plot(xlin,interp(xlin),'r-')
                    plt.scatter(data_alpha[0],data_alpha[1],color='k',marker='*')
                    plt.hlines(-ratio,xlin[0],xlin[-1],'b')
                    plt.vlines(minAlpha,0,10,'y')
                    plt.vlines(maxAlpha,0,10,'g')
                    plt.show()
                en_ = lambda Alpha: fs.sum_mf(ratio,ans,fs.eigG2(K1,K2,params,Alpha))
                E_mf2 = lambda Alpha: xi*(en_(Alpha) + (z1*J1*np.cos(Alpha)**2+ans*z2*J2*np.sin(Alpha)**2+(1-ans)*z3*J3*np.sin(Alpha)**2)*2*xi
                    + ratio*(2*S+1)) + const
                res3 = minimize_scalar(E_mf2,
                    method='bounded',
                    bounds=(minAlpha,maxAlpha),
                    options={
                        'xatol':1e-15}
                        )
                if not(res3.success):
                    print("Error in minimizing alpha")
                #Update alpha value
                alpha = res3.x
                en_arr = en_arr + [E_mf2(alpha)]
        ############
                print("\t After minimization, alpha = ",alpha," and E = ",en_arr[step])
                if np.abs(en_arr[step]-en_arr[step-1]) < convergence_tol:
                    stay = False

            print(Style.RESET_ALL)
#print("Minimization of J2 = ",J2," took ",step," steps and time: ",time.time()-temp)
#temp = time.time()
#E_gs[0,j2] = J2
#E_gs[1,j2] = en_arr[-1]


#Save energy values and phi values externally
#text_data = dirname + inp.text_ans[ans] + 'E_2nn' + '.npy'
#np.save(text_data,E_gs)

print(Style.RESET_ALL)
print("Time taken: ",time.time()-start_time)
