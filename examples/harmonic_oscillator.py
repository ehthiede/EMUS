# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 02:52:39 2019

@author: xiang
"""

# -*- coding: utf-8 -*-
import site
import sys
site.addsitedir('D:/academics/chemistry/iEMUS/iemus/src/') 
import numpy as np
import math
import emus
import numpy as np
from emus import usutils as uu
from emus import emus, avar, iter_avar
import matplotlib.pyplot as plt

# Define Simulation Parameters
meta_file = 'cv_meta.txt'         # Path to Meta File
dim = 1                             # 1 Dimensional CV space.
period = 360                        # Dihedral Angles periodicity
nbins = 60                          # Number of Histogram Bins.

def calc_states(alltraj):
    ######State Param
    r = 0.2
    #Acenter = np.array([-1.2,-0.75])
    #Bcenter = np.array([1.0,0.85])
    #Acenter = np.array([-2.7,2.7])
    #Bcenter = np.array([-1.0,2.6])
    #Bcenter = np.array([-1.3,1.3])
    Acenter = -1
    Bcenter=1
    #Bcenter = np.array([1.2,-1.2])
    #################
    stateA = [] 
    stateB = [] 
    for traj in alltraj:
        dA = (traj-Acenter)**2
        dB = (traj-Bcenter)**2
        stateA_rep = (dA  < r).astype('float')
        stateB_rep = (dB  < r).astype('float')
        stateA.append(stateA_rep)
        stateB.append(stateB_rep)
    return np.array(stateA),np.array(stateB)

kT = 1
centers=np.append(-np.sqrt(np.linspace(2,0,10,endpoint=False)),np.sqrt(np.arange(11)/5))
L=len(centers)
fk=10
#print(np.max(np.array([centers[i+1]-centers[i] for i in np.arange(20)])))
counter = 0
z_list=[]
z_iter_list=[]
fediff_list=[]
fediff_iter_list=[]
while counter < 300:
    cv_trajs=[]
    for i in np.arange(L):
        if i in set([10]):
             s0=np.random.normal(fk*centers[i]/(fk+1),1/math.sqrt(2*(fk+1)),400)
        else:
            s0=np.random.normal(fk*centers[i]/(fk+1),1/math.sqrt(2*(fk+1)),100)
        cv_trajs.append(s0)
    #import matplotlib.pyplot as plt
        #n, bins, patches = plt.hist(x=s[4], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    psis=[]
    stateA,stateB = calc_states(cv_trajs)
    for i in np.arange(L):
        l=len(cv_trajs[i])
        psi=np.zeros((l,L))
        ut=np.zeros(L)
        for t in range(l):
            ut=fk*(cv_trajs[i][t]-centers)**2
            psi[t]=np.exp(-ut/kT)
        psis.append(psi)
    #print(psis)
    kappa=[np.shape(psis[i])[0] for i in np.arange(L)]
    kappa=kappa/np.sum(kappa)
    z, F= emus.calculate_zs(psis, n_iter=0,iat_method="acor")
    z_list.append(z)
    z_iter, F_iter= emus.calculate_zs(psis, n_iter=5)
    z_iter_list.append(z_iter)
    fediff_iter = -np.log(emus.calculate_avg(psis,z_iter,stateB,stateA,use_iter=True,kappa=kappa))
    fediff_iter_list.append(fediff_iter)
    fediff = -np.log(emus.calculate_avg(psis,z,stateB,stateA,use_iter=False,kappa=kappa))
    fediff_list.append(fediff)
    #z_avarss = iter_avar.calc_log_z(psis, z, repexchange=False)
    #z, F = emus.calculate_zs(psis, neighbors=neighbors)
    counter+=1
zerr, zcontribs, ztaus = avar.calc_partition_functions(
    psis, z, F, iat_method='acor')

#log_zerr = np.sqrt(zerr) / z 
# log_zerr_iter, log_zcontribs_iter, log_ztaus_iter = iter_avar.calc_log_z(
#     psis, z_iter, iat_method='acor')
zerr_iter, log_zcontribs_iter, log_ztaus_iter = iter_avar.calc_partition_functions(
    # psis, z_iter, iat_method = np.ones(len(psis)))
    psis, z_iter,iat_method='acor',kappa=kappa)
print("Calculated variance in z_iter: ",zerr_iter)
print("True variance in z_iter:", np.var(np.array(z_iter_list),axis=0))
print("Calculated variance in z (noniter): ",zerr)
print("True variance in z:", np.var(np.array(z_list),axis=0))
fe_err_iter, fe_contribs_iter, fe_taus_iter = iter_avar.calc_log_avg_ratio(psis,z_iter,stateB,stateA,iat_method='acor')
print("true variance in fediff (iter):",np.var(np.array(fediff_iter_list)))
print("calculated avar for fediff:",fe_err_iter)
iats, fediff_EMUS, fediff_vars = avar.calc_log_avg(psis,z,F,stateB,stateA,kappa=kappa)
print("true variance in fediff (noniter):",np.var(np.array(fediff_list)))
fe_err = np.sum(fediff_vars)
print('calculated noniter FE diff error:', fe_err)

'''
#avar1=np.zeros(10)
avar2=np.zeros(10)
for i in I:
    #avar1[i]=np.sqrt(np.var(-np.log(Z[:,i])))
    avar2[i]=np.sqrt(np.var(z_iter_list[:,i]))
#print(avar1)
print("True std in z: ",avar2)
N = len(z_iter)
#plt.errorbar(np.arange(N), -np.log(z), yerr=log_zerr, label='noniterative')
#plt.legend()
#plt.show()
plt.errorbar(np.arange(N), -np.log(z_iter), yerr=np.sqrt(zerr_iter), label='iterative')
plt.legend()
plt.show()


#plt.plot(np.arange(N), log_zerr, label='noniterative')
#plt.plot(np.arange(N),avar1, label='noniterative from simulation')
plt.plot(np.arange(N), np.sqrt(zerr_iter), label='iterative')
plt.plot(np.arange(N),avar2, label='iterative from simulation')
plt.legend()
plt.show()



'''