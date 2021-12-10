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
from emus import linalg as lm


# Define Simulation Parameters
meta_file = 'cv_meta.txt'         # Path to Meta File
dim = 1                             # 1 Dimensional CV space.
period = 360                        # Dihedral Angles periodicity
nbins = 60                          # Number of Histogram Bins.
def group_inverse(A,A0,niter):
    #normA=np.linalg.norm(A)
    #A0=1/normA**2*A
    Ai=A0
    Id=np.eye(np.shape(A)[0])
    for i in np.arange(niter):
        Ai=A0+np.dot((Id-np.dot(A0,A)),Ai)
    return Ai
def calc_states(alltraj):
    ######State Param
    r = 1
    #Acenter = np.array([-1.2,-0.75])
    #Bcenter = np.array([1.0,0.85])
    #Acenter = np.array([-2.7,2.7])
    #Bcenter = np.array([-1.0,2.6])
    #Bcenter = np.array([-1.3,1.3])
    Acenter = 2
    Bcenter=8
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




kT = 2
L=10
I=np.arange(L)
neighbors = [I.astype('int') for i in I]
counter = 0
z_list=[]
z_iter_list=[]
fediff_list=[]
fediff_iter_list=[]
while counter < 300:
    cv_trajs=[]
    for i in I:
        if i in set([4,5,8]):
            s0=np.random.normal(i,math.sqrt(kT),500)
        else:
            s0=np.random.normal(i,math.sqrt(kT),100)
        cv_trajs.append(s0)
    #import matplotlib.pyplot as plt
        #n, bins, patches = plt.hist(x=s[4], bins='auto', color='#0504aa',alpha=0.7, rwidth=0.85)
    psis=[]
    stateA,stateB = calc_states(cv_trajs)
    for i in I:
        l=len(cv_trajs[i])
        psi=np.zeros((l,L))
        ut=np.zeros(L)
        for t in range(l):
            ut=0.5*(cv_trajs[i][t]-I)**2
            psi[t]=np.exp(-ut/kT)
        psis.append(psi)
    #print(psis)
    kappa=[np.shape(psis[i])[0] for i in I]
    kappa=kappa/np.sum(kappa)
    z, F = emus.calculate_zs(psis, n_iter=0)
    z_list.append(z)
    z_iter, F_iter = emus.calculate_zs(psis, neighbors=neighbors, n_iter=5)
    z_iter_list.append(z_iter)
    fediff_iter = -np.log(emus.calculate_avg(psis,z_iter,stateB,stateA,use_iter=True,kappa=kappa))
    fediff_iter_list.append(fediff_iter)
    fediff = -np.log(emus.calculate_avg(psis,z,stateB,stateA,use_iter=False))
    fediff_list.append(fediff)
    #z_avarss = iter_avar.calc_log_z(psis, z, repexchange=False)
    #z, F = emus.calculate_zs(psis, neighbors=neighbors)
    counter+=1
zerr, zcontribs, ztaus = avar.calc_partition_functions(
    psis, z, F, iat_method='acor')
print("Calculated variance in z (noniter): ",zerr)
print("True variance in z:", np.var(np.array(z_list),axis=0))
#log_zerr = np.sqrt(zerr) / z 
# log_zerr_iter, log_zcontribs_iter, log_ztaus_iter = iter_avar.calc_log_z(
#     psis, z_iter, iat_method='acor')
zerr_iter, log_zcontribs_iter, log_ztaus_iter = iter_avar.calc_partition_functions(
    # psis, z_iter, iat_method = np.ones(len(psis)))
    psis, z_iter,iat_method='acor',kappa=kappa)
print("Calculated variance in z_iter: ",zerr_iter)
print("True variance in z_iter:", np.var(np.array(z_iter_list),axis=0))

fe_err_iter, fe_contribs_iter, fe_taus_iter = iter_avar.calc_log_avg_ratio(psis,z_iter,stateB,stateA,iat_method='acor',kappa=kappa)
print("true variance in fediff (iter):",np.var(np.array(fediff_iter_list)))
print("calculated avar for fediff:",fe_err_iter)
iats, fediff_EMUS, fediff_vars = avar.calc_log_avg(psis,z,F,stateB,stateA)
print("true variance in fediff (noniter):",np.var(np.array(fediff_list)))
fe_err = np.sum(fediff_vars)
print('calculated noniter FE diff error:', fe_err)
y=np.array(np.load("err_traj.npy"))
'''
B=np.load("Bmat.npy")
D2=np.diag(kappa)
D2_inv=np.diag(1/kappa)
Z=np.diag(z_iter)
X=np.linalg.multi_dot([D2_inv,B,D2,Z])
print(np.sum(X[0]))
B_ginv=np.linalg.multi_dot([D2,lm.groupInverse_for_iteravar(np.linalg.multi_dot([D2_inv,B,D2]),z_iter),D2_inv])


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



A=np.load("F.npy")
D=np.diag(kappa)
D2=np.diag(z_iter)
D2_inv=np.diag(1/z_iter)
B=np.dot(np.identity(10)-A,D2_inv)
D_inv=np.diag(1/kappa)
B_prime=np.identity(L)-A
B=np.dot(np.dot(D_inv,B_prime),D)
B2=np.dot(B,D2_inv)
print(np.dot(B2,z_iter))
from scipy.linalg import qr
q,r=qr(B2)
print(np.sum(A_prime[2]))

B=np.load("Bmat.npy")
B_ginv=lm.groupInverse_for_iteravar(B)

print(np.linalg.norm(np.dot(B,B_ginv)-np.dot(B_ginv,B)))
print(np.linalg.norm(np.linalg.multi_dot([B,B_ginv,B])-B))
print(np.linalg.norm(np.linalg.multi_dot([B_ginv,B,B_ginv])-B_ginv))
'''