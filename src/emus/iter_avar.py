# -*- coding: utf-8 -*-
"""
Library with routines associated with the asymptotic variance for iterative EMUS.
"""

from __future__ import absolute_import
import numpy as np
#from usutils import data_from_meta
from numpy import linalg as LA
from . import autocorrelation as ac
from ._defaults import DEFAULT_IAT, DEFAULT_KT

#
#T = 310                             # Temperature in Kelvin
#k_B = 1.9872041E-3                  # Boltzmann factor in kcal/mol
#kT = k_B * T
#meta_file = 'cv_meta.txt'         # Path to Meta File
#dim = 1                             # 1 Dimensional CV space.
#period = 360                        # Dihedral Angles periodicity
#nbins = 60                          # Number of Histogram Bins.

#meta_file = 'cv_meta.txt'   
#psis, cv_trajs, neighbors = data_from_meta(
#    meta_file, dim=dim, T=T, k_B=k_B, period=period)
def calc_p_traj(psis, z):
    """
    Estimates the trajectories :math:`p_j(X_t^i)` described in REF.

    Parameters
    ----------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.  
        See `datastructures <../datastructures.html#data-from-sampling>`__ for 
        more information.
    z : 1D array
        Array containing the normalization constants calculated using 
        Iterative EMUS

    Returns
    ------
    p_traj : list of 2d arrays
        Values of the P trajectory. p_traj[i][t,j] gives the value of 
        :math:`p_j(X_t^i)`.
    
    Variables used
    --------------
    L: number of windows.
    N: 1D array
        N[i]: number of points collected in window i.
    Nt: total number of points.
    """
    L=len(psis)
    N=np.zeros(L)
    for i in range(L):
        N[i]=psis[i].shape[0]
    Nt=np.sum(N)
    p_traj=[]
    for i in range(L):
        pi=np.zeros((int(N[i]),L))
        for t in range(int(N[i])):
            for j in range(L):
                pi[t,j]=(N[j]/Nt*psis[i][t,j]/z[j])/np.dot(N/Nt,psis[i][t]/z)
        p_traj.append(pi)
    return p_traj


def calc_B_matrix(psis,p_traj):
    """
    Estimates the B matrix in REF.

    Parameters
    ----------
    p_traj : list of 2d arrays
        Values of the P trajectory. p_traj[i][t,j] gives the value of 
        :math:`p_j(X_t^i)`.

    Returns
    -------
    B : 2d matrix
        B matrix in REF.
    """
    L=len(psis)
    N=np.zeros(L)
    for i in range(L):
        N[i]=psis[i].shape[0]
    Nt=np.sum(N)
    B = np.zeros((L,L))
    for r in range(L): 
        for i in range(L):
            s1=0
            for t in range(int(N[i])):
                s1+=p_traj[i][t,r]*(1-p_traj[i][t,r])
            B[r,r]+=s1
    for r in range (L):
        for s in range (r):
            for i in range(L):
                s1=0
                for t in range(int(N[i])):
                    s1+=p_traj[i][t,s]*p_traj[i][t,r]
                B[r,s]-=s1
            B[s,r]=B[r,s]
    B=B/Nt
    return B


def calc_log_z(psis, z, repexchange=False):
    """
    Calculates the asymptotic variance in the log partition functions.
    """
    p_traj = calc_p_traj(psis, z)
    B = calc_B_matrix(psis,p_traj)

    # Construct trajectories for autocovariance.
    B_pseudo_inv = np.linalg.pinv(B)
    zeta_traj = [np.dot(p_traj_i, B_pseudo_inv.T) for p_traj_i in p_traj]
    print(np.shape(zeta_traj))
    print(B)
    if repexchange:
        # ## FINISH! ## #
        zeta_sum = np.sum(zeta_traj, axis=0)
        #avars = 
        return B
    else:
        # ## FINISH! ## #
        return B
if __name__=='__main__':
	main()