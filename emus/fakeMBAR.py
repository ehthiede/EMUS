# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 18:34:13 2015

@author: Erik
"""

import numpy as np
#from acor import acor
import linalg_methods as lam
import acor


# BOLTZ_kJ = 0.0083144621
# TEMP = 298



def _uklt_unpacker(uklt):
    """
    Unpacks values from the the pymbar uklt format into a list of i arrays, 
    where each array is T by N.
    
    Parameters
    ----------
    uklt : 3 dimensional array, reduced potential u of state k calculated at 
        state l, evaluated at time t.
        
    Returns
    -------
    q_traj : a list of two dimensional arrays, with element i j as the value of
    biased density j at time point i.
    """
    N = uklt.shape[0]
    
    qtraj = []    
    for i in xrange(N):
        qtraj.append(np.transpose(np.exp(-uklt[i,:,:])))
    
    return qtraj
    

def pl_constants(qtraj, Avals=None, use_Nj=False, return_taus = False):
    """
    Returns the normalization constants of the umbrellas, calculated according
    to the power law estimator method with power 1. (see Meng and Wong).
    
    Parameters
    ----------
    
    q_traj : A list of two dimensional arrays, with each list representing data
    from one source.  Each row of one of these arrays represents a time point,
    and each column represents the value of that ponit in another unnormalized
    probability density.
    
    Avals : An inverse coefficient to scale data from each umbrella with.  For 
    MBAR, this is 1/(c_i \tau_i), where c is the normalization constant for 
    each umbrella and tau is an estimate of the autocorrelation time of the
    umbrella.
    
    use_Nj : Whether to scale parts of the power law estimator by the number of
    samples.  Takes as options True or False (estimate the
    autocorrelation time of the trajectory using acor)
        
    Returns
    -------
    
    c : array of the partition functions for each state.
    
    F : the stochastic matrix for the eigenproblem.
    """
    
    # Initialize variables
    L = len(qtraj) # Number of Windows
    F = np.zeros((L,L)) # Initialize F Matrix
    if return_taus:
        taumat = np.zeros((L,L))
    
    Avs = Avals
    if Avals is None:
        Avs = np.ones(L)
    Njs = np.ones(L)
    if use_Nj:
        Njs  = np.array([len(qidata) for qidata in qtraj])
    print Njs
            
#    print Njs
    scaling = Njs*Avs    # Calculate scaling vector, N_k*A_k
    # Calculate Fi: for each i
    for i in xrange(L):
        qi_data = qtraj[i]
        if qi_data.shape[1] != L:
            raise Exception("Hamiltonian not calculated over all states")
        
        # Calculate the value of our estimating function, alpha
        scalemat = np.diag(scaling)
        alpha_denom = 1./np.sum(np.dot(qi_data,scalemat),axis=1)
        alpha_vals = np.outer(alpha_denom , np.ones(L)*scaling[i])
#        print alpha_vals.shape, qi_data.shape
        
        Ftraj = qi_data * alpha_vals # Calculate the trajectory of alpha times q
        Fijs = np.average(Ftraj , axis=0) # Average along time axis to get F_ij
        F[i,:]=Fijs
        if return_taus:
            for j in xrange(L):
                taumat[i,j] = acor.acor(Ftraj[:,j])[0]
                print "tau %d,%d," % (i,j), taumat[i,j]
    
    c = lam.old_stationary_distrib(F)
    
    if return_taus:
        return c, F, taumat
    else:
        return c, F

    
    
def hackishMBAR(qtraj,maxIter = 1000,tol=1e-6, cguess = None, usetaus=None, outputIter = False):
    """
    A hacked together version of MBAR that works by calling the eigenvector
    routine over and over again.
    """
    # Initialize variables
    L = len(qtraj) # Number of Windows
    
    if usetaus is None or usetaus is "true":
        taus = np.ones(L)
    elif hasattr(usetaus,'__iter__'):
        taus = usetaus
    else: raise RuntimeError("tau method not comprehended.  Supported is None, True, or fixed values (iterable)")
    if cguess is None:
        cguess = np.ones(L)
        
    # we perform the self consistent iteration.
    c_old = cguess
    for i in xrange(maxIter):
        As = 1./(c_old*taus)
        if usetaus == "true" and i == 0:
            c, Fnew, taumat = pl_constants(qtraj, use_Nj=True, Avals=As, return_taus = True)
        else:
            c, Fnew = pl_constants(qtraj, use_Nj=True, Avals=As)
        if np.linalg.norm((c-c_old)/c_old) < tol:
            break
        c_old = c        
        
        # Calc new values for the autocorrelation constants.
        if usetaus == "true" and i == 0:
            taus =  np.average(taumat,axis=1)
            print "All taus," , taus
        
        if i == maxIter:
            print "Failed to converge!"
            return c, Fnew # Temporary Hack to allow me to just do one iter.
#            raise Warning("maxIter reached.")
    if outputIter:
        return c, Fnew, i
    else:
        return c, Fnew
        
    
# run as a main file if called. 
if __name__ == '__main__':
    main()
