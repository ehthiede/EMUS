# -*- coding: utf-8 -*-
"""
Library with routines associated with the asymptotic variance of the
first EMUS iteration.  These estimates rely on estimates of 

"""
# THIS CODE NEEDS CLEANING!

import numpy as np
from emusroutines import _get_iat_method
import linalg as lm



def avar_obs_diff(psis,z,F,f1data,g1data,f2data=None,g2data=None,
        neighbors=None,iat_method='ipce'):
    """Estimates the asymptotic variance of the estimate of the ratio :math:`<f_1>/<f_2>-<g_1>/<g_2>` observables.  If f2data and g2data are not given, they are set to 1.

    Parameters
    ----------
    psis : 3D data structure
        Data structure containing psi values.  See documentation for a detailed explanation.
    z : 1D array
        Array containing the normalization constants
    F : 2D array
        Overlap matrix for the first EMUS iteration.
    f1data : 2D data structure
        Trajectory of observable in the numerator in the first term of the difference.  First dimension corresponds to the umbrella index and the second to the point in the trajectory.
    g1data : 2D data structure
        Trajectory of observable in the numerator in the second term of the difference.  
    f2data : 2D data structure, optional
        Trajectory of observable in the denominator in the first term of the difference.  
    g2data : 2D data structure, optional
        Trajectory of observable in the denominator in the second term of the difference.  
    neighbors : 2D array, optional
        List showing which states neighbor which.  See neighbors_harmonic in usutils for explanation.
    iat_method : string, optional
        Method used to estimate autocorrelation time.  See the documentation for the avar module.


    Returns
    -------
    errvals : ndarray
        Array of length L (no. windows) where the i'th value corresponds to the contribution to the error from window i.
    iatvals : ndarray
        Array of length L (no. windows) where the i'th value corresponds to the iat for window i.
        
    """
        
    iat_routine = _get_iat_method(iat_method)
    L = len(psis)
    if neighbors is None:
        neighbors = np.outer(np.ones(L),range(L)).astype(int)
    errvals = np.zeros(L)
    iatvals = np.zeros(L)
    trajs = []

    f1trajs = []
    f2trajs = []
    f1avgs = np.zeros(L)
    f2avgs = np.zeros(L)
    g1trajs = []
    g2trajs = []
    g1avgs = np.zeros(L)
    g2avgs = np.zeros(L)
    normedpsis = []
    # Normalize f1, f2,g1,g2, psi trajectories by \sum_k psi_k
    for i,psi_i in enumerate(psis):
        Lneighb = len(neighbors[i]) # Number of neighbors
        psi_i_arr = np.array(psi_i)
        psi_i_sum = np.sum(psi_i_arr,axis=1)
        f1_i = np.array(f1data[i])/psi_i_sum
        if f2data is None:
            f2_i = 1./psi_i_sum
        else:
            f2_i = np.array(f2data[i])/psi_i_sum
        g1_i = np.array(g1data[i])/psi_i_sum
        if g2data is None:
            g2_i = 1./psi_i_sum
        else:
            g2_i = np.array(g2data[i])/psi_i_sum

        f1trajs.append(f1_i)
        f2trajs.append(f2_i)
        f1avgs[i] = np.average(f1_i)
        f2avgs[i] = np.average(f2_i)
        g1trajs.append(g1_i)
        g2trajs.append(g2_i)
        g1avgs[i] = np.average(g1_i)
        g2avgs[i] = np.average(g2_i)
        norm_psi_i = np.zeros(np.shape(psis[i]))
        for j in xrange(Lneighb):
            norm_psi_i[:,j] = psi_i_arr[:,j]/psi_i_sum
        normedpsis.append(norm_psi_i)
    fnumer_avg = np.dot(z,f1avgs)
    fdenom_avg = np.dot(z,f2avgs)
    favg = fnumer_avg / fdenom_avg
    gnumer_avg = np.dot(z,g1avgs)
    gdenom_avg = np.dot(z,g2avgs)
    gavg = gnumer_avg / gdenom_avg
    diff_avg = favg-gavg

    # Calculate Group Inverse of I-F 
    groupInv = lm.groupInverse(np.eye(L)-F)
    for i in xrange(L):
        neighb_i = neighbors[i]
        # Calculate partials w.r.t. F_ij
        partials_Fij = np.zeros(np.shape(neighb_i))
        for j_ind, j in enumerate(neighb_i):
            fac1 = f1avgs-favg*f2avgs
            dBdFij = np.dot(groupInv[j,:],fac1)*z[i]/fdenom_avg
            fac2 = g1avgs-gavg*g2avgs
            dBdFij_2 = np.dot(groupInv[j,:],fac2)*z[i]/gdenom_avg
            partials_Fij[j_ind] = dBdFij - dBdFij_2
        # Calculate partials w.r.t. f1,f2
        partial_f1 = z[i]/fdenom_avg 
        partial_f2 =-1.*favg/fdenom_avg*z[i]
        partial_g1 = -1.*z[i]/gdenom_avg 
        partial_g2 = gavg/gdenom_avg*z[i]
        # Compute time series encoding error.
        err_tseries = np.dot(normedpsis[i],partials_Fij)
        err_tseries += f1trajs[i]*partial_f1
        err_tseries += f2trajs[i]*partial_f2
        err_tseries += g1trajs[i]*partial_g1
        err_tseries += g2trajs[i]*partial_g2

        iat, mean, sigma = iat_routine(err_tseries)
        errvals[i] = sigma
        iatvals[i] = iat
    return errvals, iatvals

def avar_obs(psis,z,F,f1data,f2data=None,neighbors=None,iat_method='ipce'):
    """Estimates the asymptotic variance in the ratio of two observables.
    If f2data is not given, it just calculates the asymptotic variance
    associated with the average of f1.

    Parameters
    ----------
    psis : 3D data structure
        Data structure containing psi values.  See documentation for a detailed explanation.
    z : 1D array
        Array containing the normalization constants
    F : 2D array
        Overlap matrix for the first EMUS iteration.
    f1data : 2D data structure
        Trajectory of observable in the numerator.  First dimension corresponds to the umbrella index and the second to the point in the trajectory.
    f2data : 2D data structure, optional
        Trajectory of observable in the denominator.  
    neighbors : 2D array, optional
        List showing which states neighbor which.  See neighbors_harmonic in usutils for explanation.
    iat_method : string, optional
        Method used to estimate autocorrelation time.  See the documentation for the avar module.

    Returns
    -------
    errvals : ndarray
        Array of length L (no. windows) where the i'th value corresponds to the contribution to the error from window i.
    iatvals : ndarray
        Array of length L (no. windows) where the i'th value corresponds to the iat for window i.
        
    """
    iat_routine = _get_iat_method(iat_method)
    L = len(psis)
    errvals = np.zeros(L)
    if neighbors is None:
        neighbors = np.outer(np.ones(L),range(L)).astype(int)
    iatvals = np.zeros(L)
    trajs = []

    f1trajs = []
    f2trajs = []
    f1avgs = np.zeros(L)
    f2avgs = np.zeros(L)
    normedpsis = []
    # Normalize f1, f2, psi trajectories by \sum_k psi_k
    for i,psi_i in enumerate(psis):
        Lneighb = len(neighbors[i]) # Number of neighbors
        psi_i_arr = np.array(psi_i)
        psi_i_sum = np.sum(psi_i_arr,axis=1)
        f1_i = np.array(f1data[i])/psi_i_sum
        if f2data is None:
            f2_i = 1./psi_i_sum
        else:
            f2_i = np.array(f2data[i])/psi_i_sum

        f1trajs.append(f1_i)
        f2trajs.append(f2_i)
        f1avgs[i] = np.average(f1_i)
        f2avgs[i] = np.average(f2_i)
        norm_psi_i = np.zeros(np.shape(psis[i]))
        for j in xrange(Lneighb):
            norm_psi_i[:,j] = psi_i_arr[:,j]/psi_i_sum
        normedpsis.append(norm_psi_i)
        numer_avg = np.dot(z,f1avgs)
        denom_avg = np.dot(z,f2avgs)
        favg = numer_avg / denom_avg

    # Calculate Group Inverse of I-F 
    groupInv = lm.groupInverse(np.eye(L)-F)
    for i in xrange(L):
        neighb_i = neighbors[i]
        # Calculate partials w.r.t. F_ij
        partials_Fij = np.zeros(np.shape(neighb_i))
        for j_ind, j in enumerate(neighb_i):
            fac1 = f1avgs-favg*f2avgs
            dBdFij = np.dot(groupInv[j,:],fac1)*z[i]/denom_avg
            partials_Fij[j_ind] = dBdFij
        # Calculate partials w.r.t. f1,f2
        partial_f1 = z[i]/denom_avg 
        partial_f2 =-1.*favg/denom_avg*z[i]
        # Compute time series encoding error.
        err_tseries = np.dot(normedpsis[i],partials_Fij)
        err_tseries += f1trajs[i]*partial_f1
        err_tseries += f2trajs[i]*partial_f2

        iat, mean, sigma = iat_routine(err_tseries)
        errvals[i] = sigma
        iatvals[i] = iat
    return errvals, iatvals

def avar_zfe(psis,z,F,um1,um2,neighbors=None,iat_method='ipce'):
    """
    Estimates the asymptotic variance in the free energy difference 
    between windows um2 and um1, -k_B T log(z_2/z_1). In the code, we
    arbitrarily denote um2 as 'k' and um1 as 'l' for readability.


    Parameters
    ----------
    psis : 3D data structure
        Data structure containing psi values.  See documentation for a detailed explanation.
    z : 1D array
        Array containing the normalization constants
    F : 2D array
        Overlap matrix for the first EMUS iteration.
    state_1 : int
        Index of the first state.
    state_2 : ind 
        Index of the second state.
    neighbors : 2D array, optional
        list showing which states neighbor which.  See neighbors_harmonic in umbrellautils for explanation.
    iat_method : string, optional
        Method used to estimate autocorrelation time.  Default is the initial positive correlation estimator ('ipce'), but also supported is the initial convex correlation estimator ('icce') and the acor algorithm ('acor')  See Geyer, Stat. Sci. 1992 and Jonathan Goodman's acor documentation for reference.

    Returns
    -------
    errvals : ndarray
        Array of length L (no. windows) where the i'th value corresponds to the contribution to the error from window i.
    iatvals : ndarray
        Array of length L (no. windows) where the i'th value corresponds to the iat for window i.

    """
    # REWRITE TO INCLUDE TO COMPONENTWISE?
    iat_routine = _get_iat_method(iat_method)
    L = len(psis)
    errvals = np.zeros(L)
    iatvals = np.zeros(L)
    trajs = []
    if neighbors is None:
        neighbors = np.outer(np.ones(L),range(L)).astype(int)

    # First we calculate the values of the normalization constants.
    groupInv = lm.groupInverse(np.eye(L)-F)
    partial_pt1 = (1./z[um2])*groupInv[:,um2] - (1./z[um1])*groupInv[:,um1]
    dAdFij = np.outer(z,partial_pt1) # Partial of FE difference w.r.t. Fij
    # We now create the trajectory for Fij and calculate the error.
    for i, psi_i in enumerate(psis):
        psi_i_arr = np.array(psi_i)
        Lneighb = len(neighbors[i]) # Number of neighbors
        psi_sum = np.sum(psi_i_arr,axis=1)
        normedpsis = np.zeros(psi_i_arr.shape)
        for j in xrange(Lneighb):
            normedpsis[:,j] = psi_i_arr[:,j]/psi_sum
        err_tseries = np.dot(normedpsis,dAdFij[i][neighbors[i]]) #Error time series
        # Calculate the error
        iat, mean, sigma = iat_routine(err_tseries)
        errvals[i] = sigma
        iatvals[i] = iat

    return errvals, iatvals


def getAllocations(importances,N_is,newWork):
    """Calculates the optimal allocation of sample points 
    These are the optimal weights for 
    To deal with negative weights, it removes all the negative weights, and calculates the weights for the resulting subproblem.

    """
    errs = np.copy(importances)
    ns = np.copy(N_is)
    testWeights = _calcWeightSubproblem(errs,ns,newWork)
    negativity = np.array([ weit < 0.0 for weit in testWeights])
#    print negativity
    while(any(negativity) == True):
        errs*=( 1.0-negativity)
        ns*=(1.0-negativity)
        newWeights = _calcWeightSubproblem(errs,ns,newWork)
#        print min(testWeights*newWeights)
        testWeights = newWeights
        negativity = np.array([ weit < 0.0 for weit in testWeights])
    # We return the weights, rounded and then converted to integers
    return map(int, map( round, testWeights ))


def _calcWeightSubproblem(importances,N_is,newWork):
    """Calculates the sampling weights of each region, according to the method using Lagrange Modifiers.

    """
    totalWork = np.sum(N_is)
    weights = np.zeros(importances.shape)
    varConstants = importances*np.sqrt(N_is)
    #print varConstants
    constSum = np.sum(varConstants)
    for ind, val in enumerate(varConstants):
        weights[ind] = val/constSum*(newWork+totalWork)-N_is[ind]
    return weights

