# -*- coding: utf-8 -*-
"""
Container for the primary EMUS routines
"""
import numpy as np
import linalg_methods as lm

def calc_obs(psis,z,f1data,f2data=None):
    """
    Estimates the value of an observable or ratio of observables.
    Args:
        psis (3D data struct): Data structure containing psi values.
            See documentation in emus.py for a detailed explanation.
        z (1D array or list): Normalization constants for each state
        f1data (2D array): Data structure with timeseries of data 
            going in the numerator of the ratio.  First dimension 
            specifies index, second timepoint.

    Optional Args:
        f2data (2D array): Data structure with timeseries of data going
            in the denominator of the ratio.  If not given, we are only
            interested in <f1>.

    Returns:
        avg (float): estimate of <f1>/<f2>.

    """
    f1avg = 0
    f2avg = 0
    for i,psi_i in enumerate(psis):
        psi_xi = np.array(psi_i)
        psi_i_sum = np.sum(psi_xi,axis=1)
        f1_i = np.array(f1data[i])/psi_i_sum
        if f2data is None:
            f2_i = 1./psi_i_sum
        else:
            f2_i = np.array(f2data[i])/psi_i_sum
        f1avg_i = np.average(f1_i)
        f2avg_i = np.average(f2_i)
        f1avg += z[i]*f1avg_i
        f2avg += z[i]*f2avg_i
    return f1avg / f2avg

def makeFEsurface(cv_trajectories, psis, domain, z, nbins = 100,kT=1.):
    """
    Calculates the free energy surface for an umbrella sampling run.

    Args:
        cv_trajectories (2D data struct): Data structure containing 
            trajectories in the collective variable space.  See documentation
            in emus object for more detail.
        psis (3D data struct): Data structure containing psi values.
            See documentation in emus object for a detailed explanation.
        domain (tuple): Tuple containing the dimensions of the space over
            which to construct the pmf, e.g. (-180,180) or 
            ((0,1),(-3.14,3.14))
        z (1D array or list): Normalization constants for each state

    Optional Args:
        nbins (int or tuple): Number of bins to use.  If int, uses that many
            bins in each dimension.  If tuple, e.g. (100,20), uses 100 bins in
            the first dimension and 20 in the second.
        kT (float): Value of kT to scale the PMF by.  If not provided, set
            to 1.0
    """    
    if domain is None:
        raise NotImplementedError

    domain = np.asarray(domain)
    if len(np.shape(domain)) == 1:
        domain = np.reshape(domain,(1,len(domain)))
    ndims = np.shape(domain)[0]
    if type(nbins) is int: # Make nbins to an iterable in the 1d case.
        nbins = [nbins]*ndims
    domainwdth = domain[:,1] - domain[:,0]

    hist = np.zeros(nbins)
    for i,xtraj_i in enumerate(cv_trajectories):
        xtraj_i = (xtraj_i - domain[:,0])%domainwdth + domain[:,0]
        hist_i = np.zeros(nbins) # Histogram of umbrella i
        for n,coord in enumerate(xtraj_i):
            psi_i_n = psis[i][n]
            # We find the coordinate of the bin we land in.
            coordbins = (coord - domain[:,0])/domainwdth*nbins
            coordbins = tuple(coordbins.astype(int))
            weight = 1./np.sum(psi_i_n)
            hist_i[coordbins] += weight
        hist+=hist_i/len(xtraj_i)*z[i]
    pmf =-kT* np.log(hist)
    pmf -= min(pmf.flatten())
    return pmf

def emus_iter(psis, Avals=None, neighbors=None, return_iats = False,iat_routine='ipce'):
    """
    Performs one step of the the EMUS iteration.
    
    Args:
        psis (3D data struct): Data structure containing psi values.
            See documentation in emus.py for a detailed explanation.
    
    Optional Args:
        Avals (2D matrix): weight in front of psi in the F matrix.
        neighbors (2D array): list showing which states neighbor which.
            See neighbors_harmonic in umbrellautils. 
        return_iats (Bool): whether or not to calculate integrated
            autocorrelation times of :math:`\psi_ii^*` for each window.
        iat_routine (string): routine to use for calculating said iats.
            Accepts 'ipce', 'acor', and 'icce'.
    
    Returns:
        z (1D array): Normalization constants for each state
        F (2D array): the matrix constructed for the eigenproblem.

    Optional Return:
        iats (1D array): if return_iats chosen, returns iats estimated.
    """
    
    # Initialize variables
    L = len(psis) # Number of Windows
    F = np.zeros((L,L)) # Initialize F Matrix
    if return_iats:
        iats = np.ones((L))
        iatroutine=_get_iat_method(iat_routine)
        
    
    if Avals is None:
        Avals = np.ones((L,L))
    
    if neighbors is None:
        neighbors = np.outer(np.ones(L),range(L)).astype(int)
        
    # Calculate Fi: for each i
    for i in xrange(L):
        Avi = Avals[i]
        nbrs_i = neighbors[i]
        psi_i = np.array(psis[i])
        A_nbs = Avi[nbrs_i]
        denom = np.dot(psi_i,A_nbs)
        for j_index, j in enumerate(nbrs_i):
            Ftraj = psi_i[:,j_index]/denom
            Fijunnorm = np.average(Ftraj)
            F[i,j] = Fijunnorm*Avi[i]
            if return_iats and (i == j):
                iat = iatroutine(Ftraj)[0]
                if not np.isnan(iat):
                    iats[i] = iat
    z = lm.stationary_distrib(F)
    if return_iats:
        return z, F, iats
    else:
        return z, F
		
def _get_iat_method(iatmethod):
    if iatmethod=='acor':
        from acor import acor
        iatroutine = acor
    elif iatmethod == 'ipce':
        from autocorrelation import ipce
        iatroutine = ipce
    elif iatmethod == 'icce':
        from autocorrelation import icce
        iatroutine = icce
    return iatroutine


