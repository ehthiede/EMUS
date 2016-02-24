# -*- coding: utf-8 -*-
"""
Helper routines for EMUS
"""
import numpy as np
import linalg_methods as lm
import acor

def neighbors_harmonic(cntrs,fks,kTs,period=None,nsig=4):
    """
    Calculates neighborlist for harmonic windows.  Neighbors are chosen 
    such that neighboring umbrellas are no more than nsig standard
    deviations away on a flat potential.

    Parameters
    ----------
        cntrs
        FINISH DOCUMENTATION
    """
    rad = nsig*np.sqrt(kTs/np.amax(fks,axis=1)) #TIGHTEN THIS UP!!
    nbrs = []
    for i,cntr_i in enumerate(cntrs):
        rad_i = rad[i]
        nbrs_i = []
        for j, cntr_j in enumerate(cntrs):
            rv = cntr_j-cntr_i
            if period is not None:
                for compi, component in enumerate(rv):
                    if (period[compi] is not 0.0) or (period[compi] is not None):
                        rv[compi] = minimage(component,period[compi])
            if np.linalg.norm(rv) < rad_i:
                nbrs_i.append(j)
        nbrs.append(nbrs_i)
    return nbrs


def unpackNbrs(compd_array,neighbors,L):
    """
    Unpacks an array of neighborlisted data.  Currently, assumes axis 0
    is the compressed axis.

    Parameters
    ----------
        compd_array : ndarray
            The compressed array, calculated using neighborlists
        neighbors : ndarray or list
            The list or array of ints, containing the indices of the neighboring windows
        L : integer
            The total number of windows.

    Returns
    -------
        expd_array: ndarray
            The expanded array of data
    """
    axis=0
    expd_shape = list(np.shape(compd_array))
    expd_shape[axis] = L
    expd_array = np.zeros(expd_shape)
    for n_ind, n_val in enumerate(neighbors):
        expd_array[n_val] = compd_array[n_ind]
    return expd_array

#def makeFrow(psi_traj):
#    """
#    Makes the i'th row of the F matrix from the trajectory of bias
#    function values of neighboring windows, evaluated from data in
#    window i.
#    
#    Parameters
#    ----------
#        psi_traj : ndarray
#            the values of the bias function for each point, calculated 
#            using data from window i.  Data format assumed to be 
#            N_i x L, where N_i is the number of data points in the i'th
#            window and L is the total number of neighboring windows.
#
#    Returns
#    -------
#        F_i : ndarray
#            i'th row in the matrix F in the Stochastic Matrix Method.
#    """
#    q_i = np.array(psi_traj) # Typecast to numpy to use advanced indexing
#    L = len(q_i[0]) # Number of neighboring states
#    F_i = np.zeros(L)
#    q_i_sum = np.sum(q_i,axis=1)
#    # Calculate the averages
#    for j in xrange(L):
#        qijtraj = q_i[:,j]/q_i_sum
#        F_i[j] = np.average(qijtraj)
#    return F_i


#def EMUS_weights(qs,neighbors=None):
#    """
#    Calculates the normalization constants according to the EMUS method.
#
#    Parameters
#    ----------
#    qs : list 
#    the trajectories of bias functions from each window.  Each 
#    element in the list is a list or array of size N_i x L,
#    where N_i is the number of points in the trajectory of that
#    window, and L is the number of neighboring windows
#
#    neighbors : list
#    A list containing information about the neighboring windows.
#    If none, assumes qs is evaluated for every window.  Else,
#    each elemint is a list with the indexes of the neighboring
#    windows.  Ordering in qs must be the same as in that list.
#    """
#    # We first make the matrix F
#    L = len(qs)
#    F = np.zeros((L,L))
#    for i in xrange(L):
#        q_i = qs[i]
#        F_i_compd = makeFrow(q_i)
#        if neighbors is None:
#            F[i] = F_i_compd
#        else:
#            neighb_i = neighbors[i]
#            F[i] = unpackNbrs(F_i_compd,neighb_i,L)
#    z = lm.stationary_distrib(F)
#    return z,F


def avar_zfe(qdata,neighbors,um1,um2,returntrajs=False,taumethod='acor'):
    """
    Estimates the asymptotic variance in the free energy difference 
    between windows um2 and um1, -k_B T log(z_2/z_1). In the code, we
    arbitrarily denote um2 as 'k' and um1 as 'l' for readability.

    REQUIRES ACOR TO BE INSTALLED!

    # REWRITE TO INCLUDE TO COMPONENTWISE?

    Parameters
    ----------
    qdata : ndarray or list of ndarrays
    The value of the bias on the probability density for each window.
    Expected 3d object, with indices inj, where i is the state the 
    data point comes frome, n is the number point it is, and j is the
    value of the j'th probability bias evaluated at that point.
    neighbors : ndarray or list
    The list or array of ints, containing the indices of the 
    neighboring windows
    um1 : the index of the first window we are interested in.
    um2 : the index of the second window we are interested in.
    returntrajs : Whether or not to return an array containing the
    trajectories computed that quantify the error.


    Returns
    -------
    errvals : ndarray
    Array of length L (no. windows) where the i'th value corresponds
    to the contribution to the error from window i.
    """
    if taumethod=='acor':
        from acor import acor
        tairoutine = acor
    elif taumethod == 'ipce':
        from autocorrelation import ipce
        tairoutine = ipce
    L = len(qdata)
    errvals = np.zeros(L)
    tauvals = np.zeros(L)
    trajs = []

    # First we calculate the values of the normalization constants.
    z, F = EMUS_weights(qdata,neighbors)
    # Calculate dAdFij
    groupInv = lm.groupInverse(np.eye(L)-F)
    partial_pt1 = (1./z[um2])*groupInv[:,um2] - (1./z[um1])*groupInv[:,um1]
    dAdFij = np.outer(z,partial_pt1) # Partial of FE difference w.r.t. Fij
    # We now create the trajectory for Fij and calculate the error.
#    normedqdata = np.zeros(qdata.shape)
    for i, q_i in enumerate(qdata):
        q_xi = np.array(q_i)
        Lneighb = len(neighbors[i]) # Number of neighbors
        qsum = np.sum(q_xi,axis=1)
        normedqdata = np.zeros(q_xi.shape)
        for j in xrange(Lneighb):
            normedqdata[:,j] = q_xi[:,j]/qsum
        err_tseries = np.dot(normedqdata,dAdFij[i][neighbors[i]]) #Error time series
        # Calculate the error
        tau, mean, sigma = tairoutine(err_tseries)
        errvals[i] = sigma
        tauvals[i] = tau
        if returntrajs:
            trajs.append(err_tseries)

    if returntrajs:
        return errvals, tauvals, trajs
    else:
        return errvals, tauvals

def makeFEsurface(xtraj, qtraj, domain, zvals, nbins = 100,kT=0.616033):
    """
    Calculates the free energy surface for an umbrella sampling run.

    Parameters:
    ----------

    ....finish....

    THIS CODE NEEDS FINISHING AND DEBUGGING!
    """    
    if domain is None:
        raise NotImplementedError

    domain = np.asarray(domain)
    if len(np.shape(domain)) == 1:
        domain = np.reshape(domain,(1,len(domain)))
    print domain
    ndims = np.shape(domain)[0]
    if type(nbins) is int: # Make nbins to an iterable in the 1d case.
        nbins = [nbins]*ndims
    domainwdth = domain[:,1] - domain[:,0]
    print domainwdth

    hist = np.zeros(nbins)
    for i,xtraj_i in enumerate(xtraj):
        xtraj_i = (xtraj_i - domain[:,0])%domainwdth + domain[:,0]
#        print "Histogramming umbrella %d" % i
        hist_i = np.zeros(nbins) # Histogram of umbrella i
        for n,coord in enumerate(xtraj_i):
            qs = qtraj[i][n]
            # We find the coordinate of the bin we land in.
            coordbins = (coord - domain[:,0])/domainwdth*nbins
            coordbins = tuple(coordbins.astype(int))
#            print coordbins, 'cbins'
            weight = 1./np.sum(qs)
            hist_i[coordbins] += weight
#            print hist_i[coordbins]
        hist+=hist_i/len(xtraj_i)*zvals[i]
#        print hist
    pmf = -kT * np.log(hist)
    pmf -= min(pmf.flatten())
    return pmf

def getAllocations(importances,N_is,newWork):
    """
    Calculates the weights for resampling of the windows.  
    These are (possibly) the optimal weights for 
    To deal with negative weights, it removes all the negative weights, and calculates the weights for the resulting subproblem.
    """
    errs = np.copy(importances)
    ns = np.copy(N_is)
    testWeights = calcWeightSubproblem(errs,ns,newWork)
    negativity = np.array([ weit < 0.0 for weit in testWeights])
#    print negativity
    while(any(negativity) == True):
        errs*=( 1.0-negativity)
        ns*=(1.0-negativity)
        newWeights = calcWeightSubproblem(errs,ns,newWork)
#        print min(testWeights*newWeights)
        testWeights = newWeights
        negativity = np.array([ weit < 0.0 for weit in testWeights])
    # We return the weights, rounded and then converted to integers
    return map(int, map( round, testWeights ))


def calcWeightSubproblem(importances,N_is,newWork):
    """
    Calculates the sampling weights of each region, according to the method using Lagrange Modifiers.
    """
    totalWork = np.sum(N_is)
    weights = np.zeros(importances.shape)
    varConstants = importances*np.sqrt(N_is)
    #print varConstants
    constSum = np.sum(varConstants)
    for ind, val in enumerate(varConstants):
        weights[ind] = val/constSum*(newWork+totalWork)-N_is[ind]
    return weights

def calc_harmonic_psis(xtraj, centers, fks, kTs ,period = None):
    """
    Calculates the values of each bias function from a trajectory of points.
    Currently, this assumes a 1D  umbrella system.

    Parameters
    ----------
        xtraj : ndarray, each element is a point in the cv space
        centers : ndarray, the locations of the centers of each umbrella.
        forceconstant : float, the value of the force constant of each umbrella.  This assumes that each umbrella has the same force constant (Might want to change this later).
        kT : Float, the Boltzmann factor.
        period : Either None or a number.  If None, this indicates that the collective variable is aperiodicy.  If number, indicates the width of the periodic boundary.

    Returns
    -------
        qtraj : The values of the bias functions at each point in space
            
    """
    L = len(centers)
    qvals = np.zeros((len(xtraj),L))
    forceprefacs = -0.5*np.array([fks[i]/kTs[i] for i in xrange(L)])
    qtraj = [getqvals(coord,centers,forceprefacs,period) for coord in xtraj]
    return qtraj

def getqvals( coord,centers,forceprefacs,period=360):
    rv = np.array(coord)-np.array(centers)
    if period is not None:
        rvmin = minimage(rv,period)
    else:
        rvmin = rv
    return np.exp(np.sum(forceprefacs*(rvmin)**2,axis=1))

def emus_iter(qtraj, Avals=None, neighbors=None, return_taus = False):
    """
    UPDATE THE DOCUMENTATION!!!!

    Returns the normalization constants of the umbrellas, calculated according
    to the power law estimator method with power 1. (see Meng and Wong).
    
    Parameters
    ----------
    
    q_traj : A list of two dimensional arrays, with each list representing data
    from one source.  Each row of one of these arrays represents a time point,
    and each column represents the value of that ponit in another unnormalized
    probability density.
    
    Avals : An inverse coefficient to scale data from each umbrella with.  For 
    MBAR, this is N_i/(z_i \tau_i), where c is the normalization constant for 
    each umbrella and tau is an estimate of the autocorrelation time of the
    umbrella.
    
    use_Nj : Whether to scale parts of the power law estimator by the number of
    samples.  Takes as options True or False (estimate the
    autocorrelation time of the trajectory using acor)
        
    Returns
    -------
    
    z : array of the partition functions for each state.
    
    F : the stochastic matrix for the eigenproblem.
    """
    
    # Initialize variables
    L = len(qtraj) # Number of Windows
    F = np.zeros((L,L)) # Initialize F Matrix
    if return_taus:
        taumat = np.ones((L,L))
    
    if Avals is None:
        Avals = np.ones((L,L))
    
    if neighbors is None:
        neighbors = np.outer(np.ones(L),range(L)).astype(int)
        
    # Calculate Fi: for each i
    for i in xrange(L):
        Avi = Avals[i]
        nbrs_i = neighbors[i]
        qi_data = qtraj[i]
#        print nbrs_i
#        print Avi
        A_nbs = Avi[nbrs_i]
#        print A_nbs, "A_nbs"
#        print np.shape(A_nbs),np.shape(qi_data)
        denom = np.dot(qi_data,A_nbs)
#        print np.max(denom), np.shape(denom), "DENOM"
        for j_index, j in enumerate(nbrs_i):
            Ftraj = qi_data[:,j_index]/denom
            Fijunnorm = np.average(Ftraj)
            F[i,j] = Fijunnorm*Avi[i]
#            print "i j max(Ftraj) avg(Ftraj)"
#            print i,j,np.max(Ftraj),np.average(Ftraj)
            if return_taus:
                tau = acor.acor(Ftraj)[0]
                if not np.isnan(tau):
                    taumat[i,j] = acor.acor(Ftraj)[0]
#    print "F"
#    for row in F:
#        print row
#    print "F"
#    print np.sum(F,axis=1), "FSUM"
    z = lm.stationary_distrib(F)
    if return_taus:
        return z, F, taumat
    else:
        return z, F

def minimage(rv,period):
    return rv - period * np.rint(rv/period)
