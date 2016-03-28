# -*- coding: utf-8 -*-
"""
Helper routines for EMUS
"""
import numpy as np
import linalg_methods as lm
# import acor

def neighbors_harmonic(cntrs,fks,kTs,period=None,nsig=4):
    """
    Calculates neighborlist for harmonic windows.  Neighbors are chosen 
    such that neighboring umbrellas are no more than nsig standard
    deviations away on a flat potential.

    Parameters
    ----------
        cntrs : ndarray
            Numpy array or iterable of shape Lxd, where L is the
            number of windows and d is the dimension of the cv space.
        fks : ndarray
            Numpy array or iterable containing the force constant for
            each harmonic window.  Shape is Lxd, as with centers.
        kTs : ndarray
            1D array with the Boltzmann factor in each window.
        period : optional iterable, scalar, or None
            Periodicity of the collective variable. If None, all
            collective variables are taken to be aperiodic.  If scalar,
            assumed to be 1D US calculation with period of scalar.
            If 1D iterable with each value a scalar or None, each
            cv has periodicity of that size.
        nsig : optional scalar
            Number of standard deviations of the gaussians to include
            in the neighborlist.

    Returns
    -------
        nbrs : 2dlist
            List where element i is a list with the indices of all 
            windows neighboring window i.

    """
    rad = nsig*np.sqrt(kTs/np.amax(fks,axis=1)) #TIGHTEN UP NBRS?
    if period is not None:
        if not hasattr(period,'__iter__'):
            period = [period]

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

def calc_obs(qdata,z,F,f1data,f2data=None):
    """
    Estimates the value of an observable.
    """
    f1avg = 0
    f2avg = 0
    for i,q_i in enumerate(qdata):
        q_xi = np.array(q_i)
        q_i_sum = np.sum(q_xi,axis=1)
        f1_i = np.array(f1data[i])/q_i_sum
        if f2data is None:
            f2_i = 1./q_i_sum
        else:
            f2_i = np.array(f2data[i])/q_i_sum
        f1avg_i = np.average(f1_i)
        f2avg_i = np.average(f2_i)
        f1avg += z[i]*f1avg_i
        f2avg += z[i]*f2avg_i
    return f1avg / f2avg

        
def avar_obs_diff(qdata,neighbors,z,F,f1data,g1data,f2data=None,g2data=None,returntrajs=False,iat='ipce'):
    """
    Estimates the asymptotic variance in the difference ratio of two observables.
    If f2data is not given, it just calculates the asymptotic variance
    associatied with the average of f1.
    """
    tauroutine = _get_tau_method(iat)
    L = len(qdata)
    errvals = np.zeros(L)
    tauvals = np.zeros(L)
    trajs = []

    f1trajs = []
    f2trajs = []
    f1avgs = np.zeros(L)
    f2avgs = np.zeros(L)
    g1trajs = []
    g2trajs = []
    g1avgs = np.zeros(L)
    g2avgs = np.zeros(L)
    normedqdata = []
    # Normalize f1, f2,g1,g2, psi trajectories by \sum_k psi_k
    for i,q_i in enumerate(qdata):
        Lneighb = len(neighbors[i]) # Number of neighbors
        q_xi = np.array(q_i)
        q_i_sum = np.sum(q_xi,axis=1)
        f1_i = np.array(f1data[i])/q_i_sum
        if f2data is None:
            f2_i = 1./q_i_sum
        else:
            f2_i = np.array(f2data[i])/q_i_sum
        g1_i = np.array(g1data[i])/q_i_sum
        if g2data is None:
            g2_i = 1./q_i_sum
        else:
            g2_i = np.array(g2data[i])/q_i_sum

        f1trajs.append(f1_i)
        f2trajs.append(f2_i)
        f1avgs[i] = np.average(f1_i)
        f2avgs[i] = np.average(f2_i)
        g1trajs.append(g1_i)
        g2trajs.append(g2_i)
        g1avgs[i] = np.average(g1_i)
        g2avgs[i] = np.average(g2_i)
        nqd_i = np.zeros(np.shape(qdata[i]))
        for j in xrange(Lneighb):
            nqd_i[:,j] = q_xi[:,j]/q_i_sum
        normedqdata.append(nqd_i)
    fnumer_avg = np.dot(z,f1avgs)
    fdenom_avg = np.dot(z,f2avgs)
    favg = fnumer_avg / fdenom_avg
    gnumer_avg = np.dot(z,g1avgs)
    gdenom_avg = np.dot(z,g2avgs)
    gavg = gnumer_avg / gdenom_avg
    diff_avg = favg-gavg
    
    # Calculate Group Inverse of I-F 
    groupInv = lm.groupInverse(np.eye(L)-F)
    if returntrajs:
        traj_collection = []
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
        err_tseries = np.dot(normedqdata[i],partials_Fij)
#        print err_tseries
        err_tseries += f1trajs[i]*partial_f1
#        print err_tseries
        err_tseries += f2trajs[i]*partial_f2
        err_tseries += g1trajs[i]*partial_g1
        err_tseries += g2trajs[i]*partial_g2
#        print err_tseries

        tau, mean, sigma = tauroutine(err_tseries)
        errvals[i] = sigma
        tauvals[i] = tau
        if returntrajs:
            traj_collection.append(err_tseries)

    if returntrajs:
        return errvals, tauvals, trajs
    else:
        return errvals, tauvals

def avar_obs(qdata,neighbors,z,F,f1data,f2data=None,returntrajs=False,iat='ipce'):
    """
    Estimates the asymptotic variance in the ratio of two observables.
    If f2data is not given, it just calculates the asymptotic variance
    associatied with the average of f1.
    """
    tauroutine = _get_tau_method(iat)
    L = len(qdata)
    errvals = np.zeros(L)
    tauvals = np.zeros(L)
    trajs = []

    f1trajs = []
    f2trajs = []
    f1avgs = np.zeros(L)
    f2avgs = np.zeros(L)
    normedqdata = []
    # Normalize f1, f2, psi trajectories by \sum_k psi_k
    for i,q_i in enumerate(qdata):
        Lneighb = len(neighbors[i]) # Number of neighbors
        q_xi = np.array(q_i)
        q_i_sum = np.sum(q_xi,axis=1)
        f1_i = np.array(f1data[i])/q_i_sum
        if f2data is None:
            f2_i = 1./q_i_sum
        else:
            f2_i = np.array(f2data[i])/q_i_sum

        f1trajs.append(f1_i)
        f2trajs.append(f2_i)
        f1avgs[i] = np.average(f1_i)
        f2avgs[i] = np.average(f2_i)
        nqd_i = np.zeros(np.shape(qdata[i]))
        for j in xrange(Lneighb):
            nqd_i[:,j] = q_xi[:,j]/q_i_sum
        normedqdata.append(nqd_i)
    numer_avg = np.dot(z,f1avgs)
    denom_avg = np.dot(z,f2avgs)
    favg = numer_avg / denom_avg
    
    # Calculate Group Inverse of I-F 
    groupInv = lm.groupInverse(np.eye(L)-F)
    if returntrajs:
        traj_collection = []
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
        err_tseries = np.dot(normedqdata[i],partials_Fij)
#        print err_tseries
        err_tseries += f1trajs[i]*partial_f1
#        print err_tseries
        err_tseries += f2trajs[i]*partial_f2
#        print err_tseries

        tau, mean, sigma = tauroutine(err_tseries)
        errvals[i] = sigma
        tauvals[i] = tau
        if returntrajs:
            traj_collection.append(err_tseries)

    if returntrajs:
        return errvals, tauvals, trajs
    else:
        return errvals, tauvals

def avar_zfe(qdata,neighbors,z,F,um1,um2,returntrajs=False,iat='ipce'):
    """
    Estimates the asymptotic variance in the free energy difference 
    between windows um2 and um1, -k_B T log(z_2/z_1). In the code, we
    arbitrarily denote um2 as 'k' and um1 as 'l' for readability.

    REQUIRES ACOR TO BE INSTALLED FOR iat='acor'

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
    um1 : int
        the index of the first window we are interested in.
    um2 : int
        the index of the second window we are interested in.
    returntrajs : optional Boolean
        Whether or not to return an array containing the
        trajectories computed that quantify the error.
    iat : optional string
        Method used to estimate autocorrelation time.  Default is 
        initial positive correlation estimator ('ipce'), but also
        supported is initial convex correlation estimator ('icce')
        and the acor algorithm ('acor')  See Geyer, Stat. Sci. 1992
        and Jonathan Goodman's acor documentation for reference.


    Returns
    -------
    errvals : ndarray
    Array of length L (no. windows) where the i'th value corresponds
    to the contribution to the error from window i.
    """
    tauroutine = _get_tau_method(iat)
    L = len(qdata)
    errvals = np.zeros(L)
    tauvals = np.zeros(L)
    trajs = []

    # First we calculate the values of the normalization constants.
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
        tau, mean, sigma = tauroutine(err_tseries)
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
        hist_i = np.zeros(nbins) # Histogram of umbrella i
        for n,coord in enumerate(xtraj_i):
            qs = qtraj[i][n]
            # We find the coordinate of the bin we land in.
            coordbins = (coord - domain[:,0])/domainwdth*nbins
            coordbins = tuple(coordbins.astype(int))
            weight = 1./np.sum(qs)
            hist_i[coordbins] += weight
        hist+=hist_i/len(xtraj_i)*zvals[i]
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
    qtraj = [_getqvals(coord,centers,forceprefacs,period) for coord in xtraj]
    return qtraj

def _getqvals( coord,centers,forceprefacs,period=360):
    rv = np.array(coord)-np.array(centers)
    if period is not None:
        rvmin = minimage(rv,period)
    else:
        rvmin = rv
    try:
        return np.exp(np.sum(forceprefacs*rvmin*rvmin,axis=1))
    except:
        return np.exp(forceprefacs*rvmin*rvmin)

def emus_iter(qtraj, Avals=None, neighbors=None, return_taus = False,iat_routine='ipce'):
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
        tauroutine=_get_tau_method(iat_routine)
        
    
    if Avals is None:
        Avals = np.ones((L,L))
    
    if neighbors is None:
        neighbors = np.outer(np.ones(L),range(L)).astype(int)
        
    # Calculate Fi: for each i
    for i in xrange(L):
        Avi = Avals[i]
        nbrs_i = neighbors[i]
        qi_data = np.array(qtraj[i])
        A_nbs = Avi[nbrs_i]
        denom = np.dot(qi_data,A_nbs)
        for j_index, j in enumerate(nbrs_i):
            Ftraj = qi_data[:,j_index]/denom
            Fijunnorm = np.average(Ftraj)
            F[i,j] = Fijunnorm*Avi[i]
            if return_taus:
                tau = tauroutine(Ftraj)[0]
                if not np.isnan(tau):
                    taumat[i,j] = tauroutine(Ftraj)[0]
    z = lm.stationary_distrib(F)
    if return_taus:
        return z, F, taumat
    else:
        return z, F

def parse_metafile(filestr,dim):
    """
    Parses the meta file located at filestr.
    Assumes Whamlike Syntax

    Parameters
    ----------
    filestr : string
        The path to the meta file.
    dim : int
        The number of dimensions in the cv space.

    Returns
    -------
    ks : array of strings
        Array, element i is the location of the cv trajectory for 
        window i.

    #### FINISH THE DOCUMENTATION!!!!!

    """
    trajlocs = []
    ks = []
    cntrs = []
    corrts = []
    temps = []
    with open(filestr,'r') as f:
        for line in f:
            windowparams = line.split(' ')
            trajlocs.append(windowparams[0])
            cntrs.append(windowparams[1:1+dim])
            ks.append(windowparams[1+dim:1+2*dim])
            if len(windowparams) > 1+2*dim: # If Correlation Time provided
                corrts.append(windowparams[2+2*dim])
            if len(windowparams) > 2+2*dim: # If Temperature is provided
                temps.append(windowparams[3+2*dim])
    # Move to numpy arrays, convert to appropriate data types
    ks = np.array(ks).astype('float')
    cntrs = np.array(cntrs).astype('float')
    corrts = np.array(corrts).astype('float')
    temps = np.array(temps).astype('float')
    return trajlocs,ks,cntrs,corrts,temps

def _get_tau_method(taumethod):
    if taumethod=='acor':
        from acor import acor
        tauroutine = acor
    elif taumethod == 'ipce':
        from autocorrelation import ipce
        tauroutine = ipce
    elif taumethod == 'icce':
        from autocorrelation import icce
        tauroutine = icce
    return tauroutine
    

def minimage(rv,period):
    return rv - period * np.rint(rv/period)
