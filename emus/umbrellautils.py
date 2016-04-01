# -*- coding: utf-8 -*-
"""
Container for the primary EMUS routines
"""
import numpy as np

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
        psis : The values of the bias functions at each point in space
            
    """
    L = len(centers)
    if not isinstance(kTs,list):
        kTs = np.ones(L)*kTs
    forceprefacs = -0.5*np.array([fks[i]/kTs[i] for i in xrange(L)])
    psis = [_get_psi_vals(coord,centers,forceprefacs,period) for coord in xtraj]
    return psis

def _get_psi_vals( coord,centers,forceprefacs,period=360):
    rv = np.array(coord)-np.array(centers)
    if period is not None:
        rvmin = minimage(rv,period)
    else:
        rvmin = rv
    try:
        return np.exp(np.sum(forceprefacs*rvmin*rvmin,axis=1))
    except:
        return np.exp(forceprefacs*rvmin*rvmin)

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

def minimage(rv,period):
    return rv - period * np.rint(rv/period)
