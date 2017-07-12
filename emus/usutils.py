# -*- coding: utf-8 -*-
"""Module containing methods useful for analyzing umbrella sampling 
calculations that do not rely directly on the EMUS estimator.

"""
import numpy as np
from _defaults import *
import numbers


def neighbors_harmonic(centers,fks,kTs=DEFAULT_KT,period=None,nsig=6):
    """Calculates neighborlist for harmonic windows.  Neighbors are chosen 
    such that neighboring umbrellas are no more than nsig standard
    deviations away on a flat potential.

    Parameters
    ----------
    centers : 2D array-like
        The locations of the centers of each window.  The first dimension is the window index, and the second is the collective variable index.
    fks : 2D array-like or scalar
        If array or list, data structure where the first dimension corresponds to the window index and the second corresponds to the collective variable.  If scalar, windows are assumed to have that force constant in every dimension.
    kTs : 2D array-like or scalar
        1D array with the Boltzmann factor or a single value which will be used in all windows.  Default value is the scalar 1.
    period : 1D array-like or float
        Period of the collective variable e.g. 360 for an angle. If None, all collective variables are taken to be aperiodic.  If scalar, assumed to be period of each collective variable. If 1D iterable with each value a scalar or None, each cv has periodicity of that size.
    nsig : scalar
        Number of standard deviations of the gaussians to include in the neighborlist.

    Returns
    -------
    nbrs : 2D list
        List where element i is a list with the indices of all windows neighboring window i.
    
    """
    L = len(centers) # Number of Windows

    # Enforce Typing
    if isinstance(kTs,numbers.Number):
        kTs = kTs*np.ones(L)
    if isinstance(fks,numbers.Number):
        fks = fks*np.ones(np.shape(centers))
    kTs = np.outer(kTs,np.ones(np.shape(fks[0])))
    rad = nsig*np.sqrt(kTs/fks) 
    if period is not None: 
        if isinstance(period,numbers.Number): # Check if period is scalar
            period = [period]
    
    # Iterate through window centers and find neighboring umbrellas.
    nbrs = []
    for i,cntr_i in enumerate(centers):
        rad_i = rad[i]
        nbrs_i = []
        rv = centers - cntr_i
        rvmin = _minimage_traj(rv,period) 
        for j, rv in enumerate(rvmin):
            if (np.abs(rv) < rad_i).all():
                nbrs_i.append(j)
        nbrs.append(nbrs_i)
    return nbrs

def unpack_nbrs(compd_array,neighbors,L):
    """Unpacks an array of neighborlisted data.  Currently, assumes axis 0 is the compressed axis.
    
    Parameters
    ----------
    compd_array : array-like
        The compressed array, calculated using neighborlists
    neighbors : array-like
        The list or array of ints, containing the indices of the neighboring windows
    L : int
        The total number of windows.

    Returns
    -------
    expd_array : array-like
        The expanded array of data

    """
    axis=0
    expd_shape = list(np.shape(compd_array))
    expd_shape[axis] = L
    expd_array = np.zeros(expd_shape)
    for n_ind, n_val in enumerate(neighbors):
        expd_array[n_val] = compd_array[n_ind]
    return expd_array


    
def calc_harmonic_psis(cv_traj, centers, fks, kTs, period = None):
    """Calculates the values of each bias function from a trajectory of points in a single window.

    Parameters
    ----------
    cv_traj : array-like
        Trajectory in collective variable space.  Can be 1-dimensional (one cv) or 2-dimensional (many cvs).  The first dimension is the time index, and (optional) second corresponds to the collective variable. 
    centers : array-like
        The locations of the centers of each window.  The first dimension is the window index, and the (optional) second is the collective variable index.
    fks : 2D array-like or scalar
        If array or list, data structure where the first dimension corresponds to the window index and the second corresponds to the collective variable.  If scalar, windows are assumed to have that force constant in every dimension.
    kTs : 2D array-like or scalar
        1D array with the Boltzmann factor or a single value which will be used in all windows.  Default value is the scalar 1.
    period : 1D array-like or float, optional
        Period of the collective variable e.g. 360 for an angle. If None, all collective variables are taken to be aperiodic.  If scalar, assumed to be period of each collective variable. If 1D iterable with each value a scalar or None, each cv has periodicity of that size.

    Returns
    -------
    psis : 2D array
        The values of the bias functions evaluated each window and timepoint.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
            
    """
    L = len(centers) # Number of windows
    if isinstance(kTs,numbers.Number):
        kTs = kTs*np.ones(L)
    if isinstance(fks,numbers.Number):
        fks = fks*np.ones(np.shape(centers))

    psis = np.zeros((len(cv_traj),L))
    for j in xrange(L):
        psis[:,j] = _calc_harmonic_psi_ij(cv_traj,centers[j],fks[j],kTs[j],period=period)
    return psis

def _calc_harmonic_psi_ij(cv_traj,win_center,win_fk,kT=1.0,period=None):
    """Helper routine for calc_harm_psis.  Evaluates the value of the bias
    function for a single harmonic window over a trajectory.

    Parameters
    ----------
    cv_traj : array-like
        Trajectory in collective variable space.  Can be 1-dimensional (one cv) or 2-dimensional (many cvs).  The first dimension is the time index, and (optional) second corresponds to the collective variable. 
        trajectory
    win_center : array-like or scalar
        Array of the centers of the window.
    win_fk : array-like or scalar 
        Force constants for the windows divided by -kT.
    period : 1D array-like or float, optional 
        Period of the collective variables.  See documentation for calc_harmonic_psis.

    Returns
    -------
    psivals : 1D array
        Value of :math:`\psi_{ij}(x)` evaluated at the center of the window for each point in the trajectory.

    """
    try:
        ndim = len(win_center)
    except TypeError:
        ndim = 1
    if period is not None:
        if not hasattr(period,'__getitem__'): # Check if period is a scalar
            period = [period]*ndim
    rv = cv_traj - win_center
    # Enforce Minimum Image Convention.
    rvmin = _minimage_traj(rv,period)

    # Calculate psi_ij
    U = rvmin*rvmin*win_fk
    if len(np.shape(U)) == 2:
        U = np.sum(U,axis=1)
    U/=2.

    return np.exp(-U/kT)

def fxn_data_from_meta(filepath):
    """Parses the meta file associated with observable data

    Parameters
    ----------
    filepath : string
        The path to the meta file containing the paths of the observable data.

    Returns
    -------
    fxndata : List of 2D arrays
        Three dimensional data structure containing observable information.  The first index corresponds to the observable being calculated, the second to the window index, and the third to the time point in the window.

    """
    fxn_paths = []
    with open(filepath,'r') as f:
        for full_line in f:
            line = full_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            fxn_paths.append(line)

    fxndata = []
    nfxns = None # Placeholder value
    for i,path in enumerate(fxn_paths):
        data_i = np.loadtxt(path)
        if i == 0:
            nfxns = int(len(data_i[0])-1)
            print nfxns
            for n in xrange(nfxns):
                fxndata.append([data_i[:,(n+1)]])
        else:
            for n in xrange(nfxns):
                fxndata[n].append(data_i[:,(n+1)])

    return fxndata

def data_from_meta(filepath,dim,T=DEFAULT_T,k_B=DEFAULT_K_B,nsig=None,period=None):
    """Reads collective variable data from as tabulated by a meta file of the same format used in Grossfield's implementation of the WHAM algorithm, and calculates the value of the biasing functions.

    Parameters
    ----------
    filepath : string
        The path to the meta file.
    dim : int
        The number of dimensions of the cv space.
    T : scalar, optional
        Temperature of the system if not provided in the meta file.
    k_B : scalar, optional
        Boltzmann Constant for the system. Default is in natural units (1.0)
    nsig : scalar or None, optional
        Number of standard deviations of the gaussians to include in the neighborlist.If None, does not use neighbor lists.
    period : 1D array-like or float, optional
        Variable with the periodicity information of the system.  See the Data Structures section of the documentation for a detailed explanation.

    Returns
    -------
    psis : List of 2D arrays
        The values of the bias functions evaluated each window and timepoint.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.

    """
    # Parse Meta file.
    trajlocs, cntrs, fks, iats, temps  = _parse_metafile(filepath,dim)
    L = len(cntrs)
    # Calculate kT for each window.  Involves some type management...
    if not temps:
        try:
            temps = np.ones(L)*T
        except:
            raise TypeError('No Temperatures were found in the meta file, and no valid Temperature was provided as input.')
    kT = k_B * temps
    if nsig is not None:
        neighbors = neighbors_harmonic(cntrs,fks,kTs=kT,period=period,nsig=nsig)
    else:
        neighbors = np.outer(np.ones(L),range(L)).astype(int)

    # Load in the trajectories into the cv space
    trajs = []
    for i, trajloc in enumerate(trajlocs):
        trajs.append(np.loadtxt(trajloc)[:,1:])

    # Calculate psi values
    psis = []
    for i,traj in enumerate(trajs):
        nbrs_i = neighbors[i]
        psi_i = calc_harmonic_psis(traj,cntrs[nbrs_i],fks,kT,period=period)
        psis.append(psi_i)

    return psis, trajs, neighbors

def _parse_metafile(filepath,dim):
    """
    Parses the meta file located at filepath. Assumes Wham-like Syntax.

    Parameters
    ----------
    filepath : string
        The path to the meta file.
    dim : int
        The number of dimensions of the cv space.

    Returns
    -------
    traj_paths : list of strings
        A list containing the paths to the trajectories for each window.
    centers : 2D array of floats
        Array with the center of each harmonic window. See calc_harm_psis for syntax.
    fks : 2D array of floats
        Array with the force constants for each harmonic window. See calc_harm_psis for syntax.
    iats : 1D array of floats or None
        Array with the integrated autocorrelation times of each window.  None if not given in 
        the meta file
    temps : 1D array of floats or None 
        Array with the temperature of each window in the umbrella sampling calculation.  If not given in the meta file, this will just be None.

    """
    traj_paths = []
    fks = []
    centers = []
    iats = []
    temps = []
    with open(filepath,'r') as f:
        for full_line in f:
            line = full_line.strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            windowparams = line.split()
            traj_paths.append(windowparams[0])
            centers.append(windowparams[1:1+dim])
            fks.append(windowparams[1+dim:1+2*dim])
            if len(windowparams) > 1+2*dim: # If Correlation Time provided
                iats.append(windowparams[1+2*dim])
            if len(windowparams) > 2+2*dim: # If Temperature is provided
                temps.append(windowparams[2+2*dim])
    # Move to numpy arrays, convert to appropriate data types
    fks = np.array(fks).astype('float')
    centers = np.array(centers).astype('float')
    iats = np.array(iats).astype('float')
    temps = np.array(temps).astype('float')
    return traj_paths,centers,fks,iats,temps

def _minimage(rv,period):
    """Calculates the minimum vector.

    Parameters
    ----------
    rv : array-like or scalar
        Minimum image vector
    period : array-like or scalar
        Periodicity in each dimension.

    Returns
    -------
    minimage : array-like or scalar
        minimum image vector.

    """
    return rv - period * np.rint(rv/period)

def _minimage_traj(rv,period):
    """Calculates the minimum trajectory

    Parameters
    ----------
    rv : 1 or 2D array-like 
        Minimum image trajectory
    period : array-like or scalar
        Periodicity in each dimension.

    Returns
    -------
    minimage : array-like
        minimum image trajectory
    """
    rvmin = np.array(np.copy(rv))
    if len(np.shape(rv)) == 1: # 1D trajectory array provided
        if period is not None:
            p = period[0]
            if (p is not None) and (p != 0):
                rvmin -= p*np.rint(rvmin/p)

    elif len(np.shape(rv)) == 2: # 2D trajectory array provided
        ndim = len(rv[0])
        if period is not None:
            for d in xrange(ndim):
                p = period[d]
                if (p is not None) and (p != 0):
                    rvmin[:,d]-= p*np.rint(rvmin[:,d]/p)
    else: # User provided something weird...
        raise ValueError("Trajectory provided has wrong dimensionality %d, "+ \
            "dimension should be 1 or 2."%len(np.shape(rv)))
    return rvmin
    
