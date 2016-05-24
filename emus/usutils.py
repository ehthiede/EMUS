# -*- coding: utf-8 -*-
"""Module containing methods useful for analyzing umbrella sampling 
calculations that do not rely directly on the EMUS estimator.

"""
import numpy as np

def neighbors_harmonic(centers,fks,kTs=1.,period=None,nsig=4):
    """Calculates neighborlist for harmonic windows.  Neighbors are chosen 
    such that neighboring umbrellas are no more than nsig standard
    deviations away on a flat potential.

    Parameters
    ----------
    centers : 2darray
        The locations of the centers of each window.  The
        first dimension is the window index, and the second
        is the collective variable index.
    fks : 2darray or scalar
        If array or list, data structure where the first dimension 
        corresponds to the window index and the second corresponds to the
        collective variable.  If scalar, windows are assumed to have that 
        force constant in every dimension.
    kTs : 2darray or float
        1D array with the Boltzmann factor or
        a single value which will be used in all windows.  Default
        value is the scalar 1.
    period : 1D array-like or float
        Period of the collective variable
        e.g. 360 for an angle. If None, all collective variables are 
        taken to be aperiodic.  If scalar, assumed to be period of each 
        collective variable. If 1D iterable with each value a scalar or 
        None, each cv has periodicity of that size.
    nsig : scalar
        Number of standard deviations of the gaussians to 
        include in the neighborlist.

    Returns
    -------
    nbrs : 2d list
        List where element i is a list with the indices of all 
        windows neighboring window i.
    
    FIX THIS!!!!!!!!!
    """
    L = len(centers) # Number of Windows

    # Enforce Typing
    if not hasattr(kTs,'__getitem__'): # Check if kTs is a scalar
        kTs = kTs*np.ones(L)
    if not hasattr(fks,'__getitem__'): # Check if force constant is a scalar
        fks = fks*np.ones(np.shape(centers))
    kTs = np.outer(kTs,np.ones(np.shape(fks[0])))
    rad = nsig*np.sqrt(kTs/fks) 
    if period is not None: 
        if not hasattr(period,'__getitem__'): # Check if period is scalar
            period = [period]
    
    # Iterate through window centers and find neighboring umbrellas.
    nbrs = []
    for i,cntr_i in enumerate(centers):
        rad_i = rad[i]
#        print rad_i
        nbrs_i = []
        for j, cntr_j in enumerate(centers):
            rv = cntr_j-cntr_i
            if period is not None:
                for compi, component in enumerate(rv):
                    if (period[compi] is not 0.0) or (period[compi] is not None):
                        rv[compi] = _minimage(component,period[compi])
#            print cntr_j, cntr_i, rv
            if (np.abs(rv) < rad_i).all():
                nbrs_i.append(j)
        nbrs.append(nbrs_i)
    return nbrs

def unpackNbrs(compd_array,neighbors,L):
    """Unpacks an array of neighborlisted data.  Currently, assumes axis 0
    is the compressed axis.
    
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

def data_from_WHAMmeta(filepath,dim,T=None,k_B=1.9872041E-3,period=None):
    """Reads data saved on disk according to the format used by the WHAM implementation by Grossfield.

    Parameters
    ----------
    filepath : string
        The path to the meta file.
    dim : int
        The number of dimensions of the cv space.
    T : scalar, optional
        Temperature of the system.
    k_B : scalar, optional
        Boltzmann Constant for the system. Default is in kCal/mol
    period : 1D array-like or float, optional
        Variable with the periodicity information of the system.  See the Data Structures section of the documentation for a detailed explanation.

    Returns
    -------
    psis : 2D array
        The values of the bias functions at each point in the trajectory evaluated at the windows given.  First axis corresponds to the timepoint, the second to the window index.
    cv_trajs : 2D array-like
        Two dimensional data structure with the trajectories in cv space.  The first dimension is the state where the data was collected, and the second is the value in cv space.

    """
    # Parse Wham Meta file.
    trajlocs, cntrs, fks, iats, temps  = parse_metafile(filepath,dim)
    L = len(cntrs)

    # Calculate kT for each window.  Involves some type management...
    if not temps:
        try:
            temps = np.ones(L)*T
        except:
            raise TypeError('No Temperatures were found in the meta file, and \
                no valid Temperature was provided as input.')
    kT = k_B * temps

    # Load in the trajectories into the cv space
    trajs = []
    for i, trajloc in enumerate(trajlocs):
        trajs.append(np.loadtxt(trajloc)[:,1:]) 

    # Calculate psi values
    psis = []
    for i,traj in enumerate(trajs):
        psi_i = calc_harmonic_psis(traj,cntrs,fks,kT,period=period)
        psis.append(psi_i)

    return psis, trajs

    
def calc_harmonic_psis(cv_traj, centers, fks, kTs, period = None):
    """Calculates the values of each bias function from a trajectory of points
    in a single state.

    Parameters
    ----------
    cv_traj : array-like
        Trajectory in collective variable space.  Can be 1-dimensional (one cv) or 2-dimensional (many cvs).  The first dimension is the time index, and (optional) second corresponds to the collective variable. 
    centers : array-like
        The locations of the centers of each window.  The first dimension is the window index, and the (optional) second is the collective variable index.
    fks : scalar or 2darray
        If array or list, data structure where the first dimension corresponds to the window index and the second corresponds to the collective variable.  If scalar, windows are assumed to have that force constant in every dimension.
    kTs : scalar or 2darray
        1D array with the Boltzmann factor or a single value which will be used in all windows.  Default value is the scalar 1.
    period : 1D array-like or float, optional
        Period of the collective variable e.g. 360 for an angle. If None, all collective variables are taken to be aperiodic.  If scalar, assumed to be period of each collective variable. If 1D iterable with each value a scalar or None, each cv has periodicity of that size.

    Returns
    -------
    psis : 2D array
        The values of the bias functions at each point in the trajectory evaluated at the windows given.  First axis corresponds to the timepoint, the second to the window index.
            
    """
    L = len(centers)
    if not hasattr(kTs,'__getitem__'): # Check if kTs is a scalar
        kTs = kTs*np.ones(L)
    
    if not hasattr(fks,'__getitem__'): # Check if force constant is a scalar
        fks = fks*np.ones(np.shape(centers))
        # ADJUST IF SIZE IS NOT QUITE RIGHT!!!!

    forceprefacs = -0.5*np.array([fks[i]/kTs[i] for i in xrange(L)])
    # SEE IF IT IS POSSIBLE TO SPEED THIS UP VIA NUMPY TRICKS, OR MOVE IT INTO CYTHON/C/JULIA
    psis = [_get_hpsi_vals(coord,centers,forceprefacs,period) for coord in cv_traj]
    return psis

def _get_hpsi_vals(coord,centers,forceprefacs,period=None):
    """Helper routine for calc_harm_psis.  Evaluates the value of the bias
    function for each harmonic window at the coordinate coord.

    Parameters
    ----------
    coord : 1d array
        Coordinate to evaluate the harmonics at.
    centers : array-like 
        Array of centers for the windows.
    forceprefacs : array-like 
        Force constants for the windows divided by -kT.
    period : 1D array-like or float, optional 
        Period of the collective variables.  See documentation for calc_harmonic_psis.

    Returns
    -------
    psivals : 1d array
        Value of :math:`\psi_{ij}(x)` evaluated at the center of each window provided.

    """
    rv = np.array(coord)-np.array(centers)
    if period is not None:
        rvmin = np.copy(rv)
        try:
            for i,p in enumerate(period):
                if p is not None:
                    rvmin[i] -= p*np.rint(rvmin[i]/p)  # TODO THIS IS TOTALLY BROKEN!!!! FIX IT!!!
        except:
            rvmin -= period * np.rint(rvmin/period)
    else:
        rvmin = rv
    try:
        return np.exp(np.sum(forceprefacs*rvmin*rvmin,axis=1))
    except:
        return np.exp(forceprefacs*rvmin*rvmin)

def parse_metafile(filepath,dim):
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
        for line in f:
            windowparams = line.split(' ')
            traj_paths.append(windowparams[0])
            centers.append(windowparams[1:1+dim])
            fks.append(windowparams[1+dim:1+2*dim])
            if len(windowparams) > 1+2*dim: # If Correlation Time provided
                iats.append(windowparams[2+2*dim])
            if len(windowparams) > 2+2*dim: # If Temperature is provided
                temps.append(windowparams[3+2*dim])
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
