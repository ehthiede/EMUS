# -*- coding: utf-8 -*-
"""
Container for the primary EMUS routines
"""
import numpy as np

def neighbors_harmonic(centers,fks,kTs=1.,period=None,nsig=4):
    """Calculates neighborlist for harmonic windows.  Neighbors are chosen 
    such that neighboring umbrellas are no more than nsig standard
    deviations away on a flat potential.

    Args:
        centers (2darray): The locations of the centers of each window.  The
            first dimension is the window index, and the second
            is the collective variable index.
        fks (2darray or scalar):
            If array or list, data structure where the first dimension 
            corresponds to the window index and the second corresponds to the
            collective variable.  If scalar, windows are assumed to have that 
            force constant in every dimension.
        kTs (2darray or float): 1D array with the Boltzmann factor or
            a single value which will be used in all windows.  Default
            value is the scalar 1.
        period (1D arraylike or float): Period of the collective variable
            e.g. 360 for an angle. If None, all collective variables are 
            taken to be aperiodic.  If scalar, assumed to be period of each 
            collective variable. If 1D iterable with each value a scalar or 
            None, each cv has periodicity of that size.
        nsig (scala): Number of standard deviations of the gaussians to 
            include in the neighborlist.

    Returns:
        nbrs (2d list): List where element i is a list with the indices of all 
            windows neighboring window i.

    """
    L = len(centers) # Number of Windows

    # Enforce Typing
    if not hasattr(kTs,'__getitem__'): # Check if kTs is a scalar
        kTs = kTs*np.ones(L)
    if not hasattr(fks,'__getitem__'): # Check if force constant is a scalar
        fks = fks*np.ones(np.shape(centers))

    rad = nsig*np.sqrt(kTs/np.amax(fks,axis=1)) #TIGHTEN UP Neighborlisting?
    if period is not None: 
        if not hasattr(period,'__getitem__'): # Check if period is scalar
            period = [period]
    
    # Iterate through window centers and find neighboring umbrellas.
    nbrs = []
    for i,cntr_i in enumerate(centers):
        rad_i = rad[i]
        nbrs_i = []
        for j, cntr_j in enumerate(centers):
            rv = cntr_j-cntr_i
            if period is not None:
                for compi, component in enumerate(rv):
                    if (period[compi] is not 0.0) or (period[compi] is not None):
                        rv[compi] = minimage(component,period[compi])
            if np.linalg.norm(rv) < rad_i:
                nbrs_i.append(j)
        nbrs.append(nbrs_i)

def unpackNbrs(compd_array,neighbors,L):
    """Unpacks an array of neighborlisted data.  Currently, assumes axis 0
    is the compressed axis.
    
    Args:
        compd_array (ndarray): The compressed array, calculated using neighborlists
        neighbors (ndarray): The list or array of ints, containing the indices 
            of the neighboring windows
        L (int): The total number of windows.

    Returns:
        expd_array (ndarray): The expanded array of data

    """
    axis=0
    expd_shape = list(np.shape(compd_array))
    expd_shape[axis] = L
    expd_array = np.zeros(expd_shape)
    for n_ind, n_val in enumerate(neighbors):
        expd_array[n_val] = compd_array[n_ind]
    return expd_array

def calc_harmonic_psis(xtraj, centers, fks, kTs, period = None):
    """
    Calculates the values of each bias function from a trajectory of points
    in a single state.

    Args:
        xtraj (nd array): Trajectory in collective variable space.  Can be 
            1-dimensional (one cv) or 2-dimensional (many cvs).  The first 
            dimension is the time index, and (optional) second corresponds
            to the collective variable.
        centers (ndarray): The locations of the centers of each window.  The
            first dimension is the window index, and the (optional) second
            is the collective variable index.
        fks (2darray or scalar):
            If array or list, data structure where the first dimension 
            corresponds to the window index and the second corresponds to the
            collective variable.  If scalar, windows are assumed to have that 
            force constant in every dimension.
        kTs (2darray or float): 1D array with the Boltzmann factor or
            a single value which will be used in all windows.  Default
            value is the scalar 1.

    Optional Args:
        period (1D arraylike or float): Period of the collective variable
            e.g. 360 for an angle. If None, all collective variables are 
            taken to be aperiodic.  If scalar, assumed to be period of each 
            collective variable. If 1D iterable with each value a scalar or 
            None, each cv has periodicity of that size.

    Returns:
        psis (2d array): The values of the bias functions at each point in
            the trajectory for every window in centers.
            
    """
    L = len(centers)
    if not hasattr(kTs,'__getitem__'): # Check if kTs is a scalar
        kTs = kTs*np.ones(L)

    forceprefacs = -0.5*np.array([fks[i]/kTs[i] for i in xrange(L)])
    # SEE IF IT IS POSSIBLE TO SPEED THIS UP VIA NUMPY TRICKS, OR MOVE IT INTO CYTHON/C/JULIA
    psis = [_get_hpsi_vals(coord,centers,forceprefacs,period) for coord in xtraj]
    return psis

def _get_hpsi_vals(coord,centers,forceprefacs,period=None):
    """Helper routine for calc_harm_psis.  Evaluates the value of the bias
    function for each harmonic window at the coordinate coord.

    Args:
        coord (1d array): Coordinate to evaluate the harmonics at.
        centers (ndarray): Array of centers for the windows.
        forceprefacs (ndarray): Force constants for the windows divided by -kT.

    Optional Args:
        period (1D arraylike or float): Period of the collective variables.
            See documentation for calc_harmonic_psis.

    """
    rv = np.array(coord)-np.array(centers)
    if period is not None:
        rvmin = minimage(rv,period)
    else:
        rvmin = rv
    try:
        return np.exp(np.sum(forceprefacs*rvmin*rvmin,axis=1))
    except:
        return np.exp(forceprefacs*rvmin*rvmin)

def parse_metafile(filepath,dim):
    """
    Parses the meta file located at filepath. Assumes Wham-like Syntax.

    Args:
        filepath (string): The path to the meta file.
        dim (int): The number of dimensions of the cv space.

    Returns:
        traj_paths (list of strings): A list containing the paths to the
            trajectories for each window.
        centers (2D array of floats): Array with the center of each harmonic 
            window. See calc_harm_psis for syntax.
        fks (2D array of floats): Array with the force constants for each
            harmonic window. See calc_harm_psis for syntax.
        iats (1D array of floats or None): Array with the integrated 
            autocorrelation times of each window.  None if not given in 
            the meta file
        temps (1D array of floats or None): Array with the temperature of each
            window in the umbrella sampling calculation.  If not given in the 
            meta file, this will just be None.

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
    """Calculates the minimum image of 

    """

    return rv - period * np.rint(rv/period)
