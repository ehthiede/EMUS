# -*- coding: utf-8 -*-
"""
Library with routines associated with the asymptotic variance for iterative EMUS.
"""

from __future__ import absolute_import
import numpy as np
from . import avar
from . import autocorrelation as ac
from ._defaults import DEFAULT_IAT


def calc_a(psis, z):
    """

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
    a : list of 2d arrays
         a[i][t,j] gives the value of
        :math:`a_ij(X_t^i)`.

    Variables used
    --------------
    L: number of windows.
    N: 1D array
        N[i]: number of points collected in window i.
    Nt: total number of points.
    """
    L = len(psis)
    N = np.zeros(L)
    for i in range(L):
        N[i] = psis[i].shape[0]
    a = []
    for i in range(L):
        ai = np.zeros((int(N[i]), L))
        for t in range(int(N[i])):
            for j in range(L):
                ai[t, j] = (psis[i][t, j]/z[i])/ np.sum(psis[i][t]/z)
        a.append(ai)
    return a


def calc_B_matrix(psis, z):
    L = len(psis)
    N = np.zeros(L)
    windowsum=[]
    for i in range(L):
        N[i] = psis[i].shape[0]
        windowsum1=np.zeros(int(N[i]))
        for t in range(int(N[i])):
            s1=0
            for k in range(L):
                s1+=psis[i][t,k]/z[k]
            windowsum1[t]=s1
        windowsum.append(windowsum1)
    B = np.zeros((L, L))
    for r in range(L):
        for i in range(L):
            s1 = 0
            for t in range(int(N[i])):
                s1 += psis[i][t, r]*psis[i][t,r]/(windowsum[i][t])**2
            B[r,r] -= s1/(z[r]**2*N[i])
        B[r, r] += 1
    for r in range(L):
        for s in range(r):
            for i in range(L):
                s1 = 0
                for t in range(int(N[i])):
                    s1 += psis[i][t, r]*psis[i][t, s]/(windowsum[i][t])**2
                B[r,s] -= s1/(z[r]**2*N[i])
                B[s,r]-= s1/(z[s]**2*N[i])
    return B


def calc_log_z(psis, z, repexchange=False, iat_method=DEFAULT_IAT):
    """
    Calculates the asymptotic variance in the log partition functions.
    """
    L = len(psis)  # Number of windows
    a = calc_a(psis, z)
    B = calc_B_matrix(psis, z)
    # Construct trajectories for autocovariance.
    B_pseudo_inv = np.linalg.pinv(B)
    zeta_traj=[]
    for i in range(L):
        Ni = int(psis[i].shape[0])
        zeta_i=np.zeros((Ni,L))
        for t in range(int(Ni)):
            for r in range(L):
                zeta_i[t,r]=z[i]*np.dot(a[i][t],B_pseudo_inv.T[r])
        zeta_traj.append(zeta_i)
    z_contribs = np.zeros((L, L))
    z_iats = np.zeros((L, L))
    for k in range(L):
        # Extract contributions to window k FE.
        zeta_k = [zt[:, k] for zt in zeta_traj]
        z_iats[k], z_contribs[k] = _get_iid_avars(
            zeta_k, iat_method)
    z_avar = np.sum(z_contribs, axis=1)
    return z_avar, z_contribs, z_iats


def _get_iid_avars(error_traj, iat_method):
    """
    Get asymptotic variance for a set of trajectories that come from IID sampling.

    Parameters
    ----------
    error_traj : 3D data structure
        Collection of trajectories.  error_traj[i] is a timeseries that depends only on
        sampling in window i.  The IAT of error_traj[i] gives the IAT of window i's
        contribution to the total asymptotic variance of the estimate.
    iat_method : string or 1D array-like, optional
        Method used to estimate autocorrelation time.  Choices are 'acor', 'ipce', and 'icce'.  Alternatively, if an array of length no. windows is provided, element i is taken to be the autocorrelation time of window i.


    Returns
    ------
    iats : 1D numpy array
        Value of the integrated autocorrelation time of each trajectory in error_traj
    acovars : 1D numpy array
        Value of the asymptotic autocovariance for each trajectory in error_traj

    """
    L = len(error_traj)  # Number of windows
    iat_routine, iats = avar._parse_iat_routine(iat_method, L)
    sigmas = np.zeros(L)
    for i, err_t_series in enumerate(error_traj):
        if iat_routine is not None:
            iat, mn, sigma = iat_routine(err_t_series)
            iats[i] = iat
        else:
            iat = iats[i]
            sigma = np.std(err_t_series) * np.sqrt(iat / len(err_t_series))
        sigmas[i] = sigma
    return iats, sigmas**2


