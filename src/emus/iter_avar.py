# -*- coding: utf-8 -*-
"""
Library with routines associated with the asymptotic variance for iterative EMUS.
"""

from __future__ import absolute_import
import numpy as np
from . import avar
from . import autocorrelation as ac
from ._defaults import DEFAULT_IAT


def calc_p_traj(psis, z):
    """
    Estimates the trajectories :math:`p_j(X_t^i)` described in REF.

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
    p_traj : list of 2d arrays
        Values of the P trajectory. p_traj[i][t,j] gives the value of
        :math:`p_j(X_t^i)`.

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
    Nt = np.sum(N)
    p_traj = []
    for i in range(L):
        pi = np.zeros((int(N[i]), L))
        for t in range(int(N[i])):
            for j in range(L):
                pi[t, j] = (N[j]/Nt*psis[i][t, j]/z[j]) / \
                    np.dot(N/Nt, psis[i][t]/z)
        p_traj.append(pi)
    return p_traj


def calc_B_matrix(psis, p_traj):
    """
    Estimates the B matrix in REF.

    Parameters
    ----------
    p_traj : list of 2d arrays
        Values of the P trajectory. p_traj[i][t,j] gives the value of
        :math:`p_j(X_t^i)`.

    Returns
    -------
    B : 2d matrix
        B matrix in REF.
    """
    L = len(psis)
    N = np.zeros(L)
    for i in range(L):
        N[i] = psis[i].shape[0]
    Nt = np.sum(N)
    B = np.zeros((L, L))
    for r in range(L):
        for i in range(L):
            s1 = 0
            for t in range(int(N[i])):
                s1 += p_traj[i][t, r]*(1-p_traj[i][t, r])
            B[r, r] += s1
    for r in range(L):
        for s in range(r):
            for i in range(L):
                s1 = 0
                for t in range(int(N[i])):
                    s1 += p_traj[i][t, s]*p_traj[i][t, r]
                B[r, s] -= s1
            B[s, r] = B[r, s]
    B = B/Nt
    return B


def calc_log_z(psis, z, repexchange=False, iat_method=DEFAULT_IAT):
    """
    Calculates the asymptotic variance in the log partition functions.
    """
    L = len(psis)  # Number of windows
    p_traj = calc_p_traj(psis, z)
    B = calc_B_matrix(psis, p_traj)

    # Construct trajectories for autocovariance.
    B_pseudo_inv = np.linalg.pinv(B)
    zeta_traj = [np.dot(p_traj_i, B_pseudo_inv.T) for p_traj_i in p_traj]
    if repexchange:
        zeta_sum = np.sum(zeta_traj, axis=0)
        log_z_iats = np.zeros(L)
        log_z_avar = np.zeros(L)
        for k in range(L):
            log_z_iats[k], log_z_avar[k] = _get_repexchange_avars(
                zeta_sum[:, k], iat_method)
        return log_z_avar, log_z_avar, log_z_iats
    else:
        log_z_contribs = np.zeros((L, L))
        log_z_iats = np.zeros((L, L))
        for k in range(L):
            # Extract contributions to window k FE.
            zeta_k = [zt[:, k] for zt in zeta_traj]
            log_z_iats[k], log_z_contribs[k] = _get_iid_avars(
                zeta_k, iat_method)
            # print(zeta_k)
            # print(np.shape(zeta_k))
            # print(np.shape(zeta_traj))
            # print(k, np.std(zeta_k, axis=1), log_z_contribs[k])
            # raise Exception
        log_z_avar = np.sum(log_z_contribs, axis=1)
        return log_z_avar, log_z_contribs, log_z_iats


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


def _get_repexchange_avars(error_traj, iat_method):
    """
    Get asymptotic variance for a set of trajectories that come from Replica Exchange.

    Parameters
    ----------
    error_traj : 3D data structure
        Collection of trajectories.  error_traj[i] is a timeseries that depends only on
        sampling in window i.  In contrast to the IID case, as sampling is correlated
        across windows, here each trajectory must be the same length.
    iat_method : string or 1D array-like, optional
        Method used to estimate autocorrelation time.  Choices are 'acor', 'ipce', and 'icce'.  Alternatively, if a scalar is provided, the value of the scalar is taken to be the autocorrelation time of the sampling process.


    Returns
    ------
    iat : float
        Value of the integrated autocorrelation time for the dataset
    acovar : float
        Value of the asymptotic autocovariance for the data

    """
    if isinstance(iat_method, str):
        iat_routine = ac._get_iat_method(iat_method)
        iat, mn, sigma = iat_routine(error_traj)
        acovar = sigma**2
    else:
        try:  # Try to interpret iat_method as a single number.
            iat = float(iat_method)
        except (ValueError, TypeError) as err:
            err.message = "Was unable to interpret the input provided as a method to calculate the autocorrelation time or as a sequence of autocorrelation times.  Original error message follows: " + err.message
            raise err
        acovar = np.var(error_traj) * iat / len(error_traj)
    return iat, acovar
