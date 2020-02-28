
# -*- coding: utf-8 -*-
""" Library with routines associated with the asymptotic variance of the first EMUS iteration.  These estimates rely on estimates of the autocorrelation time of observables.  Multiple methods for estimating autocorrelation times are supported, these include the initial positive correlation estimator ('ipce') and the initial convex correlation estimator ('icce') by Geyer, and the acor algorithm ('acor') by Jonathan Goodman.  See the documentation to the `autocorrelation module <autocorrelation.html>`__ for more details.
"""

from __future__ import absolute_import
import numpy as np
from . import linalg as lm
from . import emus
from . import autocorrelation as ac
from ._defaults import DEFAULT_IAT


def buildF(psis, v):
    L = len(psis)  # Number of windows
    # Npnts = np.array([len(psis_i) for psis_i in psis]).astype('float')
    # Apart = Npnts/np.max(Npnts)
    F = []
    z = v[:L]
    # print(Avals)
    for i in range(L):
        psis_i = np.array(psis[i])
        denom = np.sum(np.array([psis_i[:, j]/z[j] for j in range(L)]), axis=0)
        Fi = np.zeros(len(v))
        for j in range(len(v)):
            Ftraj = psis_i[:, j]/v[j] / denom  # traj \psi_j/{\sum_k \psi_k A_k}
            Fi[j] = np.average(Ftraj)
        F.append(Fi)
    return F


def calc_avg_avar(psis, z, g1data, g2data=None, neighbors=None, iat_method=DEFAULT_IAT, combine='ratio'):
    """Estimates the asymptotic variance of averages of the ratio of two functions.
    ----------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    z : 1D array
        Array containing the normalization constants
    F : 2D array
        Overlap matrix for the first EMUS iteration.
    neighbors : 2D array, optional
        List showing which windows neighbor which.  See neighbors_harmonic in usutils for explanation.
    iat_method : string or 1D array-like, optional
        Method used to estimate autocorrelation time.  See the documentation above.
    Returns
    -------
    autocovars : ndarray
        Array of length L (no. windows) where the i'th value corresponds to the autocovariance estimate for :math:`z_i`
    z_var_contribs : ndarray
        Two dimensional array, where element i,j corresponds to window j's contribution to the autocovariance of window i.
    z_var_iats : ndarray
        Two dimensional array, where element i,j corresponds to the autocorrelation time associated with window j's contribution to the autocovariance of window i.
    """
    L = len(z)
    z_var_contribs = np.zeros((L+2, L))
    z_var_iats = np.zeros((L+2, L))
    if isinstance(iat_method, str):
        iat_routine = ac._get_iat_method(iat_method)
    else:  # Try to interpret iat_method as a collection of numbers
        try:
            iats = np.array([float(v) for v in iat_method])
        except (ValueError, TypeError) as err:
            err.message = "Was unable to interpret the input provided as a method to calculate the autocorrelation time or as a sequence of autocorrelation times.  Original error message follows: " + err.message
        iat_routine = None
        if len(iats) != L:
            raise ValueError('IAT Input was interpreted to be a collection of precomputed autocorrelation times.  However, the number of autocorrelation times found (%d) is not equal to the number of states (%d).' % (len(iats), L))
    if neighbors is None:  # If no neighborlist, assume all windows neighbor
        neighbors = np.outer(np.ones(L), range(L)).astype(int)
    if g2data is None:
        g2data = [np.ones(np.shape(g1data_i)) for g1data_i in g1data]

    g1star = emus._calculate_win_avgs(
        psis, z, g1data, neighbors, use_iter=True)
    g2star = emus._calculate_win_avgs(
        psis, z, g2data, neighbors, use_iter=True)
    g1 = np.dot(g1star, z)
    g2 = np.dot(g2star, z)

    v = np.append(z, [g1, g2])
    gs = np.stack((np.array(g1data), np.array(g2data)), axis=-1)
    psis = [np.hstack((psi_i, g_i)) for (psi_i, g_i) in zip(psis, gs)]
    F = buildF(psis, v)
    F = np.array(F)
    Bmat = np.dot(np.diag(1./v), np.eye(L+2, L)-np.transpose(F))
    Bmat = np.hstack((Bmat, np.zeros((Bmat.shape[0], 2))))
    # Bmat[L, L] = -1/g1
    # Bmat[L+1, L+1] = -1/g2
    Bmat[L, L] = 1/g1
    Bmat[L+1, L+1] = 1/g2
    dzdFij = lm.groupInverse(Bmat)
    # Iterate over windows, getting err contribution from sampling in each
    for i, psi_i in enumerate(psis):
        # Data cleaning
        psi_i_arr = np.array(psi_i)
        Lneighb = len(neighbors[i])  # Number of neighbors
        # Normalize psi_j(x_i^t) for all j
        psi_sum = np.sum(np.array([psi_i_arr[:, j]/z[j] for j in range(Lneighb)]), axis=0)
        normedpsis = np.zeros(psi_i_arr.shape)  # psi_j / sum_k psi_k
        for j in range(L+2):
            normedpsis[:, j] = psi_i_arr[:, j]/v[j] / psi_sum
        # Calculate contribution to as. err. for each z_k
        for k in range(L+2):
            err_t_series = np.dot(normedpsis, dzdFij[:, k])
            if iat_routine is not None:
                iat, mn, sigma = iat_routine(err_t_series)
                z_var_contribs[k, i] = sigma * sigma
            else:
                iat = iats[i]
                z_var_contribs[k, i] = np.var(
                    err_t_series) * (iat / len(err_t_series))
            z_var_iats[k, i] = iat
    autocovars = np.sum(z_var_contribs, axis=1)
    autocovar_g1 = autocovars[-2]
    autocovar_g2 = autocovars[-1]

    if combine == 'ratio':
        output_avar = autocovar_g1 / g2 - (g1 / g2**2) * autocovar_g2
    elif combine == 'log_ratio':
        output_avar = autocovar_g1 / g1 - autocovar_g2 / g2
    else:
        raise ValueError('combination not recognized')

    return autocovars, z_var_contribs, z_var_iats, output_avar
