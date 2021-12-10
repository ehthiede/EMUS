# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from . import linalg as lm
from . import emus
from . import autocorrelation as ac
from ._defaults import DEFAULT_IAT
from .usutils import unpack_nbrs


def calc_avg_ratio(psis, z, g1data, g2data=None, neighbors=None, iat_method=DEFAULT_IAT, kappa=None):
    """Estimates the asymptotic variance in the average of the ratio of two functions.
    ----------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    z : 1D array
        Array containing the normalization constants
    g1data : 2D data structure
        The values of the function in the numerator, evaluated for sample points from all the windows.
    g2data : 2D data structure
        The values of the function in the denomenator, evaluated for sample points from all the windows.
    neighbors : 2D array, optional
        List showing which windows neighbor which.  See neighbors_harmonic in usutils for explanation.
    iat_method : string or 1D array-like, optional
        Method used to estimate autocorrelation time.  See the documentation above.
    kappa : 1D array
        The effective sample size of each window.
    Returns
    -------
    autocovars : ndarray
        Array of length L (no. windows) where the i'th value corresponds to the autocovariance estimate for :math:`z_i`
    z_var_contribs : ndarray
        Two dimensional array, where element i,j corresponds to window j's contribution to the autocovariance of window i.
    z_var_iats : ndarray
        Two dimensional array, where element i,j corresponds to the autocorrelation time associated with window j's contribution to the autocovariance of window i.
    """
    if g2data is None:
        g2data = [np.ones(np.shape(g1data_i)) for g1data_i in g1data]
    g1star = emus._calculate_win_avgs(
        psis, z, g1data, neighbors, use_iter=True, kappa=kappa)
    g2star = emus._calculate_win_avgs(
        psis, z, g2data, neighbors, use_iter=True, kappa=kappa)
    g1 = np.dot(g1star, z)
    g2 = np.dot(g2star, z)
    partial_1 = 1. / g2
    partial_2 = - g1 / g2**2
    return calc_avg_avar(psis, z, partial_1, partial_2, g1data, g2data, neighbors, iat_method, kappa=kappa)


def calc_log_avg_ratio(psis, z, g1data, g2data=None, neighbors=None, iat_method=DEFAULT_IAT, kappa=None):
    """Estimates the asymptotic variance in the average of log(g1/g2).
    ----------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    z : 1D array
        Array containing the normalization constants
    g1data : 2D data structure
        The values of the function in the numerator, evaluated for sample points from all the windows.
    g2data : 2D data structure
        The values of the function in the denomenator, evaluated for sample points from all the windows.
    neighbors : 2D array, optional
        List showing which windows neighbor which.  See neighbors_harmonic in usutils for explanation.
    iat_method : string or 1D array-like, optional
        Method used to estimate autocorrelation time.  See the documentation above.
    kappa : 1D array
        The effective sample size of each window.
    Returns
    -------
    autocovars : ndarray
        Array of length L (no. windows) where the i'th value corresponds to the autocovariance estimate for :math:`z_i`
    z_var_contribs : ndarray
        Two dimensional array, where element i,j corresponds to window j's contribution to the autocovariance of window i.
    z_var_iats : ndarray
        Two dimensional array, where element i,j corresponds to the autocorrelation time associated with window j's contribution to the autocovariance of window i.
    """
    if g2data is None:
        g2data = [np.ones(np.shape(g1data_i)) for g1data_i in g1data]
    g1star = emus._calculate_win_avgs(
        psis, z, g1data, neighbors, use_iter=True, kappa=kappa)
    g2star = emus._calculate_win_avgs(
        psis, z, g2data, neighbors, use_iter=True, kappa=kappa)
    g1 = np.dot(g1star, z)
    g2 = np.dot(g2star, z)
    partial_1 = 1. / g1
    partial_2 = - 1. / g2
    return calc_avg_avar(psis, z, partial_1, partial_2, g1data, g2data, neighbors, iat_method, kappa=kappa)


def calc_fe_avar(psis, z, partial1, partial2, win1, win2, neighbors=None, iat_method=DEFAULT_IAT, kappa=None):
    """Estimates the asymptotic variance in the free energy difference between two windows in the form of -log(z_win1/z_win2).
    ----------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    z : 1D array
        Array containing the normalization constants
    partial1 : the partial derivative of the observable of interest wrt to g1 
    partial2 : the partial derivative of the observable of interest wrt to g2
    win1 : the index of the first window
    win1 : the index of the second window
    neighbors : 2D array, optional
        List showing which windows neighbor which.  See neighbors_harmonic in usutils for explanation.
    iat_method : string or 1D array-like, optional
        Method used to estimate autocorrelation time.  See the documentation above.
    Returns
    -------
    autocovars : ndarray
        Array of length L (no. windows) where the i'th value corresponds to the autocovariance estimate for :math:`z_i`
    fe_var : 1D array
        Window contributions to the asymptotic variance in the free energy difference
    fe_var_iats : 1D array
        The autocorrelation time of each window
    """
    L = len(z)
    if kappa is None:
        kappa = np.ones(L)
    fe_var = np.zeros(L)
    fe_var_iats = np.zeros(L)
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
    B_ginv = lm.calculate_GI_from_QR(psis, z, neighbors, kappa)
    for i, psi_i in enumerate(psis):
        # Data cleaning
        psi_i_arr = np.array(psi_i)
        Lneighb = len(neighbors[i])  # Number of neighbors
        # Normalize psi_j(x_i^t) for all j
        psi_sum = np.sum(np.array(
            [psi_i_arr[:, j]*kappa[neighbors[i][j]]/z[neighbors[i][j]] for j in range(Lneighb)]), axis=0)
        normedpsis = np.zeros(psi_i_arr.shape)  # psi_j / sum_k psi_k
        for j in range(Lneighb):
            normedpsis[:, j] = psi_i_arr[:, j] * \
                kappa[i]/z[neighbors[i][j]] / psi_sum
        # Calculate contribution to as. err.
        total_derivative = partial1 * \
            B_ginv[neighbors[i], win1]+partial2*B_ginv[neighbors[i], win2]
        err_t_series = np.dot(normedpsis, total_derivative)
        if iat_routine is not None:
            iat, mn, sigma = iat_routine(err_t_series)
            fe_var[i] = sigma * sigma
        else:
            iat = iats[i]
            fe_var[i] = np.var(
                err_t_series) * (iat / len(err_t_series))
        fe_var_iats[i] = iat
    autocovars = np.sum(fe_var)
    return autocovars, fe_var, fe_var_iats


def calc_avg_avar(psis, z, partial_1, partial_2, g1data, g2data=None, neighbors=None, iat_method=DEFAULT_IAT, kappa=None):
    """Estimates the asymptotic variance of a function of two averages.
    ----------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    z : 1D array
        Array containing the normalization constants
    partial1 : the partial derivative of the observable of interest wrt to g1 
    partial2 : the partial derivative of the observable of interest wrt to g2
    g1data : 2D data structure
        The values of the function in the numerator, evaluated for sample points from all the windows.
    g2data : 2D data structure
        The values of the function in the denomenator, evaluated for sample points from all the windows.
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
    if kappa is None:
        kappa = np.ones(L)
    z_var_iats = np.zeros(L)
    z_var_contribs = np.zeros(L)
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
        psis, z, g1data, neighbors, use_iter=True, kappa=kappa)
    g2star = emus._calculate_win_avgs(
        psis, z, g2data, neighbors, use_iter=True, kappa=kappa)
    g1 = np.dot(g1star, z)
    g2 = np.dot(g2star, z)
    v = np.append(z, [g1, g2])
    gs = [np.stack((np.array(g1data[i]), np.array(g2data[i])), axis=-1)
          for i in np.arange(L)]
    _, right_col = lm.GI_augmented(
        psis, z, g1, g2, g1data, g2data, neighbors, kappa)
    for i, psi_i in enumerate(psis):
        psi_i_arr = np.array(np.hstack((np.array(psi_i), gs[i])))
        Lneighb = len(neighbors[i])  # Number of neighbors
        # Normalize psi_j(x_i^t) for all j
        psi_sum = np.sum(np.array(
            [psi_i_arr[:, j]*kappa[neighbors[i][j]]/z[neighbors[i][j]] for j in range(Lneighb)]), axis=0)
        normedpsis = np.zeros(psi_i_arr.shape)  # psi_j / sum_k psi_k
        v_index = np.append(neighbors[i], [L, L+1])
        for j in range(Lneighb+2):
            normedpsis[:, j] = psi_i_arr[:, j]*kappa[i]/v[v_index[j]] / psi_sum
        total_deriv = partial_1 * \
            right_col[v_index, -2] + partial_2 * right_col[v_index, -1]
        err_t_series = np.dot(normedpsis, total_deriv)
        if iat_routine is not None:
            iat, mn, sigma = iat_routine(err_t_series)
            z_var_contribs[i] = sigma * sigma
        else:
            iat = iats[i]
            z_var_contribs[i] = np.var(
                err_t_series) * (iat / len(err_t_series))
        z_var_iats[i] = iat
    autocovars = np.sum(z_var_contribs)

    return autocovars, z_var_contribs, z_var_iats


def calc_partition_functions(psis, z, neighbors=None, iat_method=DEFAULT_IAT, kappa=None):
    """Estimates the asymptotic variance of the partition function (normalization constant) for each window.  To get an estimate of the autocovariance of the free energy for each window, multiply the autocovariance of window :math:`i` by :math:` (k_B T / z_i)^2`.
    Parameters
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
    if kappa is None:
        kappa = np.ones(L)
    z_var_contribs = np.zeros((L, L))
    z_var_iats = np.zeros((L, L))
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
    B_ginv = lm.calculate_GI_from_QR(psis, z, neighbors, kappa)
    # Iterate over windows, getting err contribution from sampling in each
    for i, psi_i in enumerate(psis):
        # Data cleaning
        psi_i_arr = np.array(psi_i)
        Lneighb = len(neighbors[i])  # Number of neighbors
        # Normalize psi_j(x_i^t) for all j
        psi_sum = np.sum(np.array(
            [psi_i_arr[:, j]*kappa[neighbors[i][j]]/z[neighbors[i][j]] for j in range(Lneighb)]), axis=0)
        normedpsis = np.zeros(psi_i_arr.shape)  # psi_j / sum_k psi_k
        for j in range(Lneighb):
            normedpsis[:, j] = psi_i_arr[:, j] * \
                kappa[i]/z[neighbors[i][j]] / psi_sum
        # Calculate contribution to as. err. for each z_k
        for k in range(L):
            err_t_series = np.dot(normedpsis, B_ginv[neighbors[i], k])
            if iat_routine is not None:
                iat, mn, sigma = iat_routine(err_t_series)
                z_var_contribs[k, i] = sigma * sigma
            else:
                iat = iats[i]
                z_var_contribs[k, i] = np.var(
                    err_t_series) * (iat / len(err_t_series))
            z_var_iats[k, i] = iat
    autocovars = np.sum(z_var_contribs, axis=1)
    return autocovars, z_var_contribs, z_var_iats
