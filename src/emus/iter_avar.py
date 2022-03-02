# -*- coding: utf-8 -*-
"""
Routines for 
"""
from __future__ import absolute_import
import numpy as np
from . import linalg as lm
from . import emus
from . import autocorrelation as ac
from ._defaults import DEFAULT_IAT


def calc_log_avg_ratio(psis, z, g1_data, g2_data=None, neighbors=None, iat_method=DEFAULT_IAT, kappa=None):
    """
    Calculates the asymptotic variance in the log ratio of two averages

    Parameters
    ----------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.
        See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    z : 1D array
        Array containing the normalization constants
    g1data : 2D data structure
        Trajectory of observable in the numerator.  First dimension corresponds
        to the window index and the second to the point in the trajectory.
    g2data : 2D data structure, optional
        Trajectory of observable in the denominator.
    neighbors : 2D array-like, optional
        List showing which windows neighbor which.  See neighbors_harmonic in usutils.
    iat_method : string or 1D array-like, optional
        Method used to estimate autocorrelation time.  Choices are 'acor', 'ipce', and 'icce'.
        Alternatively, if an array of length no. windows is provided,
        element i is taken to be the autocorrelation time of window i.
    kappa ; 1D array-like, optional
        Weighting factor for each state, loosely corresponds to the fraction of statistical
        power coming from that state.

    Returns
    -------
    err : float
        Estimated asymptotic variance
    contribs : 1D numpy array
        Array of length L (no. windows) where the i'th value corresponds to the
        autocovariance corresponding to window i's contribution to the error.
        The total autocavariance of the ratio can be calculated by summing over
        the array.
    iats : 1D numpy array
        Array of length L (no. windows) where the i'th value is the autocorrelation
        time associated with window i.
    """
    if g2_data is None:
        g2_data = [np.ones(np.shape(g1_data_i)) for g1_data_i in g1_data]
    if kappa is None:
        kappa = np.array([len(psi_i) for psi_i in psis])
        kappa = kappa / np.sum(kappa)
    if neighbors is None:
        L = len(z)
        neighbors = [np.arange(L)] * L

    # Calculate partial derivs
    g1star = emus._calculate_win_avgs(
        psis, z, g1_data, neighbors, use_iter=True, kappa=kappa)
    g2star = emus._calculate_win_avgs(
        psis, z, g2_data, neighbors, use_iter=True, kappa=kappa)
    g1 = np.dot(g1star, z)
    g2 = np.dot(g2star, z)
    partial_1 = 1. / g1
    partial_2 = - 1. / g2

    state_fe = -np.log(z)
    out = _calc_acovar_from_derivs(psis, state_fe, partial_1,
                                   partial_2, g1_data, g2_data,
                                   neighbors, iat_method, kappa)
    return out


def calc_fe_avar(psis, z, win_1, win_2, neighbors=None, iat_method=DEFAULT_IAT, kappa=None):
    """
    Calculates the asyptotic variance in the free energy difference between two states.

    Parameters
    ----------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.
        See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    z : 1D array
        Array containing the normalization constants
    win_1 : int
        index of the first state
    win_2 : int
        index of the second state
    neighbors : 2D array-like, optional
        List showing which windows neighbor which.  See neighbors_harmonic in usutils.
    iat_method : string or 1D array-like, optional
        Method used to estimate autocorrelation time.  Choices are 'acor', 'ipce', and 'icce'.
        Alternatively, if an array of length no. windows is provided,
        element i is taken to be the autocorrelation time of window i.
    kappa ; 1D array-like, optional
        Weighting factor for each state, loosely corresponds to the fraction of statistical
        power coming from that state.

    Returns
    -------
    err : float
        Estimated asymptotic variance
    contribs : 1D numpy array
        Array of length L (no. windows) where the i'th value corresponds to the
        autocovariance corresponding to window i's contribution to the error.
        The total autocavariance of the ratio can be calculated by summing over
        the array.
    iats : 1D numpy array
        Array of length L (no. windows) where the i'th value is the autocorrelation
        time associated with window i.
    """
    # TODO: integrate with _calc_acovar_from_derivs
    g1_data = [np.ones(np.shape(psi_i)[0]) for psi_i in psis]
    g2_data = [np.ones(np.shape(psi_i)[0]) for psi_i in psis]

    if kappa is None:
        kappa = np.array([len(psi_i) for psi_i in psis])
        kappa = kappa / np.sum(kappa)
    if neighbors is None:
        L = len(z)
        neighbors = [np.arange(L)] * L
    state_fe = -np.log(z)

    L = len(state_fe)
    normed_psis, normed_w1, normed_w2 = _build_normed_trajs(psis, state_fe, g1_data, g2_data, neighbors, kappa)
    fixed_point_deriv = _build_fixed_point_deriv(normed_psis, normed_w1, normed_w2, neighbors, kappa)

    partial_vec = np.zeros(L+2)
    partial_vec[win_1] = 1.
    partial_vec[win_2] = -1.
    partial_derivs = fixed_point_deriv @ partial_vec

    err_trajs = _build_err_trajs(normed_psis, normed_w1, normed_w2, partial_derivs, kappa, neighbors)
    err, contribs, iats = _get_iats_and_acov_from_traj(err_trajs, iat_method)
    return err, contribs, iats


def _calc_acovar_from_derivs(psis, state_fe, partial_1, partial_2, g1_data, g2_data=None,
                             neighbors=None, iat_method=DEFAULT_IAT, kappa=None):
    """
    Estimates the autocovariance from the individual derivative
    """
    L = len(state_fe)

    normed_psis, normed_w1, normed_w2 = _build_normed_trajs(psis, state_fe, g1_data, g2_data, neighbors, kappa)

    fixed_point_deriv = _build_fixed_point_deriv(normed_psis, normed_w1, normed_w2, neighbors, kappa)

    partial_vec = np.zeros(L+2)
    partial_vec[L] = partial_1
    partial_vec[L+1] = partial_2
    
    partial_derivs = fixed_point_deriv.T @ partial_vec
    err_trajs = _build_err_trajs(normed_psis, normed_w1, normed_w2, partial_derivs, kappa, neighbors)
    err, contribs, iats = _get_iats_and_acov_from_traj(err_trajs, iat_method)
    return err, contribs, iats


def _build_fixed_point_deriv(normed_psis, normed_w1, normed_w2, neighbors, kappa):
    H, omega = _build_fixed_deriv_mats(normed_psis, normed_w1, normed_w2, neighbors, kappa)
    H_ginv = lm.groupInverse(H)
    H_o = - omega @ H_ginv

    # Put submatrices in the correct position
    L = H_ginv.shape[0]
    total_deriv = np.eye(L+2)
    total_deriv[:L, :L] = H_ginv
    total_deriv[L:, :L] = H_o
    return total_deriv


def _build_fixed_deriv_mats(normed_psis, normed_w1, normed_w2, neighbors, kappa):
    L = len(normed_psis)

    H_raw = np.zeros((L, L))
    for i, psi_i in enumerate(normed_psis):
        nbrs_i = np.array(neighbors[i])
        H_avg = kappa[i] * np.mean(psi_i, axis=0)
        H_raw[i, nbrs_i] = H_avg

    beta = np.zeros((2, L))
    for i in range(L):
        # Build beta matrix
        beta_1 = kappa[i] * np.mean(normed_w1[i])
        beta_2 = kappa[i] * np.mean(normed_w2[i])
        beta[0, i] = beta_1
        beta[1, i] = beta_2

    # subtract from scaled identity.
    H = (np.eye(L) * kappa) - H_raw
    return H, beta


def _build_normed_trajs(psis, state_fe, g1_data, g2_data, neighbors, kappa):
    normed_psis = []
    normed_w1 = []
    normed_w2 = []
    z = np.exp(-state_fe)

    for i, psi_i in enumerate(psis):
        psi_i = np.array(psi_i)
        n_i = neighbors[i]

        psi_weights = kappa[n_i] / z[n_i]
        weighted_psis = psi_i * psi_weights
        psi_sum = np.sum(weighted_psis, axis=1, keepdims=True)
        normed_psis.append(weighted_psis / psi_sum)

        normed_w1_i = g1_data[i] / psi_sum.ravel()
        normed_w2_i = g2_data[i] / psi_sum.ravel()
        normed_w1.append(normed_w1_i)
        normed_w2.append(normed_w2_i)
    return normed_psis, normed_w1, normed_w2


def _build_err_trajs(normed_psis, normed_w1, normed_w2, partial_derivs,
                     kappa, neighbors):
    xis = []
    for i, nbrs_i in enumerate(neighbors):
        n_psi_i = normed_psis[i]
        n_w1 = normed_w1[i]
        n_w2 = normed_w2[i]

        xi_i = np.dot(n_psi_i, partial_derivs[nbrs_i])
        xi_i += n_w1 * partial_derivs[-2]
        xi_i += n_w2 * partial_derivs[-1]
        xis.append(xi_i * kappa[i])
    return xis


def _get_iats_and_acov_from_traj(err_trajs, iat_method):
    L = len(err_trajs)
    # iat_routine, iats = _parse_iat_method(iat_method, L)

    contribs = np.zeros(L)
    iats = np.zeros(L)

    if isinstance(iat_method, str):  # IATS need to be computed
        for i, err_traj_i in enumerate(err_trajs):
            iat_routine = ac._get_iat_method(iat_method)
            iat_i, mn_i, sigma_i = iat_routine(err_traj_i)
            contribs[i] = sigma_i * sigma_i
            iats[i] = iat_i
    else:  # IATS seem to be precomputed
        try:
            iats = np.array([float(v) for v in iat_method])
        except (ValueError, TypeError) as err:
            err.message = "Was unable to interpret the input provided as a method to calculate the " +\
                "autocorrelation time or as a sequence of autocorrelation times.  Original error message follows: " +\
                err.message
        for i, err_traj_i in enumerate(err_trajs):
            contribs[i] = np.var(err_traj_i) * (iats[i] / len(err_traj_i))

    return np.sum(contribs), contribs, iats
