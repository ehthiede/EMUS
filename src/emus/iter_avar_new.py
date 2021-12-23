# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from . import linalg as lm
from . import emus
from . import autocorrelation as ac
from ._defaults import DEFAULT_IAT


def calc_avg_ratio(psis, z, g1data, g2data=None, neighbors=None, iat_method=DEFAULT_IAT, kappa=None):
    """Estimates the asymptotic variance in the average of the ratio of two functions.
    ----------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.
        See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
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
        Two dimensional array, where element i,j corresponds to window j's contribution
        to the autocovariance of window i.
    z_var_iats : ndarray
        Two dimensional array, where element i,j corresponds to the autocorrelation time associated with
        window j's contribution to the autocovariance of window i.
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
    return calc_avg_avar(psis, -np.log(z), partial_1, partial_2, g1data, g2data, neighbors, iat_method, kappa=kappa)


def calc_log_avg_ratio(psis, z, g1data, g2data=None, neighbors=None, iat_method=DEFAULT_IAT, kappa=None):
    """Estimates the asymptotic variance in the average of log(g1/g2).
    ----------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.
        See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
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
        Two dimensional array, where element i,j corresponds to window j's contribution to the
        autocovariance of window i.
    z_var_iats : ndarray
        Two dimensional array, where element i,j corresponds to the autocorrelation time associated with window j's
        acontribution to the autocovariance of window i.
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
    return calc_avg_avar(psis, -np.log(z), partial_1, partial_2, g1data, g2data, neighbors, iat_method, kappa=kappa)


def _calc_acovar_from_derivs(psis, state_fe, partial_1, partial_2, g1_data, g2_data=None,
                             neighbors=None, iat_method=DEFAULT_IAT, kappa=None):
    """
    Estimates the autocovariance from the individual derivative
    """
    L = len(state_fe)
    normed_psis, normed_w1, normed_w2 = _build_normed_trajs(psis, g1_data, g2_data, neighbors, kappa)
    fixed_point_deriv = _build_fixed_point_deriv(normed_psis, normed_w1, normed_w2, neighbors, kappa)

    partial_vec = np.zeros(L+2)
    partial_vec[L+1] = partial_1
    partial_vec[L+2] = partial_2

    partial_derivs = fixed_point_deriv @ partial_vec
    err_trajs = _build_err_trajs(normed_psis, normed_w1, normed_w2, partial_derivs, neighbors)

    err, contribs, iats = _get_iats_and_acov_from_traj(err_trajs, iat_method)


def _build_fixed_point_deriv(normed_psis, normed_w1, normed_w2, neighbors, kappa):
    H, omega = _build_fixed_deriv_mats(normed_psis, normed_w1, normed_w2, neighbors, kappa)

    H_ginv = lm.groupInverse(H)
    H_o = H_ginv @ omega

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
        beta_1 = kappa[i] * np.mean(normed_w1)
        beta_2 = kappa[i] * np.mean(normed_w2)
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
                     neighbors):
    xis = []
    for i, nbrs_i in enumerate(neighbors):
        n_psi_i = normed_psis[i]
        n_w1 = normed_w1[i]
        n_w2 = normed_w2[i]

        xi_i = np.dot(n_psi_i, partial_derivs[nbrs_i])
        xi_i += n_w1 * partial_derivs[-2]
        xi_i += n_w2 * partial_derivs[-1]
        xis.append(xi_i)
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


def calc_avg_avar(psis, state_fe, partial_1, partial_2, g1data, g2data=None,
                  neighbors=None, iat_method=DEFAULT_IAT, kappa=None):
    """Estimates the asymptotic variance of a function of two averages.
    ----------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.
        See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    state_fe : 1D array
        Array containing the state free energies
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
        Two dimensional array, where element i,j corresponds to window j's
        contribution to the autocovariance of window i.
    z_var_iats : ndarray
        Two dimensional array, where element i,j corresponds to the autocorrelation time
        associated with window j's contribution to the autocovariance of window i.
    """
    L = len(state_fe)
    if kappa is None:
        kappa = np.ones(L)
    z_var_iats = np.zeros(L)
    z_var_contribs = np.zeros(L)
    iat_routine, iats = _parse_iat_method(iat_method, L)
    if neighbors is None:  # If no neighborlist, assume all windows neighbor
        neighbors = np.outer(np.ones(L), range(L)).astype(int)
    if g2data is None:
        g2data = [np.ones(np.shape(g1data_i)) for g1data_i in g1data]
    g1star = emus._calculate_win_avgs(
        psis, np.exp(-state_fe), g1data, neighbors, use_iter=True, kappa=kappa)
    g2star = emus._calculate_win_avgs(
        psis, np.exp(-state_fe), g2data, neighbors, use_iter=True, kappa=kappa)
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
    """
    Estimates the asymptotic variance of the partition function (normalization constant) for each window.
    To get an estimate of the autocovariance of the free energy for each window, multiply the autocovariance
    of window :math:`i` by :math:` (k_B T / z_i)^2`.

    Parameters
    ----------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.
        See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
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
        Two dimensional array, where element i,j corresponds to window j's contribution
        to the autocovariance of window i.
    z_var_iats : ndarray
        Two dimensional array, where element i,j corresponds to the autocorrelation time associated with
        window j's contribution to the autocovariance of window i.
    """
    L = len(z)
    if kappa is None:
        kappa = np.ones(L)
    z_var_contribs = np.zeros((L, L))
    z_var_iats = np.zeros((L, L))

    iat_routine, iats = _parse_iat_method(iat_method, L)

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


def _parse_iat_method(iat_method, L):
    if isinstance(iat_method, str):
        iat_routine = ac._get_iat_method(iat_method)

    else:  # Try to interpret iat_method as a collection of numbers
        try:
            iats = np.array([float(v) for v in iat_method])
        except (ValueError, TypeError) as err:
            err.message = "Was unable to interpret the input provided as a method to calculate the " +\
                "autocorrelation time or as a sequence of autocorrelation times.  Original error message follows: " +\
                err.message
        iat_routine = None
        if len(iats) != L:
            raise ValueError(
                "IAT Input was interpreted to be a collection of precomputed autocorrelation times."
                + "However, the number of autocorrelation times found (%d) is not equal to the number of states (%d)."
                % (len(iats), L))
    return iat_routine, iats
