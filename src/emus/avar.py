# -*- coding: utf-8 -*-
""" Library with routines associated with the asymptotic variance of the first EMUS iteration.  These estimates rely on estimates of the autocorrelation time of observables.  Multiple methods for estimating autocorrelation times are supported, these include the initial positive correlation estimator ('ipce') and the initial convex correlation estimator ('icce') by Geyer, and the acor algorithm ('acor') by Jonathan Goodman.  See the documentation to the `autocorrelation module <autocorrelation.html>`__ for more details.

"""

from __future__ import absolute_import
import numpy as np
from . import linalg as lm
from . import emus
from . import autocorrelation as ac
from ._defaults import DEFAULT_IAT, DEFAULT_KT
import warnings


def calc_avg_ratio(psis, z, F, g1data, g2data=None, neighbors=None, iat_method=DEFAULT_IAT, repexchange=False, kappa=None):
    """Estimates the asymptotic variance in the estimate of :math:`<g_1>/<g_2>`. If :math:`g_2` is not given, it just calculates the asymptotic variance associated with the average of :math:`g_1`.

    Parameters
    ----------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    z : 1D array
        Array containing the normalization constants
    F : 2D array
        Overlap matrix for the first EMUS iteration.
    g1data : 2D data structure
        Trajectory of observable in the numerator.  First dimension corresponds to the window index and the second to the point in the trajectory.
    g2data : 2D data structure, optional
        Trajectory of observable in the denominator of the ratio.  If not provided, taken to be all ones.
    neighbors : 2D array, optional
        List showing which windows neighbor which.  Element i,j is the j'th neighboring window of window i.
    iat_method : string or 1D array-like, optional
        Method used to estimate autocorrelation time.  Choices are 'acor', 'ipce', and 'icce'.  Alternatively, if an array of length no. windows is provided, element i is taken to be the autocorrelation time of window i.

    Returns
    -------
    iats : ndarray
        Array of length L (no. windows) where the i'th value corresponds to the iat for window i's contribution to the error.
    mean : scalar
        Estimate of the ratio
    variances : ndarray
        Array of length L (no. windows) where the i'th value corresponds to the autocovariance corresponding to window i's contribution to the error.  The total autocavariance of the ratio can be calculated by summing over the array.

        """

    # Clean the input and set defaults
    L = len(psis)
    if neighbors is None:
        neighbors = np.outer(np.ones(L), range(L)).astype(int)
    if kappa is None:
        kappa = np.ones(L)
    g1data = [np.array(g1i) for g1i in g1data]
    if g2data is None:
        g2data = [np.ones(np.shape(g1data_i)) for g1data_i in g1data]

    # Compute average of functions in each window.
    g1star = emus._calculate_win_avgs(
        psis, z, g1data, neighbors, use_iter=False)
    g2star = emus._calculate_win_avgs(
        psis, z, g2data, neighbors, use_iter=False)
    g1avg = np.dot(g1star, z)
    g2avg = np.dot(g2star, z)

    # Compute partial derivatives
    gI = lm.groupInverse(np.eye(L) - F)
    '''
    D=np.diag(kappa)
    D_inv=np.diag(1/kappa)
    B=np.eye(L)-F
    gI=np.linalg.multi_dot([D,lm.groupInverse(np.linalg.multi_dot([D_inv,B,D])),D_inv])
    '''
    dBdF = np.outer(z, np.dot(gI, g1star - g1avg / g2avg * g2star)) / g2avg
    dBdg1 = z / g2avg
    dBdg2 = -(g1avg / g2avg) * z / g2avg
    if repexchange:
        iats, variances = _calculate_acovar_repexchange(
            psis, dBdF, (g1data, g2data), (dBdg1, dBdg2), neighbors=neighbors, iat_method=iat_method)
    else:
        iats, variances = _calculate_acovar(
            psis, dBdF, (g1data, g2data), (dBdg1, dBdg2), neighbors=neighbors, iat_method=iat_method)
    return iats, g1avg / g2avg, variances


def calc_log_avg(psis, z, F, g1data, g2data=None, neighbors=None, iat_method=DEFAULT_IAT, repexchange=False , kappa=None):
    """Estimates the asymptotic variance in the EMUS estimate of :math:`-log <g_1>/<g_2>`.  If :math:`g_2` data is not provided, it estimates the asymptotic variance in the estimate of :math:`-log <g_1>/<g_2>`.  Input and output is as in average_ratio.  Note that if this is used for free energy differences, the result does not use the Boltzmann factor (i.e. :math:`k_B T=1`).  In that case, resulting variances should be scaled by the Boltzmann factor *squared*.

    """
    # Clean the input and set defaults
    L = len(psis)
    if neighbors is None:
        neighbors = np.outer(np.ones(L), range(L)).astype(int)
    if kappa is None:
        kappa=np.ones(L)
    g1data = [np.array(g1i) for g1i in g1data]
    if g2data is None:
        g2data = [np.ones(np.shape(g1data_i)) for g1data_i in g1data]
        
    # Compute average of functions in each window.
    g1star = emus._calculate_win_avgs(
        psis, z, g1data, neighbors, use_iter=False)
    g2star = emus._calculate_win_avgs(
        psis, z, g2data, neighbors, use_iter=False)
    g1avg = np.dot(g1star, z)
    g2avg = np.dot(g2star, z)
    B=np.eye(L)-F
    gI=lm.groupInverse(B)
    #gI=np.linalg.multi_dot([D,lm.groupInverse(np.linalg.multi_dot([D_inv,B,D])),D_inv])
    # Compute partial derivatives
    #gI = lm.groupInverse(np.eye(L) - F)
    dBdF = np.outer(z, np.dot(gI, g1star / g1avg - g2star / g2avg))
    dBdg1 = z / g1avg
    dBdg2 = -z / g2avg
    if repexchange:
        iats, variances = _calculate_acovar_repexchange(
            psis, dBdF, (g1data, g2data), (dBdg1, dBdg2), neighbors=neighbors, iat_method=iat_method)
    else:
        iats, variances = _calculate_acovar(
            psis, dBdF, (g1data, g2data), (dBdg1, dBdg2), neighbors=neighbors, iat_method=iat_method)
    return iats, -np.log(g1avg / g2avg), variances


def calc_avg_on_pmf(cv_trajs, psis, domain, z, F, g1data, g2data=None, neighbors=None, nbins=100, iat_method=DEFAULT_IAT, repexchange=False):
    """Estimates the asymptotic variance of an average on a pmf.

    Parameters
    ----------
    cv_trajs : 2D data structure
        Data structure containing trajectories in the collective variable space.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    domain : tuple
        Tuple containing the dimensions of the space over which to construct the pmf, e.g. (-180,180) or ((0,1),(-3.14,3.14)) z (1D array or list): Normalization constants for each window
    z : 1D array
        Array containing the normalization constants
    F : 2D array
        Overlap matrix for the first EMUS iteration.
    g1data : 2D data structure
        Trajectory of the observable in the numerator.  First dimension corresponds to the window index and the second to the point in the trajectory.
    g2data : 2D data structure, optional
        Trajectory of observable in the denominator of the ratio.  If not provided, taken to be all ones.
    neighbors : 2D array-like, optional
        List showing which windows neighbor which.  See neighbors_harmonic in usutils.
    nbins : int or tuple, optional
        Number of bins to use.  If int, uses that many bins in each dimension.  If tuple, e.g. (100,20), uses 100 bins in the first dimension and 20 in the second.
    iat_method : string or 1D array-like, optional
        Method used to estimate autocorrelation time.  Choices are 'acor', 'ipce', and 'icce'.  Alternatively, if an array of length no. windows is provided, element i is taken to be the autocorrelation time of window i.

    Returns
    -------

    """
    # Clean the input and set defaults
    L = len(psis)
    if neighbors is None:
        neighbors = np.outer(np.ones(L), range(L)).astype(int)
    domain = np.asarray(domain)
    if len(np.shape(domain)) == 1:
        domain = np.reshape(domain, (1, len(domain)))
    ndims = np.shape(domain)[0]
    if type(nbins) is int:  # Make nbins to an iterable in the 1d case.
        nbins = [nbins] * ndims
    if g2data is None:
        g2data = [np.ones(np.shape(g1data_i)) for g1data_i in g1data]
    g1data = [np.array(g1_i) for g1_i in g1data]
    g2data = [np.array(g2_i) for g2_i in g2data]

    # Warn user if they want to calculate each autocorrelation by hand.
    if isinstance(iat_method, str):
        warnings.warn("Program is set to compute the iat for every observable.  Since for a potential of mean force each histogram bin is an observable, this may be slow.  It may be better to compute a representative autocorrelation time for each window, and use those instead.")

    # Get the edges for each histogram bin
    edges = [np.linspace(domain[i, 0], domain[i, 1], nb + 1)
             for i, nb in enumerate(nbins)]

    # Get Group Inverse for the matrix
    gI = lm.groupInverse(np.eye(L) - F)

    means = np.zeros(nbins)
    avars = np.zeros(nbins)
    # Iterate over histogram_bins.
    for index, aval in np.ndenumerate(avars):
        g1data_hist = []
        g2data_hist = []
        for i, traj in enumerate(cv_trajs):
            if len(np.shape(traj)) == 1:
                traj = np.transpose([traj])
            inbin = np.ones(len(traj))
            for d, edge_d in enumerate(edges):
                hd_ndx = index[d]
                inhist_d = (traj[:, d] > edge_d[hd_ndx])
                inhist_d *= (traj[:, d] <= edge_d[hd_ndx + 1])
                inbin *= inhist_d
            g1data_hist.append(inbin * g1data[i])
            g2data_hist.append(inbin * g2data[i])
        g1star = emus._calculate_win_avgs(
            psis, z, g1data_hist, neighbors, use_iter=False)
        g1avg = np.dot(g1star, z)
        g2star = emus._calculate_win_avgs(
            psis, z, g2data_hist, neighbors, use_iter=False)
        g2avg = np.dot(g2star, z)
        dBdF = np.outer(z, np.dot(gI, g1star - g1avg / g2avg * g2star)) / g2avg
        dBdg1 = z / g2avg
        dBdg2 = -(g1avg / g2avg) * z / g2avg
        if repexchange:
            iats, variances = _calculate_acovar_repexchange(
                psis, dBdF, (g1data, g2data), (dBdg1, dBdg2), neighbors=neighbors, iat_method=iat_method)
        else:
            iats, variances = _calculate_acovar(
                psis, dBdF, (g1data, g2data), (dBdg1, dBdg2), neighbors=neighbors, iat_method=iat_method)
        avars[index] = np.sum(variances)
        means[index] = g1avg / g2avg
    return means, avars


def calc_pmf(cv_trajs, psis, domain, z, F, g2data=None, neighbors=None, nbins=100, kT=DEFAULT_KT, iat_method=DEFAULT_IAT, repexchange=False):
    """Estimates the asymptotic variance of a free energy surface.

    Parameters
    ----------
    cv_trajs : 2D data structure
        Data structure containing trajectories in the collective variable space.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    domain : tuple
        Tuple containing the dimensions of the space over which to construct the pmf, e.g. (-180,180) or ((0,1),(-3.14,3.14)) z (1D array or list): Normalization constants for each window
    z : 1D array
        Array containing the normalization constants
    F : 2D array
        Overlap matrix for the first EMUS iteration.
    neighbors : 2D array-like, optional
        List showing which windows neighbor which.  See neighbors_harmonic in usutils.
    nbins : int or tuple, optional
        Number of bins to use.  If int, uses that many bins in each dimension.  If tuple, e.g. (100,20), uses 100 bins in the first dimension and 20 in the second.
    kT : float, optional
        Value of kT to scale the PMF by.  If not provided, set to the default value.
    iat_method : string or 1D array-like, optional
        Method used to estimate autocorrelation time.  Choices are 'acor', 'ipce', and 'icce'.  Alternatively, if an array of length no. windows is provided, element i is taken to be the autocorrelation time of window i.

    Returns
    -------
    fes : ndarray
        Value of the free energy in each histogram bin.
    avars : ndarray
        Asymptotic variance of each histogram bin.

    """
    # Clean the input and set defaults
    L = len(psis)
    if neighbors is None:
        neighbors = np.outer(np.ones(L), range(L)).astype(int)
    domain = np.asarray(domain)
    if len(np.shape(domain)) == 1:
        domain = np.reshape(domain, (1, len(domain)))
    ndims = np.shape(domain)[0]
    if type(nbins) is int:  # Make nbins to an iterable in the 1d case.
        nbins = [nbins] * ndims
    if g2data is None:
        g2data = [np.ones(len(traj)) for traj in cv_trajs]
    if isinstance(iat_method, str):
        warnings.warn("Program is set to compute the iat for every observable.  Since for a potential of mean force each histogram bin is an observable, this may be slow.  It may be better to compute a representative autocorrelation time for each window, and use those instead.")

    # Get the edges for each histogram bin
    edges = [np.linspace(domain[i, 0], domain[i, 1], nb + 1)
             for i, nb in enumerate(nbins)]

    # Calculate quantities used for each histogram bin.
    gI = lm.groupInverse(np.eye(L) - F)
    g2star = emus._calculate_win_avgs(
        psis, z, g2data, neighbors, use_iter=False)
    g2avg = np.dot(g2star, z)

    fes = np.zeros(nbins)
    avars = np.zeros(nbins)
    # Iterate over histogram_bins.
    for index, aval in np.ndenumerate(avars):
        # Find part of trajectory inside the histogram bin.
        g1data = []
        for i, traj in enumerate(cv_trajs):
            if len(np.shape(traj)) == 1:
                traj = np.transpose([traj])
            g1_i = np.ones(len(traj))
            for d, edge_d in enumerate(edges):
                hd_ndx = index[d]
                inhist_d = (traj[:, d] > edge_d[hd_ndx])
                inhist_d *= (traj[:, d] <= edge_d[hd_ndx + 1])
                g1_i *= inhist_d
            g1data.append(g1_i)
        dA = np.prod([(edg_i[1] - edg_i[0]) for edg_i in edges])

        g1star = emus._calculate_win_avgs(
            psis, z, g1data, neighbors, use_iter=False)
        g1avg = np.dot(g1star, z)
        dBdF = np.outer(z, np.dot(gI, g1star / g1avg - g2star / g2avg))
        dBdg1 = z / g1avg
        dBdg2 = -z / g2avg
        if repexchange:
            iats, variances = _calculate_acovar_repexchange(
                psis, dBdF, (g1data, g2data), (dBdg1, dBdg2), neighbors=neighbors, iat_method=iat_method)
        else:
            iats, variances = _calculate_acovar(
                psis, dBdF, (g1data, g2data), (dBdg1, dBdg2), neighbors=neighbors, iat_method=iat_method)
        avars[index] = np.sum(variances) * (kT**2)
        fes[index] = -kT * np.log(g1avg / (dA * g2avg))
    return fes, avars


def calc_partition_functions(psis, z, F, neighbors=None, iat_method=DEFAULT_IAT, repexchange=False,kappa=None):
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
    z_var_contribs = []
    z_var_iats = []
    if neighbors is None:  # If no neighborlist, assume all windows neighbor
        neighbors = np.outer(np.ones(L), range(L)).astype(int)
    if kappa is None:
        kappa=np.ones(L)
    #D=np.diag(kappa)
    #D_inv=np.diag(1/kappa)
    B=np.eye(L)-F
    #groupInv=np.linalg.multi_dot([D,lm.groupInverse(np.linalg.multi_dot([D_inv,B,D])),D_inv])    
    groupInv=lm.groupInverse(B)
    # Calculate the partial derivatives of z .
    # (i,j,k)'th element is partial of z_k w.r.t. F_ij
    dzdFij = np.outer(z, groupInv).reshape((L, L, L))

    # Iterate over windows, getting err contribution from sampling in each
    for k in range(L):
        dzkdFij = dzdFij[:, :, k]

        if repexchange:
            iats, variances = _calculate_acovar_repexchange(
                psis, dzkdFij, neighbors=neighbors, iat_method=iat_method)
        else:
            iats, variances = _calculate_acovar(
                psis, dzkdFij, neighbors=neighbors, iat_method=iat_method)
        z_var_contribs.append(variances)
        z_var_iats.append(iats)
    z_var_contribs = np.array(z_var_contribs)
    z_var_iats = np.array(z_var_iats)
    autocovars = np.copy(z_var_contribs)
    if not repexchange:
        autocovars = np.sum(z_var_contribs, axis=1)
    else:
        autocovars = autocovars.ravel()
        z_var_contribs = z_var_contribs.ravel()
        z_var_iats = z_var_iats.ravel()
    return autocovars, z_var_contribs, z_var_iats


def _calculate_acovar(psis, dBdF, gdata=None, dBdg=None, neighbors=None, iat_method=DEFAULT_IAT):
    """
    Estimates the autocovariance and autocorrelation times for each window's contribution to the autocovariance of some observable B.

    Parameters
    ----------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    dBdF : array-like
        Two dimensional array, where element :math:`i,j` is the derivative of the estimate of B with respect to :math:`F_{ij}`
    gdata : array-like, optional
        Three dimensional data structure containing data from various observables.  The first index n
    dBdg : array-like, optional
        Two dimensional array, where element :math:`n,j` is the derivative of the estimate of B with respect to :math:`gn_j^*`.


    Returns
    -------
    iats : 1d array
        The value of the autocorrelation time for each trajectory.
    avars : 1d array
        Each window's contribution to the asymptotic variance.  Summing over windows gives the asymptotic variance of the system.

    """
    L = len(psis)
    if gdata is not None:
        if len(gdata) != len(dBdg):
            raise ValueError('Function data provided is mismatched with derivatives: respective sizes are ',
                             np.shape(gdata), ' and ', np.shape(dBdg))
    if neighbors is None:
        neighbors = np.outer(np.ones(L), range(L)).astype(int)
    dBdF = np.array(dBdF)
    iat_routine, iats = _parse_iat_routine(iat_method, L)

    sigmas = np.zeros(L)
    for i, psi_i in enumerate(psis):
        # Extract info for that window.
        nbrs_i = neighbors[i]
        dBdF_ij = dBdF[i, nbrs_i]
        if gdata is not None:
            gdata_i = [g_n[i] for g_n in gdata]
            dBdg_i = [dBdg_n[i] for dBdg_n in dBdg]
        else:
            gdata_i = None
            dBdg_i = None

        # Build time series, calculate error.
        err_t_series = _build_err_t_series(psi_i, dBdF_ij, gdata_i, dBdg_i)
        if iat_routine is not None:
            iat, mn, sigma = iat_routine(err_t_series)
            iats[i] = iat
        else:
            iat = iats[i]
            sigma = np.std(err_t_series) * np.sqrt(iat / len(err_t_series))
        sigmas[i] = sigma
    return iats, sigmas**2


def _calculate_acovar_repexchange(psis, dBdF, gdata=None, dBdg=None, neighbors=None, iat_method=DEFAULT_IAT):
    """
    Estimates the autocovariance and autocorrelation times for each window's contribution to the autocovariance of some observable B.

    Parameters
    ----------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    dBdF : array-like
        Two dimensional array, where element :math:`i,j` is the derivative of the estimate of B with respect to :math:`F_{ij}`
    gdata : array-like, optional
        Three dimensional data structure containing data from various observables.  The first index n
    dBdg : array-like, optional
        Two dimensional array, where element :math:`n,j` is the derivative of the estimate of B with respect to :math:`gn_j^*`.


    Returns
    -------
    iats : 1d array
        The value of the autocorrelation time for each trajectory.
    avars : 1d array
        Each window's contribution to the asymptotic variance.  Summing over windows gives the asymptotic variance of the system.

    """
    L = len(psis)
    if gdata is not None:
        if len(gdata) != len(dBdg):
            raise ValueError('Function data provided is mismatched with derivatives: respective sizes are ',
                             np.shape(gdata), ' and ', np.shape(dBdg))
    if neighbors is None:
        neighbors = np.outer(np.ones(L), range(L)).astype(int)
    dBdF = np.array(dBdF)

    if isinstance(iat_method, str):
        iat_routine = ac._get_iat_method(iat_method)
    else:
        try:  # Try to interpret iat_method as a single number.
            iat = float(iat_method)
        except (ValueError, TypeError) as err:
            err.message = "Was unable to interpret the input provided as a method to calculate the autocorrelation time or as a sequence of autocorrelation times.  Original error message follows: " + err.message
            raise err
        iat_routine = None

    err_t_series = np.zeros(psis[0].shape[0])
    for i, psi_i in enumerate(psis):
        # Extract info for that window.
        nbrs_i = neighbors[i]
        dBdF_ij = dBdF[i, nbrs_i]
        if gdata is not None:
            gdata_i = [g_n[i] for g_n in gdata]
            dBdg_i = [dBdg_n[i] for dBdg_n in dBdg]
        else:
            gdata_i = None
            dBdg_i = None

        # Build time series, calculate error.
        err_t_series += _build_err_t_series(psi_i, dBdF_ij, gdata_i, dBdg_i)
    if iat_routine is not None:
        iat, mn, sigma = iat_routine(err_t_series)
    else:
        sigma = np.std(err_t_series) * np.sqrt(iat / len(err_t_series))
    return iat, sigma**2


def _parse_iat_routine(iat_method, L):
    if isinstance(iat_method, str):
        iat_routine = ac._get_iat_method(iat_method)
        iats = np.zeros(L)
    else:  # Try to interpret iat_method as a collection of numbers
        try:
            iats = np.array([float(v) for v in iat_method])
        except (ValueError, TypeError) as err:
            err.message = "Was unable to interpret the input provided as a method to calculate the autocorrelation time or as a sequence of autocorrelation times.  Original error message follows: " + err.message
            raise err
        iat_routine = None
        if len(iats) != L:
            raise ValueError('IAT Input was interpreted to be a collection of precomputed autocorrelation times.  However, the number of autocorrelation times found (%d) is not equal to the number of states (%d).' % (len(iats), L))
    return iat_routine, iats


def _build_err_t_series(psi_i, dBdF_ij, gdata_i=None, dBdg_i=None):
    # Include contribution from psis
    denom_i = 1. / np.sum(psi_i, axis=1)
    err_t_series = psi_i * np.transpose([denom_i])
    Fi = np.average(err_t_series, axis=0)
    err_t_series = np.dot(
        (psi_i * np.transpose([denom_i]) - Fi), dBdF_ij)

    # Include contribution from observables.
    if gdata_i is not None:
        for n, g_ni in enumerate(gdata_i):
            dBdg_ni = dBdg_i[n]
            g_ni_wtd = g_ni * denom_i
            err_t_series += dBdg_ni * (g_ni_wtd - np.average(g_ni_wtd))
    return err_t_series


def getAllocations(importances, N_is, newWork):
    """Calculates the optimal allocation of sample points
    These are the optimal weights for
    To deal with negative weights, it removes all the negative weights, and calculates the weights for the resulting subproblem.

    """
    errs = np.copy(importances)
    ns = np.copy(N_is)
    testWeights = _calcWeightSubproblem(errs, ns, newWork)
    negativity = np.array([weit < 0.0 for weit in testWeights])
    while(any(negativity)):
        errs *= (1.0 - negativity)
        ns *= (1.0 - negativity)
        newWeights = _calcWeightSubproblem(errs, ns, newWork)
        testWeights = newWeights
        negativity = np.array([weit < 0.0 for weit in testWeights])
    # We return the weights, rounded and then converted to integers
    return map(int, map(round, testWeights))


def _calcWeightSubproblem(importances, N_is, newWork):
    """Calculates the sampling weights of each region, according to the method using Lagrange Modifiers.

    """
    totalWork = np.sum(N_is)
    weights = np.zeros(importances.shape)
    varConstants = importances * np.sqrt(N_is)
    constSum = np.sum(varConstants)
    for ind, val in enumerate(varConstants):
        weights[ind] = val / constSum * (newWork + totalWork) - N_is[ind]
    return weights
