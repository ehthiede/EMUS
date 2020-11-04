# -*- coding: utf-8 -*-
""" Library with routines associated with the asymptotic variance of the first EMUS iteration.  These estimates rely on estimates of the autocorrelation time of observables.  Multiple methods for estimating autocorrelation times are supported, these include the initial positive correlation estimator ('ipce') and the initial convex correlation estimator ('icce') by Geyer, and the acor algorithm ('acor') by Jonathan Goodman.  See the documentation to the `autocorrelation module <autocorrelation.html>`__ for more details.
"""

from __future__ import absolute_import
import numpy as np
from . import linalg as lm
from . import emus
from . import autocorrelation as ac
from ._defaults import DEFAULT_IAT
from .usutils import unpack_nbrs

def group_inverse(A,A0,niter):
    #normA=np.linalg.norm(A)
    #A0=1/normA**2*A
    Ai=A0
    Id=np.eye(np.shape(A)[0])
    for i in np.arange(niter):
        Ai=A0+np.dot((Id-np.dot(A0,A)),Ai)
    return Ai


def check_GI(mat, gi_mat):
    test_1_p1 = np.dot(mat, gi_mat)
    test_1_p2 = np.dot(gi_mat, mat)
    test_1_denom = min(np.linalg.norm(test_1_p1), np.linalg.norm(test_1_p2))
    test_1 = np.linalg.norm(np.dot(mat, gi_mat) - np.dot(gi_mat, mat))/test_1_denom
    test_2 = np.linalg.norm(gi_mat @ mat @ gi_mat - gi_mat) / np.linalg.norm(gi_mat)
    test_3 = np.linalg.norm(mat @ gi_mat @ mat - mat) / np.linalg.norm(mat)
    print("(A A^# - A^# A) / min(||A^# A||, ||A A^#||) : ", test_1)
    print("(A^# A A^# - A^#)/||A^#|| : ", test_2)
    print("(A A^# A - A)/||A|| : ", test_3)

def build_F(psis, v,neighbors,kappa,g1=None,g2=None):
    """
    Builds matrix used in iteration.

    Parameters
    ---------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    v : 1D array-like
        fixed point of the iemus iteration.  Elements 0..L-1 are the z values.
        Elements L and onward are averages of observables.

    Returns
    -------
    F : 2d numpy array
        Matrix where element ij is <a_ij>_i as depicted in ????
    """
    L = len(psis)  # Number of windows
    F = []
    z = v[:L]
    for i in range(L):
        psis_i = np.array(psis[i])
        Lneighb = len(neighbors[i])
        denom = np.sum(np.array([kappa[neighbors[i][j]]*psis_i[:, j]/z[neighbors[i][j]] for j in range(Lneighb)]), axis=0)
        Fi=kappa[i]*psis_i[:,:Lneighb] / z[neighbors[i]]
        Fi /= denom.reshape(-1, 1)
        Fi = np.mean(Fi, axis=0)
        Fi=unpack_nbrs(Fi,neighbors[i],L)
        if len(v)>L:
            #Fi_additional=psis_i[:,-2:]/v[-2:]
            Fi_additional_1=kappa[i]*g1[i]/v[-2]
            Fi_additional_1 /= denom
            Fi_additional_1 = float(np.mean(Fi_additional_1, axis=0))
            Fi_additional_2=kappa[i]*g2[i]/v[-1]
            Fi_additional_2 /= denom
            Fi_additional_2 = float(np.mean(Fi_additional_2, axis=0))
            Fi=np.append(np.array(Fi),np.array([Fi_additional_1,Fi_additional_2]))
        F.append(Fi)
    return np.array(F)


def calc_partition_functions(psis, z, neighbors=None, iat_method=DEFAULT_IAT,kappa=None):
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
        kappa=np.ones(L)
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

    F = build_F(psis, z,neighbors,kappa)
    F = np.array(F)
    # START POINT A
    # F = F.T
    # END POINT A

    # Old Groupinv code
    # groupInv = lm.groupInverse(np.eye(L) - F)
    # # groupInv=np.linalg.pinv(np.eye(L) - F)
    # # Calculate the partial derivatives of z .
    # # (i,j,k)'th element is partial of z_k w.r.t. F_ij
    # # dzdFij = np.outer(z, groupInv).reshape((L, L, L))
    # dzdFij = np.matmul(groupInv, np.diag(z))
    Bmat = np.dot(np.diag(1./z), np.eye(L)-np.transpose(F))
    # print(np.linalg.norm(dzdFij - lm.groupInverse(Lf)))
    # print((dzdFij - lm.groupInverse(Lf)))
    # print((dzdFij - lm.groupInverse(Lf))/ lm.groupInverse(Lf))
    dzdFij = lm.groupInverse_for_iteravar(Bmat)
    # Iterate over windows, getting err contribution from sampling in each
    for i, psi_i in enumerate(psis):
        # Data cleaning
        psi_i_arr = np.array(psi_i)
        Lneighb = len(neighbors[i])  # Number of neighbors
        # Normalize psi_j(x_i^t) for all j
        psi_sum = np.sum(np.array([psi_i_arr[:, j]/z[neighbors[i][j]] for j in range(Lneighb)]), axis=0)
        normedpsis = np.zeros(psi_i_arr.shape)  # psi_j / sum_k psi_k
        for j in range(Lneighb):
            normedpsis[:, j] = psi_i_arr[:, j]/z[neighbors[i][j]] / psi_sum
        # Calculate contribution to as. err. for each z_k
        for k in range(L):
            # dzkdFij = dzdFij[:, :, k]
            # err_t_series = np.array([np.dot(normedpsis[t], dzkdFij[neighbors[i],k]) for t in range(np.shape(normedpsis)[0])])
            # err_t_series = np.array([np.dot(normedpsis[t], dzdFij[k][neighbors[i]]) for t in range(np.shape(normedpsis)[0])])
            # err_t_series = np.array([np.dot(normedpsis[t], dzdFij[k][neighbors[i]]) for t in range(np.shape(normedpsis)[0])])
            err_t_series = np.dot(normedpsis, dzdFij[neighbors[i], k])
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


def calc_fe_avar(psis, z,partial1,partial2, win1,win2,neighbors=None, iat_method=DEFAULT_IAT):
    L = len(z)
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
    F = build_F(psis, z, neighbors)
    F = np.array(F)
    Bmat = np.dot(np.diag(1./z), np.eye(L)-np.transpose(F))
    dzdFij = lm.groupInverse(Bmat)
    for i, psi_i in enumerate(psis):
        # Data cleaning
        psi_i_arr = np.array(psi_i)
        Lneighb = len(neighbors[i])  # Number of neighbors
        # Normalize psi_j(x_i^t) for all j
        psi_sum = np.sum(np.array([psi_i_arr[:, j]/z[neighbors[i][j]] for j in range(Lneighb)]), axis=0)
        normedpsis = np.zeros(psi_i_arr.shape)  # psi_j / sum_k psi_k
        for j in range(Lneighb):
            normedpsis[:, j] = psi_i_arr[:, j]/z[neighbors[i][j]] / psi_sum
        # Calculate contribution to as. err. for each z_k
        total_derivative=partial1*dzdFij[neighbors[i], win1]+partial2*dzdFij[neighbors[i], win2]
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


def calc_avg_ratio(psis, z, g1data, g2data=None, neighbors=None, iat_method=DEFAULT_IAT,kappa=None):
    if g2data is None:
        g2data = [np.ones(np.shape(g1data_i)) for g1data_i in g1data]
    g1star = emus._calculate_win_avgs(
        psis, z, g1data, neighbors, use_iter=True)
    g2star = emus._calculate_win_avgs(
        psis, z, g2data, neighbors, use_iter=True)
    g1 = np.dot(g1star, z)
    g2 = np.dot(g2star, z)
    partial_1 = 1. / g2
    partial_2 = - g1 / g2**2
    return calc_avg_avar(psis, z, partial_1, partial_2, g1data, g2data, neighbors, iat_method,kappa=kappa)


def calc_log_avg_ratio(psis, z, g1data, g2data=None, neighbors=None, iat_method=DEFAULT_IAT,kappa=None):
    if g2data is None:
        g2data = [np.ones(np.shape(g1data_i)) for g1data_i in g1data]
    g1star = emus._calculate_win_avgs(
        psis, z, g1data, neighbors, use_iter=True,kappa=kappa)
    g2star = emus._calculate_win_avgs(
        psis, z, g2data, neighbors, use_iter=True,kappa=kappa)
    g1 = np.dot(g1star, z)
    g2 = np.dot(g2star, z)
    partial_1 = 1. / g1
    partial_2 = - 1. / g2
    return calc_avg_avar(psis, z, partial_1, partial_2, g1data, g2data, neighbors, iat_method,kappa=kappa)


def build_deriv_mat(F, v, g1, g2):
    L, num_avgs = F.shape
    Bmat = np.eye(num_avgs)
    Bmat[:L] -= F
    Bmat = np.dot(np.diag(1. / v), Bmat)
    return Bmat


def build_B_inverse(B):
    A=B[0:-2,0:-2]
    L=np.shape(A)[0]
    A_inv=lm.groupInverse_for_iteravar(A)
    print('Checking submat')
    check_GI(A, A_inv)
    v=B[0:-2,-2:]
    b=B[-2:,-2:]
    b_inv= np.linalg. inv(b) 
    T=-np.linalg.multi_dot([A_inv,v,b_inv])+np.linalg.multi_dot([np.eye(L)-np.dot(A,A_inv),v,b_inv,b_inv])
    GI = np.vstack((np.hstack((A_inv,T)),np.hstack((np.zeros((2,L)),b_inv))))
    print("CHecking Full")
    check_GI(B, GI)
    return GI

def calc_avg_avar(psis, z, partial_1, partial_2, g1data, g2data=None, neighbors=None, iat_method=DEFAULT_IAT,kappa=None):
    """Estimates the asymptotic variance of a function of two averages.
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
        kappa=np.ones(L)
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
        psis, z, g1data, neighbors, use_iter=True,kappa=kappa)
    g2star = emus._calculate_win_avgs(
        psis, z, g2data, neighbors, use_iter=True,kappa=kappa)
    g1 = np.dot(g1star, z)
    g2 = np.dot(g2star, z)
    v = np.append(z, [g1, g2])
    #gs = np.stack((np.array(g1data), np.array(g2data)), axis=-1)
    gs=[np.stack((np.array(g1data[i]), np.array(g2data[i])), axis=-1) for i in np.arange(L)]
    #psis = [np.hstack((psi_i, g_i)) for (psi_i, g_i) in zip(psis, gs)]
    '''
    F = build_F(psis, v,neighbors)
    F = np.transpose(np.array(F))
    Bmat = np.dot(np.diag(1./v), np.eye(L+2, L)-F)
    print(Bmat.shape, "Bmat shape")
    Bmat = np.hstack((Bmat, np.zeros((Bmat.shape[0], 2))))
    print(Bmat.shape, "Bmat shape")
    print("Bmat[:, -1]", Bmat[:, -1])
    print("Bmat[-1]", Bmat[-1])
    Bmat[L, L] = 1/g1
    Bmat[L+1, L+1] = 1/g2
    print('----------------')
    print("Bmat[:, -2]", Bmat[:, -2:])
    print('--------')
    print("Bmat[-2:]", Bmat[-2:])
    print('----------------')
    B_ginv = lm.groupInverse(Bmat)
    # bottom_block = dzdFij[-2:]
    total_deriv = partial_1 * B_ginv[-2] + partial_2 * B_ginv[-1]
    # total_deriv = partial_1 * B_ginv[:, -2] + partial_2 * B_ginv[:, -1]
    '''
    F = build_F(psis, v,neighbors,kappa,g1data,g2data)
    #np.save("F",F)
    #print(np.shape(F))
    Bmat = build_deriv_mat(F, v, g1, g2)
    #np.save('Bmat.npy',Bmat)
    #np.save('F_and_B/w_neighbs/F_%d_windows'%L, F)
    #np.save('F_and_B/w_neighbs/B_%d_windows'%L, Bmat)
    #B_ginv = lm.groupInverse(Bmat)
    B_ginv=build_B_inverse(Bmat)
    #total_deriv = partial_1 * B_ginv[:, -2] + partial_2 * B_ginv[:, -1]
    # Iterate over windows, getting err contribution from sampling in each
    for i, psi_i in enumerate(psis):
        psi_i_arr = np.array(np.hstack((np.array(psi_i),gs[i])))
        Lneighb = len(neighbors[i])  # Number of neighbors
        # Normalize psi_j(x_i^t) for all j
        psi_sum = np.sum(np.array([psi_i_arr[:, j]/z[neighbors[i][j]] for j in range(Lneighb)]), axis=0)
        normedpsis = np.zeros(psi_i_arr.shape)  # psi_j / sum_k psi_k
        v_index = np.append(neighbors[i], [L, L+1])
        for j in range(Lneighb+2):
            normedpsis[:, j] = psi_i_arr[:, j]/v[v_index[j]] / psi_sum
        total_deriv = partial_1 * B_ginv[v_index, -2] + partial_2 * B_ginv[v_index, -1]
        #total_deriv = partial_1 * B_ginv[-2,v_index] + partial_2 * B_ginv[-1,v_index]
        err_t_series = np.dot(normedpsis, total_deriv)
        np.save("err_traj",err_t_series)

        #print(err_t_series.shape, 'err t series')
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




def calc_partition_functions_2(psis, z, neighbors=None, iat_method=DEFAULT_IAT):
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
    F = build_F(psis, z)
    F = np.array(F)
    dzdFij = np.dot(lm.groupInverse(np.eye(L)-np.transpose(F)), np.diag(z))
    for i, psi_i in enumerate(psis):
        # Data cleaning
        psi_i_arr = np.array(psi_i)
        Lneighb = len(neighbors[i])  # Number of neighbors
        # Normalize psi_j(x_i^t) for all j
        psi_sum = np.sum(np.array([psi_i_arr[:, j]/z[j] for j in range(Lneighb)]), axis=0)
        normedpsis = np.zeros(psi_i_arr.shape)  # psi_j / sum_k psi_k
        for j in range(Lneighb):
            normedpsis[:, j] = psi_i_arr[:, j]/z[j] / psi_sum
        # Calculate contribution to as. err. for each z_k
        for k in range(L):
            # dzkdFij = dzdFij[:, :, k]
            # err_t_series = np.array([np.dot(normedpsis[t], dzkdFij[neighbors[i],k]) for t in range(np.shape(normedpsis)[0])])
            # err_t_series = np.array([np.dot(normedpsis[t], dzdFij[k][neighbors[i]]) for t in range(np.shape(normedpsis)[0])])
            # err_t_series = np.array([np.dot(normedpsis[t], dzdFij[k][neighbors[i]]) for t in range(np.shape(normedpsis)[0])])
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

    sigmas = np.zeros(L)
    for i, psi_i in enumerate(psis):
        nbrs_i = neighbors[i]
        denom_i = 1. / np.sum(psi_i, axis=1)
        err_t_series = psi_i * np.transpose([denom_i])
        Fi = np.average(err_t_series, axis=0)
        err_t_series = np.dot(
            (psi_i * np.transpose([denom_i]) - Fi), dBdF[i, nbrs_i])
        if gdata is not None:
            for n, g_n in enumerate(gdata):
                g_ni = g_n[i]
                dBdg_n = dBdg[n]
                g_ni_wtd = g_ni * denom_i
                err_t_series += dBdg_n[i] * (g_ni_wtd - np.average(g_ni_wtd))
        if iat_routine is not None:
            iat, mn, sigma = iat_routine(err_t_series)
            iats[i] = iat
        else:
            iat = iats[i]
            sigma = np.std(err_t_series) * np.sqrt(iat / len(err_t_series))
        sigmas[i] = sigma
    return iats, sigmas**2
