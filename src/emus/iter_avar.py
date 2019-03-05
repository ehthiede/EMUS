# -*- coding: utf-8 -*-
"""
Library with routines associated with the asymptotic variance for iterative EMUS.
"""

from __future__ import absolute_import
import numpy as np


def calc_p_traj(psis, z):
    """
    Estimates the trajectories :math:`p_j(X_t^i)` described in REF.

    Parameters
    ----------
    psis : 3D data structure
        The values of the bias functions evaluated each window and timepoint.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.
    z : 1D array
        Array containing the normalization constants calculated using Iterative EMUS

    Returns
    ------
    p_traj : list of 2d arrays
        Values of the P trajectory. p_traj[i][t,j] gives the value of :math:`p_j(X_t^i)`.
    """
    return


def calc_B_matrix(p_traj):
    """
    Estimates the B matrix in REF.

    Parameters
    ----------
    p_traj : list of 2d arrays
        Values of the P trajectory. p_traj[i][t,j] gives the value of :math:`p_j(X_t^i)`.

    Returns
    -------
    B : 2d matrix
        B matrix in REF.
    """
    return


def calc_log_z(psis, z, repexchange=False):
    """
    Calculates the asymptotic variance in the :
    """
    p_traj = calc_p_traj(psis, z)
    B = calc_B_matrix(p_traj)

    # Construct trajectories for autocovariance.
    B_pseudo_inv = np.linalg.pinv(B)
    zeta_traj = [np.dot(p_traj_i, B_pseudo_inv.T) for p_traj_i in p_traj]

    if repexchange:
        # ## FINISH! ## #
        return
    else:
        # ## FINISH! ## #
        return
