# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:05:57 2019
"""
import numpy as np
import os
import sys
from emus import usutils as uu
from emus import emus


def data_from_meta(filepath, run, dim, T=310, k_B=1.9872041E-3, nsig=None, period=None, subsample=1, trunc=None):
    """Reads collective variable data from as tabulated by a meta file of the same format used in Grossfield's implementation of the WHAM algorithm, and calculates the value of the biasing functions.

    Parameters
    ----------
    filepath : string
        The path to the meta file.
    dim : int
        The number of dimensions of the cv space.
    T : scalar, optional
        Temperature of the system if not provided in the meta file.
    k_B : scalar, optional
        Boltzmann Constant for the system. Default is in natural units (1.0)
    nsig : scalar or None, optional
        Number of standard deviations of the gaussians to include in the neighborlist.If None, does not use neighbor lists.
    period : 1D array-like or float, optional
        Variable with the periodicity information of the system.  See the Data Structures section of the documentation for a detailed explanation.

    Returns
    -------
    psis : List of 2D arrays
        The values of the bias functions evaluated each window and timepoint.  See `datastructures <../datastructures.html#data-from-sampling>`__ for more information.

    """
    # Parse Meta file.
    _, cntrs, fks, iats, temps = uu._parse_metafile(filepath, dim)
    L = len(cntrs)
    # Calculate kT for each window.  Involves some type management...
    temps = np.ones(L)*T
    kT = k_B * temps
    if nsig is not None:
        neighbors = uu.neighbors_harmonic(cntrs, fks, kTs=kT, period=period, nsig=nsig)
    else:
        neighbors = np.outer(np.ones(L), range(L)).astype(int)

    # Load in the trajectories into the cv space
    trajs = []
    for i in range(L):
        traj_i = np.loadtxt('./data/run'+str(run)+'/cvum%d.txt' % i)[:, 1]
        print(traj_i.shape)
        if trunc is not None:
            traj_i = traj_i[-trunc:]
        print(traj_i.shape)
        trajs.append(traj_i[::subsample])

    # Calculate psi values
    psis = []
    for i, traj in enumerate(trajs):
        nbrs_i = neighbors[i]
        psi_i = uu.calc_harmonic_psis(traj, cntrs[nbrs_i], fks, kT, period=period)
        psis.append(psi_i)

    return psis, trajs, neighbors


def main():
    T = 310                             # Temperature in Kelvin
    k_B = 1.9872041E-3               # Boltzmann factor in kcal/mol
    kT = k_B * T
    kT /= 4.184                         # Convert to kCal/mol
    dim = 1                             # 1 Dimensional CV space.
    period = 2 * np.pi                        # Dihedral Angles periodicity
    subsample = 100
    trunc = None
    rank = int(sys.argv[1])

    meta_file = './cv_meta_modified.txt'
    psis, cv_trajs, neighbors = data_from_meta(
        meta_file, rank, dim, T=T, k_B=k_B, period=period, subsample=subsample, trunc=trunc)
    z_iter, F_iter = emus.calculate_zs(psis, neighbors=neighbors, n_iter=5)
    fdata = [((traj > 0.436) & (traj < 1)).flatten() for traj in cv_trajs]
    favg = emus.calculate_avg(psis, z_iter, fdata)
    output_dir = "All_Zs_Avgs/rank_%d/" % rank
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    np.save(output_dir + "z_iter_estimate_%d_iter.npy" % rank, z_iter)
    np.save(output_dir + "F_iter_estimate_%d_iter.npy" % rank, F_iter)
    np.save(output_dir + "f_iter_estimate_%d_iter.npy" % rank, favg)
    np.save(output_dir + "psis_iter_estimate_%d_iter.npy" % rank, psis)
    np.save(output_dir + "cv_trajs_iter_estimate_%d_iter.npy" % rank, cv_trajs)
    print(rank)


if __name__ == "__main__":
    main()
