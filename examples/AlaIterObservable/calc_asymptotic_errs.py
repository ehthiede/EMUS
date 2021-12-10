# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:05:57 2019
"""
import numpy as np
import sys
from emus import emus, iter_avar


def get_avg_components(z, psis, g1data, g2data=None, neighbors=None):
    L = len(psis)
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
    return g1, g2


def main():
    T = 310                             # Temperature in Kelvin
    k_B = 1.9872041E-3               # Boltzmann factor in kcal/mol
    kT = k_B * T
    kT /= 4.184                         # Convert to kCal/mol
    rank = int(sys.argv[1])

    output_dir = "All_Zs_Avgs/rank_%d/" % rank
    psis = np.load(output_dir + "psis_iter_estimate_%d_iter.npy" % rank)
    cv_trajs = np.load(output_dir + "cv_trajs_iter_estimate_%d_iter.npy" % rank)
    z = np.load(output_dir + "z_iter_estimate_%d_iter.npy" % rank)
    fdata = [((traj > 0.436) & (traj < 1)).flatten() for traj in cv_trajs]

    __, __, __, avar_est = iter_avar.calc_avg_avar(psis, z, fdata, combine='ratio')
    __, __, __, avar_log_est = iter_avar.calc_avg_avar(psis, z, fdata, combine='log_ratio')

    estimate_str = 'All_Zs_Avgs/rank_%d/f_iter_estimate_%d_iter.npy'
    all_f_avgs = np.array([np.load(estimate_str % (r, r)) for r in range(21)])
    fa_var = np.var(all_f_avgs)
    log_fa_var = np.var(np.log(all_f_avgs))

    print('Estimated Asymptotic Avar: ', avar_est)
    print('Avar over Replicates: ', fa_var)

    print('Estimated Asymptotic Avar (log ratio): ', avar_log_est)
    print('Avar over Replicates (log ratio): ', log_fa_var)


if __name__ == "__main__":
    main()
