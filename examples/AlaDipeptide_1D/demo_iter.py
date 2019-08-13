# -*- coding: utf-8 -*-
import numpy as np
from emus import usutils as uu
from emus import emus, iter_avar_2, avar
import matplotlib.pyplot as plt


# Define Simulation Parameters
T = 310                             # Temperature in Kelvin
k_B = 1.9872041E-3                  # Boltzmann factor in kJ/mol
kT = k_B * T
kT /= 4.184                         # Convert to kCal/mol
meta_file = 'cv_meta_0.txt'         # Path to Meta File
dim = 1                             # 1 Dimensional CV space.
period = [2 * np.pi]                  # Dihedral Angles periodicity
ss = 100


# Load data
psis, cv_trajs, neighbors = uu.data_from_meta(
    meta_file, dim, T=T, k_B=k_B, period=period)
psis = [pi[::100] for pi in psis]
cv_trajs = [cvi[::100] for cvi in cv_trajs]


zs_ref = np.array([np.load('./z_iter_%d.npy' % i) for i in range(20)])
z_ref_avar = np.var(zs_ref, axis=0)

old_zs_ref = np.array([np.load('./z_%d.npy' % i) for i in range(20)])
old_z_ref_avar = np.var(old_zs_ref, axis=0)

# Calculate the partition function for each window
z_iter, F_iter = emus.calculate_zs(psis, n_iter=10)
z, F = emus.calculate_zs(psis, n_iter=0)

z_err_iter, log_zcontribs_iter, log_ztaus_iter = iter_avar_2.calc_log_z(
    psis, z_iter, repexchange=False)

z_err, log_zcontribs, log_ztaus = avar.calc_partition_functions(
    psis, z, F, repexchange=False)

plt.plot(z_ref_avar, label='True Variance')
plt.plot(z_err_iter, label='Estimated AVAR')
plt.plot(z_err, label='Old AVAR estimate')
plt.plot(old_z_ref_avar,label='True old avar')
plt.ylabel('Variance in $z$')
plt.xlabel('Window Index')
plt.yscale('log')
plt.legend()

plt.savefig("adp_variance_estimation2.png")
plt.savefig("adp_variance_estimation2.pdf")
plt.show()


plt.plot(np.sqrt(z_ref_avar)/z_iter, label='True Variance')
plt.plot(np.sqrt(z_err_iter)/z_iter, label='Estimated AVAR')
plt.plot(np.sqrt(z_err)/z, label='Old AVAR estimate')
plt.plot(np.sqrt(old_z_ref_avar)/z,label='True old avar')
plt.ylabel('STDev in $-log(z)$')
plt.xlabel('Window Index')
# plt.yscale('log')
plt.legend()

plt.savefig("adp_stdev_estimation2.png")
plt.savefig("adp_stdev_estimation2.pdf")
plt.show()
