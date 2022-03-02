# -*- coding: utf-8 -*-
"""
Example script with basic usage of the EMUS package for MBAR estimates
using the iterative EMUS solver.
(for simplicity we have moved all plotting commands to the bottom of the script).
Please note that the demo requires matplotlib, which is not a dependency in the emus package to keep things lightweight.
"""
import numpy as np
from emus import usutils as uu
from emus import emus, iter_avar
import matplotlib.pyplot as plt

# Define Simulation Parameters
T = 310                             # Temperature in Kelvin
k_B = 1.9872041E-3                  # Boltzmann factor in kcal/mol
kT = k_B * T
meta_file = 'cv_meta.txt'         # Path to Meta File
dim = 1                             # 1 Dimensional CV space.
period = 360                        # Dihedral Angles periodicity
nbins = 60                          # Number of Histogram Bins.

# Load data
psis, cv_trajs, neighbors = uu.data_from_meta(
    meta_file, dim, T=T, k_B=k_B, period=period)

# Calculate the partition function for each window
z_iter, F_iter = emus.calculate_zs(psis, neighbors=neighbors, n_iter=5)

plt.plot(np.arange(len(z_iter)), -np.log(z_iter), label='Window Free Energies')
plt.xlabel(r'Window Index')
plt.ylabel('Window FE Unitless')
plt.legend()
plt.show()

domain = ((-180.0, 180.))            # Range of dihedral angle values
# Calculate new PMF
iterpmf, edges = emus.calculate_pmf(
    cv_trajs, psis, domain, nbins=nbins, z=z_iter, kT=kT)

fdata = [((traj > 25) & (traj < 100)).flatten() for traj in cv_trajs]
prob_C7ax_iter = emus.calculate_obs(
    psis, z_iter, fdata, use_iter=True)  # Just calculate the probability

fe_C7ax = -np.log(prob_C7ax_iter)
fe_C7ax_err, fe_C7ax_contribs, fe_C7ax_taus = iter_avar.calc_log_avg_ratio(psis, z_iter, fdata, neighbors=neighbors)

avg_pmf, edges = emus.calculate_avg_on_pmf(
    cv_trajs, psis, (-180, 180), z_iter, fdata, use_iter=True, nbins=nbins)


# ~~~ Data Output Section ~~~ #
# Plot the EMUS, Iterative EMUS pmfs.
pmf_centers = (edges[0][1:]+edges[0][:-1])/2.
plt.figure()
plt.plot(pmf_centers, iterpmf, label='Iter EMUS PMF')
plt.xlabel(r'$\psi$ dihedral angle')
plt.ylabel('Unitless FE')
plt.legend()
plt.title('Iterative EMUS potentials of Mean Force')
plt.show()

# Print the C7 ax basin probability
print("Iterative EMUS Free Energy of C7ax basin is %f +/- %f" % (fe_C7ax, fe_C7ax_err))

fig, (ax1, ax2) = plt.subplots(2)
ax1.semilogy(np.arange(len(z_iter)), fe_C7ax_contribs, c="C0")
ax2.semilogy(np.arange(len(z_iter)), fe_C7ax_taus, c="C1")
ax2.set_xlabel(r'Window Index')
ax1.set_ylabel('Contribs to Error')
ax2.set_ylabel('ACTimes')
plt.show()
