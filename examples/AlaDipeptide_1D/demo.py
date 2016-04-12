# -*- coding: utf-8 -*-
"""
Example script with basic usage of the EMUS package.  The script follows the quickstart guide closely, with slight adjustments (for simplicity we have moved all plotting commands to the bottom of the script).
"""
import numpy as np                  
import usutils as uu
from emus import emus

import matplotlib
#matplotlib.use('Qt4Agg')
#import matplotlib.pyplot as plt

# Define Simulation Parameters
T = 310                             # Temperature in Kelvin
meta_file = 'wham_meta.txt'         # Path to Meta File
dim = 1                             # 1 Dimensional CV space.
period = 360                        # Dihedral Angles periodicity

# Load data
psis, cv_trajs = uu.data_from_WHAMmeta('wham_meta.txt',dim,T=T,period=period)

# Create the EMUS object
EM = emus(psis,cv_trajs)

# Calculate the PMF
domain = ((-180.0,180.))            # Range of dihedral angle values
pmf = EM.calc_pmf(domain,nbins=60)   # Calculate the pmf

# Calculate z using the MBAR iteration.
z_MBAR_1, F_MBAR_1 = EM.calc_zs(nMBAR=1)
z_MBAR_2, F_MBAR_2 = EM.calc_zs(nMBAR=2)
z_MBAR_5, F_MBAR_5 = EM.calc_zs(nMBAR=5)
z_MBAR_1k, F_MBAR_1k = EM.calc_zs(nMBAR=1000)
# Calculate new PMF
MBARpmf = EM.calc_pmf(domain,nbins=60,z=z_MBAR_1k)

# Estimate probability of being in C7 ax basin
fdata =  [(traj>25) & (traj<100) for traj in EM.cv_trajs]
prob_C7ax = EM.calc_obs(fdata)

### Plotting Section ###

# Plot the EMUS, MBAR pmfs.
centers = np.linspace(-177,177,60)  # Center of the histogram bins
plt.figure()
plt.plot(centers,pmf,label='EMUS PMF')
plt.plot(centers,MBARpmf,label='MBAR PMF')
plt.xlabel('$\psi$ dihedral angle')
plt.ylabel('Unitless FE')
plt.legend()
plt.title('EMUS and MBAR potentials of Mean Force')
plt.show()

# Plot the relative normalization constants as fxn of max iteration. 
plt.plot(-np.log(EM.z),label="Iteration 0")
plt.plot(-np.log(z_MBAR_1),label="Iteration 1")
plt.plot(-np.log(z_MBAR_1k),label="Iteration 1k",linestyle='--')
plt.xlabel('Window Index')
plt.ylabel('Unitless Free Energy')
plt.title('Window Free Energies and MBAR Iter No.')
plt.legend(loc='upper left')
plt.show()

# Print the C7 ax basin probability
print "Probability of C7ax basin is %f"%prob_C7ax