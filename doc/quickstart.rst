Quickstart
==========

This guide covers perform typical tasks with the EMUS package.  The required data files, as well as a (slightly modified) script containing all of the python commands used below, can be found in the examples directory of the package in the AlaDipeptide_1D directory. The guide will make use of the numpy and matplotlib packages. 

Loading from WHAM-like Formats
------------------------------
The usutils module provides a method that loads data in the format used by WHAM.  It outputs the trajectory in collective variable space as well as the :math:`\psi_ij(x_n)` values.

>>> import numpy as np                  
>>> import matplotlib.pyplot as plt
>>> import usutils as uu
>>>
>>> # Define Simulation Parameters
>>> T = 310                             # Temperature in Kelvin
>>> meta_file = 'wham_meta.txt'         # Path to Meta File
>>> dim = 1                             # 1 Dimensional CV space.
>>> period = 360                        # Dihedral Angles periodicity
>>>
>>> # Load data
>>> psis, cv_trajs = uu.data_from_WHAMmeta('wham_meta.txt',dim,T=T,period=period)

Calculating the PMF
-------------------
We can now build the EMUS object (this automatically calculates the relative normalization constants according to the first EMUS iteration). 

>>> from emus import emus
>>> EM = emus(psis,cv_trajs)

To calculate the potential of mean force, we provide the number of histogram bins and the range of the collective variable, and call the appropriate method of the EMUS object.

>>> domain = ((-180.0,180.))            # Range of dihedral angle values
>>> pmf = EM.calc_pmf(domain,nbins=60)   # Calculate the pmf

We can now plot the potential of mean force using pyplot or other tools.  Note that this returns the unitless free energy by default: the user can either multiply the pmf by :math:`k_B T` in postprocessing, or specify :math:`k_B T` as a parameter for calc_pmf.

>>> centers = np.linspace(-177,177,60)  # Center of the histogram bins
>>> plt.plot(centers,pmf)
>>> plt.show()

Estimating Window Partition Functions
-------------------------------------
Upon creation, the EMUS object already estimates the relative partition function (denoted :math:`z`) of each window using the EMUS estimator.  These are contained in the object, and can be accessed directly.

>>> print EM.z

The EMUS object also has the ability to calculate the relative partition functions from the MBAR estimator.  This requires solving a self-consistent iteration.  The nMBAR parameter specifies the maximum number of iterations.  Note that truncating early still provides a consistent estimator, and introduces no systematic bias.

>>> z_MBAR_1, F_MBAR_1 = EM.calc_zs(nMBAR=1)
>>> z_MBAR_2, F_MBAR_2 = EM.calc_zs(nMBAR=2)
>>> z_MBAR_5, F_MBAR_5 = EM.calc_zs(nMBAR=5)
>>> z_MBAR_1k, F_MBAR_1k = EM.calc_zs(nMBAR=1000)

We can plot the unitless window free energies for each max iteration number to see how our estimates converge.

>>> plt.plot(-np.log(EM.z),label="Iteration 0")
>>> plt.plot(-np.log(z_MBAR_1),label="Iteration 1")
>>> plt.plot(-np.log(z_MBAR_2),label="Iteration 2")
>>> plt.plot(-np.log(z_MBAR_5),label="Iteration 5")
>>> plt.plot(-np.log(z_MBAR_1k),label="Iteration 1k")
>>> plt.show()

The pmf can be constructed using these values for the relative partition functions. [#estimatornote]_

>>> MBARpmf = EM.calc_pmf(domain,nbins=60,z=z_MBAR_1k)


.. [#estimatornote] Technically speaking, this is mixing estimators: the :math:`z`'s are being estimated using the MBAR estimator, whereas the pmf is estimated using the MBAR :math:`z`'s, but with the EMUS estimator (MBAR iteration 0).  However, in practice the majority of the error comes from estimating the normalization constant.  Consequently, the estimator used for estimating the pmf affects the results much less. 

