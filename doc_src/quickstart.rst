Quickstart 
==========

This guide covers how to perform typical tasks with the EMUS package in a pythonic environment.  The required data files, as well as a (slightly modified) script containing all of the python commands used below, can be found in the examples directory of the package in the AlaDipeptide_1D directory. The guide will make use of the numpy and matplotlib packages. 

Note that most common functionality of the EMUS package can be accessed from command line using the wemus.py script.  To see a full list of command line options, use the command

>>> wemus.py --help

.. The wemus script has a syntax similar to the WHAM script by Grossfield.  The command 
   
   >>> python wemus.py 1 wham_meta.txt -180 180 60 -f fxn_meta.txt -p 360 -T 310.0 -k 'kCal' --ext txt -e acor 
   
   runs EMUS analysis on the data specified by the wham_meta and fxn_meta files.  This command specifies that the collective variable data is one dimensional, and is located at the locations specified in wham_meta.txt.  The collective variable ranges from -180 to 180 degrees, and the pmf is requested with 60 histogram bins.  T

Loading from WHAM-like Formats
------------------------------
The usutils module provides a method that loads data in the format used by WHAM.  It outputs the trajectory in collective variable space as well as the :math:`\psi_ij(x_n)` values.

>>> import numpy as np                  
>>> import matplotlib.pyplot as plt
>>> from emus import usutils as uu
>>>
>>> # Define Simulation Parameters
>>> T = 310                             # Temperature in Kelvin
>>> meta_file = 'wham_meta.txt'         # Path to Meta File
>>> dim = 1                             # 1 Dimensional CV space.
>>> period = 360                        # Dihedral Angles periodicity
>>>
>>> # Load data
>>> psis, cv_trajs, neighbors = uu.data_from_WHAMmeta('wham_meta.txt',dim,T=T,period=period)

Calculating the PMF
-------------------
We now import the emus code, and calculate the normalization constants. 

>>> from emus import emus
>>> z, F = emus.calculate_zs(psis,neighbors=neighbors,) 

To calculate the potential of mean force, we provide the number of histogram bins and the range of the collective variable, and call the appropriate method of the EMUS object.

>>> domain = ((-180.0,180.))            # Range of dihedral angle values
>>> T = 310                             # Temperature in Kelvin
>>> k_B = 1.9872041E-3                  # Boltzmann factor in kcal/mol
>>> kT = k_B * T
>>> pmf = emus.calculate_pmf(cv_trajs,psis,domain,z,nbins=60,kT=kT)   # Calculate the pmf

We can now plot the potential of mean force using pyplot or other tools.  Note that this returns the unitless free energy by default: the user can either multiply the pmf by :math:`k_B T` in postprocessing, or specify :math:`k_B T` as a parameter for calc_pmf.

>>> centers = np.linspace(-177,177,60)  # Center of the histogram bins
>>> plt.plot(centers,pmf)
>>> plt.show()

Estimating Window Partition Functions
-------------------------------------

The EMUS package also has the ability to calculate the relative partition functions from the MBAR estimator.  This requires solving a self-consistent iteration.  The nMBAR parameter specifies the maximum number of iterations.  Note that truncating early still provides a consistent estimator, and introduces no systematic bias.

>>> z_MBAR_1, F_MBAR_1 = emus.calculate_zs(psis,nMBAR=1)
>>> z_MBAR_2, F_MBAR_2 = emus.calculate_zs(psis,nMBAR=2)
>>> z_MBAR_5, F_MBAR_5 = emus.calculate_zs(psis,nMBAR=5)
>>> z_MBAR_1k, F_MBAR_1k = emus.calculate_zs(psis,nMBAR=1000)

We can plot the unitless window free energies for each max iteration number to see how our estimates converge.

>>> plt.plot(-np.log(EM.z),label="Iteration 0")
>>> plt.plot(-np.log(z_MBAR_1),label="Iteration 1")
>>> plt.plot(-np.log(z_MBAR_2),label="Iteration 2")
>>> plt.plot(-np.log(z_MBAR_5),label="Iteration 5")
>>> plt.plot(-np.log(z_MBAR_1k),label="Iteration 1k")
>>> plt.show()

The pmf can be constructed using these values for the relative partition functions. [#estimatornote]_

>>> MBARpmf = emus.calculate_pmf(cv_trajs,psis,domain,nbins=60,z=z_MBAR_1k,kT=kT)

Calculating Averages
--------------------
It is possible to use the EMUS package to calculate the averages of functions.  Here, we will calculate the probability that the dihedral takes values between 25 and 100 degrees (this roughly corresponds to the molecule being in the C7 axial basin).  This is equivalent to the average of an indicator function that is 1 if the molecule is in the desired configuration and 0 otherwise.  First, we construct the timeseries of this function for each window.  Note that if the EMUS object was constructed with the collective variable trajectories, they are contained at :samp:`EM.cv_trajs`. [#boolnote]_

>>> fdata =  [(traj>25) & (traj<100) for traj in cv_trajs]

We can now calculate the probability of being in this state. 

>>> prob_C7ax = EM.calculate_obs(fdata)
>>> print prob_C7ax

The EMUS package also introduces a new meta file for functions of configuration space.  The format is a simple text file, where the i'th line is the path to the function data collected in window i.

>>> data/fdata_0.txt
>>> data/fdata_1.txt
>>> data/fdata_2.txt
>>> ...

In each of the data files, the first column is the timestamp, and each successive column is the value of the n'th function at that timestep.  The data can be loaded using a method in usutils

>>> fxndata = uu.data_from_fxnmeta('fxn_meta.txt')

The format of the data is similar to psis.  The first index is the index of the umbrella where the data was collected.  The second is the timepoint of the data, and the third index corresponds to the function.

>>> fxndata[0]  # Data collected in the first window.
>>> fxndata[fxn_i[:,0] for fxn_i in fxndata] # First function evaluated in each window.

.. [#estimatornote] Technically speaking, this is mixing estimators: the :math:`z`'s are being estimated using the MBAR estimator, whereas the pmf is estimated using the MBAR :math:`z`'s, but with the EMUS estimator (MBAR iteration 0).  However, in practice the majority of the error comes from estimating the normalization constant.  Consequently, the estimator used for estimating the pmf affects the results much less. 

.. [#boolnote] The following code relies on True and False having integer comprehensions as 1 and 0, respectively.  If, hypothetically speaking, the programmer has done something like in :samp:`False=5` earlier in the code, then this won't give the correct answer.  Then again, if the programmer did this they are probably a bad person.  
