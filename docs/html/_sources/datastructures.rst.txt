Data Structures
===============

This section discusses commonly used variables and naming conventions used throughout the EMUS package. These are not strict standards: the EMUS package attempts to duck-type as much as possible to interpret user input.  The documentation below is intended to indicate approximately what form inputs can take, and to aid comprehension of the source code.

Common data structures
----------------------
:cv_trajs: Two or three-dimensional array-like containing the trajectories in the collective variable space.  The first dimension corresponds to the window index, the second to the timepoint in the trajectory, and the optional third to the collective variable.

           >>> cv_trajs[i] # CV trajectory of the i'th window
           >>> cv_trajs[i][n] # cv coordinate in window i, time n. 
           
          Note the trajectories need not have the same number of time points: :samp:`cv_trajs[i]` and :samp:`cv_trajs[j]` may have different lengths.
:psis: Three-dimensional array-like with the values of :math:`\psi_j` evaluated at each point in the trajectory.  As above, the first and second indices are the window index and timepoint, respectively.  The third index is the window where :math:`\psi` is being evaluated: :samp:`psis[i][n][j]` returns the value of :math:`\psi_j\left(X_n^i\right)`.  If the neighborlist functionality is used, the third index does not need to span over all of the windows, only nearby ones (Note: using neighborlists can introduces systematic bias into the estimator.  However, for intelligently chosen neighborlists, this bias should be negligible.).
:gdata: Two-dimensional array-like containing the values of an observable.  The data structure is similar to cv_trajs.

        >>> gdata1[i][n] # observable at window i at timepoint n.
:neighbors: Two-dimensional array-like used for the neighborlist functionality.  In practice, it is often known in advance that many combinations of i and j will result in values of :math:`\psi` that are zero or effectively zero (for instance, two harmonic windows far away in collective variable space).  In these situations, it is cheaper only to calculate and store psis for neighboring windows, and :samp:`psis[i][n][j]` will return :math:`\psi` evaluated for the j'th *neighbor*.  The :samp:`neighbors` data element then gives the true indeces of each neighbor: :samp:`neighbors[i][j]` returns the index of the j'th window neighboring window i.  Below is an example neighborlist for nearest neighbors in 1D, aperiodic space.

        >>> neighbors = [[0,1],[0,1,2],[1,2,3],[2,3,4],[3,4,5],[4,5]]

        For harmonic windows, a neighborlist can be constructed using the neighbors_harmonic function.
:iats: One-dimensional array-like giving the integrated autocorrelation time (iat) for the values of :math:`\psi` each window.  :samp:`iats[i]` gives the iat for window i. [#iatnote]_

Parameters for Harmonic windows
---------------------------------
:centers: Two-dimensional array-like containing the center of each harmonic window in collective variable space.  The first index corresponds to the window index, and the second to the collective variable index.
          
          >>> centers = np.array([[180.,180.],[180.,160.],[180.,140.]]) 
          >>> centers[1] # Returns array with the center of the middle window
          >>> centers[:,1] # Returns each window's y-coordinate.
:fks: Two-dimensional array-like containing the force constant for each window.  The syntax and format is the same as for centers.
:kTs: One-dimensional array-like with the Boltzmann factor for each window, where the index corresponds to the window index.  Alternatively, if the Boltzmann factor is the same for all windows, EMUS will accept a scalar value for.
:period: One-dimensional array-like encoding the period of the collective variable.  :samp:`period[i]` gives the periodicity of the i'th collective variable.  If the periodicity in that dimension is none, a value of None is used.  For instance, the command below specifies that the first collective variable has a period of 360, the second has one of 1, and that the third is aperiodic. 

         >>> period = np.array([360.,1,None])



.. [#iatnote] Claiming that there is a single iat for all :math`\psi` is a
   polite mathematical fiction, as different :math:`\psi` values will have 
   different integrated autocorrelations times.  However, it is reasonable 
   to expect :math:`psi` values to have comparable iats, and estimating
   multiple iats is numerically challenging.  Consequently, we use one iat
   for all.
