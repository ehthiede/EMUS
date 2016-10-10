Data Structures
===============

This section discusses commonly used variables and naming conventions used throughout the EMUS package. These are not strict standards: the EMUS package attempts to duck-type as much as possible to interpret user input.  The documentation below is intended to indicate approximately what form inputs can take, and to aid comprehension of the source code.

Common data structures
----------------------
:cv_trajs: Two-dimensional array-like containing the trajectories in the collective variable space.  The first dimension corresponds to the state index and the second to the timepoint.  :samp:`cv_trajs[i]` gives the the collective variable trajectory of the i'th state, and :samp:`cv_trajs[i][n]` gives the value of the collective variable of state i at the nth time point.  Note the trajectories can have the same number of time points: :samp:`cv_trajs[i]` and :samp:`cv_trajs[j]` may have different lengths.
:psis: Three-dimensional array-like with the values of :math:`\psi_j` evaluated at each point in the trajectory.  As above, the first and second indices are the state index and timepoint, respectively.  The third index is the state where :math:`\psi` is being evaluated: :samp:`psis[i][n][j]` returns the value of :math:`\psi_j\left(X_n^i\right)`.  If the neighborlist functionality is used, the third index does not need to span over all of the windows, only nearby ones (this introduces slight systematic bias into the estimator).
:gdata: Two-dimensional array-like containing the values of an observable.  The data structure is similar to cv_trajs: :samp:`fdata[i][n]` gives the value of the observable evaluated in state i at timepoint n. 
:neighbors: Two-dimensional array-like used for the neighborlist functionality.  In practice, it is often known in advance that many combinations of i and j will result in values of :math:`\psi` that are effectively zero (for instance, two harmonic windows far away in collective variable space).  In these situations, it is cheaper to only calculate and store psis for neighboring states, and :samp:`psis[i][n][j]` will return the return :math:`\psi` for the j'th *neighbor*.  The :samp:`neighbors` data element then gives the true indeces of each neighbor: :samp:`neighbors[i][j]` returns the index of the j'th state neighboring state i.  For harmonic windows, a neighborlist can be constructed using the neighbors_harmonic function.
:iats: One-dimensional array-like giving the integrated autocorrelation time for the values of :math:`\psi` each state (iat).  :samp:`iats[i]` gives the iat for state i.[#iatnote]_

Parameters for Harmonic windows
---------------------------------
:centers: Two-dimensional array-like containing the center of each harmonic window in collective variable space.  The first index corresponds to the window index, and the second to the collective variable index: :samp:`centers[2]` gives the coordinate of the center of the third window.
:fks: Two-dimensional array-like containing the force constant for each window.  The syntax and format is the same as for centers.
:kTs: One-dimensional array-like with the Boltzmann factor for each window, where the index corresponds to the window index.  Alternatively, if the Boltzmann factor is the same for all windows, EMUS will accept a scalar value for.
:period: One-dimensional array-like encoding the period of the collective variable.  :samp:`period[i]` gives the periodicity of the i'th collective variable.  If the periodicity in that dimension is none, a value of None is used.  For instance, :samp:`period=[360.,1,None]`  specifies that the first collective variable has a period of 360, the second has one of 1, and that the third is aperiodic. 



.. [#iatnote] Claiming that there is a single iat for all :math`\psi` is a
   polite mathematical fiction, as different :math:`\psi` values will have 
   different integrated autocorrelations times.  However, it is reasonable 
   to expect :math:`psi` values to have comparable iats, and estimating
   multiple iats is numerically challenging.  Consequently, we use one iat
   for all.
