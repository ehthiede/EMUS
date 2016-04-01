import numpy as np
import argparse
import emusroutines as emr
import avar


# Default Parameters
#_default_kB = 1.9872041*10**-3 # Boltzmann Constant, given in kcal/mol
#_default_T = 310.0 # Default Temperature in Kelvin.

class emus:
    """Class containing methods and data for the EMUS algorithm.  An EMUS 
    object has the following data structures which can be interacted with 
    or modified:
        self.psis (3D array): array containing values of the biases in each 
            state.
        self.cv_trajectories (2D Array): array containing the trajectories 
            in cv space.  None if not used.
        self.z (1d array): array containing the normalization constants.
            Calculated according to the first iteration of EMUS. 
        self.F (2d array): F matrix for the first iteration of EMUS.
        self.iats (1d array): array containing integrated autocorrelation times
            of :math:`psi_{ii}(x)*` in each window.
    """

    def __init__(self,psis,cv_trajectories=None,neighborlist=None):
        """Initialize the emus object.
        Args:
            psis (3D data struct): Three dimensional data structure with the 
                trajectories of the psi values.  Expected to be either a three 
                dimensional array or a list of two dimensional numpy arrays.  
                element i,j,k corresponds to :math:`\psi_k` evaluated at
                timepoint j in state i.  If neighborlists are used, then k is 
                the index in the neighborlist, not the index in the overall 
                structure.
                
        Optional Args:
            cvtrajectories (2D list or array): Two dimensional data structure
                with the trajectories in cv space.  The first dimension is the
                state where the data was collected, and the second is the value
                in cv space.
            neighborlist (2D list or array): Two dimensional data structure.
                The first dimension is the state index, and the second is the 
                index of a neighboring state.
		
        """
        self.psis = psis
        if neighborlist is not None: # Neighborlist Provided
            self.neighborlist = neighborlist
        else:
            L = len(psis)
            self.neighborlist = np.outer(np.ones(L),np.arange(L)).astype(int)
        self.cv_trajectories = cv_trajectories 
        z,F,iats = emr.emus_iter(self.psis,neighbors=self.neighborlist,return_iats=True)
        self.z = z
        self.F = F
        self.iats = iats

    def calc_zs(self,nMBAR=0,tol=1.E-8,use_iats=False,iats=None):
        """Calculates the normalization constants for the states.
        Optional Args:
            nMBAR (int): Maximum number of MBAR iterations to perform.
            tol (float): relative residual for truncating the MBAR 
                iterations
            use_iats (bool or string): If true, estimate integrated 
                autocorrelation time in each MBAR iteration.  Likely 
                unnecessary unless dynamics are expected to be 
                drastically different in each state.
            iats (1D array): Array of size L (no. states) with values of the
                integrated autocorrelation time estimated in each state.  
                These values will be used in each iteration.  Overrides 
                use_iats.
            iat_method (string): Method to use to calculate autocorrelation.
                Default is the initial positive change estimator ('ipce', 
                Geyer, Stat. Sci. 1992), also implemented is 'icce' and 'acor'
                by Jonathan Goodman.
        Returns:
            z (1D array): Values for the Normalization constant in each state.
            F (2D array): Matrix to take the eigenvalue of for MBAR.
        Optionally Returns:
            iats (1D array): Estimated values of the autocorrelation time if
                use_iats is true.

        """
        L = len(self.psis) # Number of States
        Npnts = np.array([len(psis_i) for psis_i in self.psis])
        Npnts /= np.max(Npnts)
        if iats is None:
            iats = np.ones(L)
            if use_iats is True:
                iats = self.iats

        # we perform the self-consistent polishing iteration
        z = self.z
        F = self.F
        for n in xrange(nMBAR):
            z_old = z
            Apart = Npnts/z_old
            Amat = np.outer(np.ones(L),Apart)
            Amat /= np.outer(np.ones(L),iats)
            if use_iats:
                z, F, iats = emr.emus_iter(self.psis,Amat,neighbors=self.neighborlist,return_iats=True)
            else:
                z, F = emr.emus_iter(self.psis,Amat,neighbors=self.neighborlist)
            # Check if we have converged.	
            if np.max(np.abs(z-z_old)/z_old) < tol:
                break
				
        if use_iats:
            return z, F, iats
        else:
            return z, F

    def calc_obs(self,fdata,z=None):
        """Estimates the average of an observable function. 
        Args:
            fdata (2d data struct): Two dimensional data structure where the
                first dimension corresponds to the state index, and the second
                to the value of the observable at that time point.  Must have 
                the same number of data-points as the collective variable
                trajectory.
        Optional Args:
            z (1D array): User-provided values for the normalization constants
                If not provided, uses values from the first iteration.
                        
        Returns:
            favg (float): estimated average.
        
        """
        if z is None:
            z = self.z
        favg = emr.calc_obs(self.psis,z,fdata)
        return favg

    def avar_zfe(self,state_1,state_2):
        """Calculates the asymptotic variance for the free energy difference
        between the two states specified.
        Args:
            state_1 (int): Index of the first state.
            state_2 (ind): Index of the second state.
                
        Returns:
            errs (array): Array containing each state's contribution to the 
                asymptotic error.  The total asymptotic error is taken by
                summing the entries.
        """
        errs, iats = avar.avar_zfe(self.psis,self.neighborlist,self.z,self.F,state_1,state_2)
        return errs

    def pmf(self,domain,nbins=100,kT=1.0,z=None):
        """Calculates the potential of mean force for the system
        Args:
            domain (tuple): Tuple containing the dimensions of the space over
                which to construct the pmf, e.g. (-180,180) or 
                ((0,1),(-3.14,3.14))
        Optional Args:
            nbins (int or tuple): Number of bins to use.  If int, uses that many
                bins in each dimension.  If tuple, e.g. (100,20), uses 100 bins in
                the first dimension and 20 in the second.
            z (1D array): User-provided values for the normalization constants
                If not provided, uses values from the first iteration.	
            kT (float): Value of kT to scale the PMF by.  If not provided, set
                to 1.0
        
        Returns:
            FINISH THIS!
				
        """
        if z is None:
            z = self.z
        pmf = emr.makeFEsurface(self.cv_trajectories,self.psis,domain)
        return pmf

