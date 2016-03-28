import numpy as np
import argparse
import usroutines as usr


# Default Parameters
#_default_kB = 1.9872041*10**-3 # Boltzmann Constant, given in kcal/mol
#_default_T = 310.0 # Default Temperature in Kelvin.

class emus:
    """
    Class containing methods and data for the EMUS algorithm.
    """


    def __init__(self,psitrajs,xtrajs=None,neighborlist=None,ftrajs=None, 
            kB=1.0,T=1.0,tol=1.0E-8):
        self.psitrajs = psitrajs
        if neighborlist is not None:
            self.neighborlist = neighborlist
        else:
            L = len(psitrajs)
            self.neighborlist = np.outer(np.ones(L),np.arange(L)).astype(int)
            print "-------------------------------------"
            print self.neighborlist
            print "-------------------------------------"
        self.xtrajs = xtrajs 
        self.ftrajs = ftrajs
        self.kB = kB
        self.T = T
        self.tol = tol
        self.z = None
        self.EMUS_F = None
        self.MBAR_F = None

    def calc_zs(self,zguess=None,tol=1.E-8,npolish=0,usetaus=True,taus=None):
        """
        Calculates the z values for 
        

        """
#        self.EMUSzs = zs
        L = len(self.psitrajs) # Number of Windows
        Npnts = np.array([len(psitrajs_i) for psitrajs_i in self.psitrajs])
        Npnts /= np.max(Npnts)
        if zguess is None: # Perform Initial EMUS iteration
            Amguess = np.ones((L,L))
            if taus is not None:
                taumat = np.outer(np.ones(L),taus)
                zguess, F = usr.emus_iter(self.psitrajs,neighbors=self.neighborlist,return_taus=False)
            else:
                if usetaus is True:
                    zguess, F, taumat = usr.emus_iter(self.psitrajs,Amguess,neighbors=self.neighborlist,return_taus=True)
                    print "Nontrivial Taumat"
                    taudiag = np.diag(taumat)
                    taumat = np.outer(np.ones(L),taudiag)
                else:
                    zguess, F = usr.emus_iter(self.psitrajs,neighbors=self.neighborlist,return_taus=False)
                    taumat = np.ones((L,L))

        self.EMUS_F = F
        z_new = zguess
        # we perform the self-consistent polishing iteration
        z_old = zguess
        for n in xrange(npolish):
            print z_old, "z_old start iter"
            Apart = Npnts/z_old
            Amat = np.outer(np.ones(L),Apart)
            # Uncommented Code
            print np.max(taumat), np.min(taumat), "Taumat"
            if usetaus is True:
                Amat /= taumat
            z_new, F_new, taumat = usr.emus_iter(self.psitrajs,Amat,neighbors=self.neighborlist,return_taus=True)
            taudiag = np.diag(taumat)
            taumat = np.outer(np.ones(L),taudiag)
            # Check if we have converged.
            print "z"
            print z_new, z_old
            print "z"
            if np.max(np.abs(z_new-z_old)/z_old) < tol:
                break
            z_old = z_new
            print n
        self.z = z_new
        return z_new

    def calc_obs(self,fdata):
        """
        
        """
        favg = usr.calc_obs(self.psitrajs,self.z,self.EMUS_F,fdata)
        return favg

    def asymptotic_var_zfe(self,um1,um2,iat='ipce'):
        """
        Calculates the asymptotic variance for the free energy difference
        between windows indexed um1 and um2
        """
        errs, taus = usr.avar_zfe(self.psitrajs,self.neighborlist,self.z,self.EMUS_F,um1,um2,iat=iat)
        return errs, taus

    def pmf(self,domain,nbins=100):
        """
        Calculates the potential of mean force for the system
        """
        pmf = usr.makeFEsurface(self.xtrajs,self.psitrajs,domain,self.z,kT=self.kB*self.T)
        return pmf
