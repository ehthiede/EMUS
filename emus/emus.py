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
        self.xtrajs =xtrajs 
        self.ftrajs = ftrajs
        self.kB = kB
        self.T = T
        self.tol = tol

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

        z_new = zguess
        # we perform the self-consistent polishing iteration
        z_old = zguess
        for n in xrange(npolish):
            print z_old, "z_old start iter"
            Apart = Npnts/z_old
            Amat = np.outer(np.ones(L),Apart)
            # UNcommented Code
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
#                print "Flag"
#                print tol, np.abs(z_new-z_old)/z_old
#                print z_old
#                print "Flag"
                break
            z_old = z_new
            print n
        self.z = z_new
        return z_new
        


#    def compute(self,z=True,avar_fxn=None,avar_fe=None,avar_wfe=None,niter=1
#        """
#        Interface Code the user calls to compute quantities with EMUS.  By 
#        choosing various options, the user specifies 
#        """
#    def iteration():
