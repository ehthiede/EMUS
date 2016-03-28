import numpy as np
import emus
import usroutines as usr
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

# Parameters
metafile = '1dwhammeta.txt'
histinfo = (-180.,180.,100)
period=360
kB = 1.9872041*10**-3 # Boltzmann Constant, given in kcal/mol
T = 310.
nsig = 6
um1 = 5 
um2 = 13

# Calculate the value of Psi for each trajectory.
# Load in the meta file
def main():
    trajlocs, ks, cntrs, corrts, temps  = usr.parse_metafile(metafile,1)
    L = len(trajlocs) # Number of states
    kTs = np.ones(L)*kB*T
    nbrs = usr.neighbors_harmonic(cntrs,ks,kTs,period=period,nsig=nsig) # Optional Neighborlist


    # Load in the trajectories in cv space
    trajs = []
    for i, trajloc in enumerate(trajlocs):
        trajs.append(np.loadtxt(trajloc)[:,1:]) 
    print np.shape(trajs)
    # Calculate psi values
    psis = []
    for i in xrange(L):
        nbcenters = cntrs[nbrs[i]]
        psis.append(usr.calc_harmonic_psis(trajs[i],nbcenters,ks,kTs,period=period))
    psis = np.array(psis)
    print np.shape(psis), "Shape Psis"
    EM = emus.emus(psis,trajs,neighborlist = nbrs)
    z = EM.calc_zs()
    wfes = -np.log(z)
    print "# Unitless free energy for each window, -ln(z):"
    for i, wfes_i in enumerate(wfes):
        print "Window_FE: %d %f"%(i,wfes_i)

    # Calculate asymptotic error, importances
    errs, taus  = EM.asymptotic_var_zfe(um1,um2)
    print "Estimated Free Energy, Asymptotic Variance:"
    print "Delta_FE: ",wfes[um2]-wfes[um1], np.dot(errs,errs)
    imps = errs * np.array([len(traj) for traj in trajs])
    imps *= L/np.sum(imps)
    for i, imp_i in enumerate(imps):
        print "Window_imp: %d %f"%(i,imp_i)
    domain = (-180.,180.)
    pmf = EM.pmf(domain)
    xax = np.linspace(domain[0],domain[1],100)
    plt.plot(xax,pmf)
    plt.show()

if __name__ == "__main__":
    main()

