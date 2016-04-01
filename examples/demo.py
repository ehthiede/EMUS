import numpy as np
import emus
import umbutils as uu
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt

# Parameters
metafile = '1dwhammeta.txt'     # Location of the Meta File
histinfo = (-180.,180.,100)     # Range to histogram over
period=360                      # Period of the CV
window_1 = 5                         # First Interesting Window
window_2 = 13                        # Second Interesting Window
savestring = 'ala_dipeptide_pmf.txt' # String to save the PMF to.
kT = 310*1.9872041*10**-3 # Boltzmann Constant, given in kcal/mol

# Calculate the value of Psi for each trajectory.
# Load in the meta file
def main():
    # Parse Wham Meta file.
    trajlocs, ks, cntrs, corrts, temps  = uu.parse_metafile(metafile,1)

    # Load in the trajectories into the cv space
    trajs = []
    for i, trajloc in enumerate(trajlocs):
        trajs.append(np.loadtxt(trajloc)[:,1:]) 

    # Calculate psi values
    psis = []
    for i,traj in enumerate(trajs):
        psis.append(uu.calc_harmonic_psis(traj,cntrs,ks,kT,period=period))

    EM = emus.emus(psis,trajs)
    z,F = EM.calc_zs()
    wfes = -np.log(z)
    print "# Unitless free energy for each window, -ln(z):"
    for i, wfes_i in enumerate(wfes):
        print "Window_FE: %d %f"%(i,wfes_i)

    # Calculate asymptotic error, importances
    errs  = EM.avar_zfe(window_1,window_2)
    print "Estimated Free Energy Windows %d to %d, Asymptotic Variance:"%(window_1,window_2)
    print "Delta_FE: ",wfes[window_2]-wfes[window_1], np.dot(errs,errs)
    imps = errs * np.array([len(traj) for traj in trajs])
    imps *= len(imps)/np.sum(imps)
    for i, imp_i in enumerate(imps):
        print "Window_imp: %d %f"%(i,imp_i)

    # Calculate the PMF
    domain = (-180.,180.)
    pmf = EM.pmf(domain,kT=kT)
    xax = np.linspace(domain[0],domain[1],len(pmf)+1)
    xax = (xax[1:]+xax[:-1])/2 # Hist midpoints
    print "Saving PMF to %s" % savestring
    np.savetxt('ala_dipeptide_pmf.txt',zip(xax,pmf))
    plt.plot(xax,pmf)
    plt.show()

if __name__ == "__main__":
    main()

