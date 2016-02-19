#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Given a guess for a one dimensional potential of mean force, attempts to calculate an
optimal arrangement of windows for an umbrella sampling calculation by 
minimizing the estimate for the EMUS asymptotic variance with respect
to window parameters
"""

import numpy as np
from scipy.optimize import minimize
import linalg_methods as lam

def minimize_1Davar(xcoords,pmf_guess,f_guess,L,f2_guess=None,kT=1.,periodic=False,minwidth=None,tau_scaling=-2,windowtype='harmonic',initial_centers=None):
    """
    Minimizes the asymptotic variance of an umbrella sampling calculation on a 1D potential.
    Parameters
    ----------
        xcoords : 1D iterable
            X coordinates on which the potential of mean force is defined.
        pmf_guess : 1D iterable
            Guess of the potential of mean force.  Same dimensions as xcoords
        f_guess : 1D iterable
            Guess for the values of the potential of mean force.
        L : Number of umbrellas.

    FINISH DOCUMENTATION!!!

    Check that everything still works if xcoords is decre
    asing, rather than
    increasing.
    """
    # Take care of bookkeeping first
    # Just remap the x axis to the interval 1 to len(PMF)
    newcntrs = [np.argmax(xcoord > cntr) for cntr in centers]
    #
    print newcntrs
    # Normalize PMF (i.e. set kT to 1.)


    return

    
    
    #
def avar_harmonic(pmf,f1,L,centers,widths,f2=None,periodic=False,tau_scaling=-2,bftype ='harmonic'):
    # Calculate the values of all the bias functions.

    

def get_max_barrier(xcoords,pmf,center):
    """
    Calculates the maximum free energy barrier within the limits.

    There is an implicit assumption here that the pm=f is pretty smooth over 
    the range considered, i.e. that all minima are correspond to metastable 
    regions.

    """

    # rotate the pmf so that the center is more or less in the center
    center_index = np.argmax(xcoords >= center) # find center in xcoords
    npnts = len(pmf)
    pmf = np.roll(pmf,npnts/2-center_index) # pmf is now kinda centered.
    # Trim the ends off the pmf to avoid any artifacts
    pmf -= np.min(pmf) # Zero the PMF
    
    delta_pmf = np.diff(pmf)
    # Location first minimum.
    l_desc = np.argmax(delta_pmf < 0.0) # Where we start descending
    l_min = np.argmax(delta_pmf[l_desc:] > 0.0) # Where we start ascending again
    # Location last minimum
    rdelta_pmf = delta_pmf[::-1]
    r_desc = np.argmax(rdelta_pmf > 0.0) # Where we start descending
    r_min = np.argmax(rdelta_pmf[r_desc:] < 0.0) # Where we start ascending again
    r_min = npnts-r_min # have to flip back because we flipped the array.
    print "Descending values", l_desc, r_desc

    if l_min >= r_min: # The searches met or passed each other (no. minima <= 1)
        print "left",l_min,"right",r_min
        return 0.0
    else:
        middlepmf = pmf[l_min:r_min+1]
        print l_min, r_min, xcoords[l_min], xcoords[r_min]
        print np.max(middlepmf),np.min(middlepmf)
        return np.max(middlepmf)



def gaussian_bias(xcoords,center,sigma,periodicity=None):
    """
    Returns the values of a gaussian bias function
    
    Parameters
    ----------
        xcoords : 
    """
    # Enforce any periodic boundary conditions.
    rv = xcoords-center
    if periodicity is not None:
        rv = minimage(rv,periodicity)
    potential = rv**2 / (2.*sigma**2)
    return np.exp(-potential)

def triangular_bias(xcoords,center,widthleft,widthright,periodicity=None):
    # Enforce any periodic boundary conditions.
    rv = xcoords-center
    if periodicity is not None:
        rv = minimage(rv,periodicity)
    # This code could probably stand some optimization?
    mleft = 1./widthleft
#    bleft = 1.-mleft*center
    bleft = 1.
    mright = -1./widthright
#    bright = 1.-mright*center
    bright = 1.
    triangular_lines = np.minimum(bleft+mleft*rv,bright+mright*rv)
    biasfxn = np.maximum(0,triangular_lines)
    return biasfxn

def minimage(rv,period):
    return rv - period * np.rint(rv/period)

def bf_testing():
    import matplotlib
    matplotlib.use('Qt4Agg')
    import matplotlib.pyplot as plt

    xaxis = np.arange(100)*0.1
    period = 10.
    print " Testing the guassian bias fxn"
    center =  1.
    sigma = 3.
    gaubias = gaussian_bias(xaxis,center,sigma)
    plt.plot(xaxis,gaubias)
    gaubias2 = gaussian_bias(xaxis,center,sigma,periodicity=10)
    plt.plot(xaxis,gaubias2)
    plt.show()

    print " Testing the triangular bias fxn"
    center =  1.
    wl = 0.5
    wr = 1.0
    tribias1 = triangular_bias(xaxis,center,0.5,1.2,periodicity=10)
    plt.plot(xaxis,tribias1)
    tribias2 = triangular_bias(xaxis,center+0.1,2.0,1.5)
    plt.plot(xaxis,tribias2)
    tribias3 = triangular_bias(xaxis,center,2.0,1.5,periodicity=10)
    plt.plot(xaxis,tribias3)
    plt.show()
    return

def minfinder_testing():
    import matplotlib
    matplotlib.use('Qt4Agg')
    import matplotlib.pyplot as plt
    
    xaxis = np.linspace(0.,4.,100)
    potential = np.cos(np.pi*xaxis) + np.cos(np.pi/2.*xaxis)
    potential-= min(potential)
    center = .1
    sigma = 0.4
    periodicity = 4.
    gaubias = gaussian_bias(xaxis,center,sigma,periodicity=periodicity)
    biased_pmf = potential + -np.log(gaubias)
    maxbarr = get_max_barrier(xaxis,biased_pmf,center)


    plt.plot(xaxis,potential)
    plt.plot(xaxis,biased_pmf-min(biased_pmf),color='g')
    print maxbarr
    plt.axhline(maxbarr,color='g')
    plt.ylim(0,4.)
    plt.show()
 

def code_testing():
#    bf_testing()
    minfinder_testing()

if __name__ == "__main__":
    code_testing()
