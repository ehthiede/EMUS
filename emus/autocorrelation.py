# -*- coding: utf-8 -*-
"""
Tools for analyzing the autocorrelation of a time series
"""
import numpy as np

def autocorrfxn(timeseries,lagmax):
    ts = np.asarray(timeseries)
    ts -= np.average(ts) # Set to mean 0
    N = len(timeseries)
    corrfxn = np.zeros(lagmax)
    for dt in xrange(lagmax):
        corrfxn[dt] = (np.dot(timeseries[0:N-dt],timeseries[dt:N])) # sum of ts[t+dt]*ts[t]
    corrfxn /= corrfxn[0] # Normalize
    return corrfxn


def ipce(timeseries,lagmax=None):
    """
    Initial positive correlation time estimator
    """
    if lagmax == None:
        lagmax = len(timeseries)/2
    corrfxn = autocorrfxn(timeseries,lagmax)
    i = 0
    t = 0
    while i < 0.5*lagmax:
        gamma =  corrfxn[2*i] + corrfxn[2*i+1]
        if gamma < 0.0:
#            print 'stop at ',2*i
            break
        else:
            t += gamma 
        i += 1
    tau = 2*t - 1
    var = np.var(timeseries)
    mean = np.average(timeseries)
    sigma = np.sqrt(var * tau / len(timeseries))
    return tau, mean, sigma
    
def icce(timeseries,lagmax=None):
    """
    Initial convex correlation time estimator
    REWRITE TO BE MORE EFFICIENT!
    """
    if lagmax == None:
        lagmax = len(timeseries)/2
    corrfxn = autocorrfxn(timeseries,lagmax)
    t = corrfxn[0] + corrfxn[1]
    i = 1
    gammapast = t
    gamma = corrfxn[2*i] = corrfxn[2*i+1]
    while i < 0.5*lagmax-2:
        gammafuture =  corrfxn[2*i+2] + corrfxn[2*i+3]
        if gamma > 0.5*(gammapast+gammafuture) :
            print 'stop at ',2*i
            break
        else:
            t += gamma 
            gammapast = gamma
            gamma = gammafuture
        i += 1
    tau = 2*t - 1
    var = np.var(timeseries)
    mean = np.average(timeseries)
    sigma = np.sqrt(var * tau / len(timeseries))
    return tau, mean, sigma

