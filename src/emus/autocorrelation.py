# -*- coding: utf-8 -*-
""" Tools for analyzing the autocorrelation time of a time series.

The ipce and icce routines are implementations of the initial positive correlation time estimator, and the initial convex correlation estimator proposed by Geyer [1]_.
The acor algorithm was proposed by Sokal [2]_.  The associated code, as well as the code for constructiing autocorrelation functions is taken from the emcee package [3]_.

.. [1] C.J. Geyer. Statistical Science (1992): 473-483.
.. [2] A. Sokal, Functional Integration. Spring, Boston, MA, 1997. 131-192.
.. [3] D. Foreman-Mackey, D.W. Hogg, D. Lang, and J. Goodman. Publications of the Astronomical Society of the Pacific 125.925 (2013): 306.
"""
import numpy as np
import logging


def _next_pow_two(n):
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < n:
        i = i << 1
    return i


def _auto_window(taus, c):
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return np.argmin(m)
    return len(taus) - 1


def autocorrfxn(x):
    """Estimate the normalized autocorrelation function of a 1-D series

    Parameters
    ----------
    x : ndarray
        The time series of which to calculate the autocorrelation function.

    Returns
    -------
    acfxn : ndarray
        The autocorrelation as a function of lag time.
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = _next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2*n)
    acf = np.fft.ifft(f * np.conjugate(f))[:len(x)].real
    acf /= acf[0]
    return acf


def ipce(x):
    """ The initial positive correlation time estimator for the autocorrelation time, as proposed by Geyer.

    Parameters
    ----------
    x : ndarray
        The time series of which to calculate the autocorrelation function.

    Returns
    -------
    tau : float
        Estimate of the autocorrelation time.
    mean : float
        Average value of x
    sigma : float
        Estimate of the square root of the autocovariance of x

    """
    x = np.copy(x)
    mean = np.average(x)
    corrfxn = autocorrfxn(x)
    lagmax = int(len(x) / 2)
    i = 0
    t = 0
    while i < 0.5*lagmax:
        gamma = corrfxn[2*i] + corrfxn[2*i+1]
        if gamma < 0.0:
            #            print 'stop at ',2*i
            break
        else:
            t += gamma
        i += 1
    tau = 2*t - 1
    var = np.var(x)
    sigma = np.sqrt(var * tau / len(x))
    return tau, mean, sigma


def integrated_time(x, c=5, tol=50, quiet=False):
    """ Estimate the integrated autocorrelation time of a time series.
    This estimate uses the iterative procedure described on page 16 of
    `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ to
    determine a reasonable window size.

    Parameters
    ----------
    x : ndarray
        The time series of which to calculate the autocorrelation function.
    c : float
        The step size for the window search. Default is 5.
    tol : int
        The minimum number of autocorrelation times needed to trust the estimate. Default is 50.
    quiet : bool, optional
        This argument controls the behavior when the chain is too short. If True, gives a warning instead of raising an error.  Default is True

    Returns
    -------
    tau : float
        Estimate of the autocorrelation time.

    Raises
    ------
        ValueError: If the autocorrelation time can't be reliably estimated
            from the chain and ``quiet`` is ``False``. This normally means
            that the chain is too short.
    """
    x = np.atleast_1d(x)
    if len(x.shape) == 1:
        x = x[:, np.newaxis, np.newaxis]
    if len(x.shape) == 2:
        x = x[:, :, np.newaxis]
    if len(x.shape) != 3:
        raise ValueError("invalid dimensions")

    n_t, n_w, n_d = x.shape
    tau_est = np.empty(n_d)
    windows = np.empty(n_d, dtype=int)

    # Loop over parameters
    for d in range(n_d):
        f = np.zeros(n_t)
        for k in range(n_w):
            f += autocorrfxn(x[:, k, d])
        f /= n_w
        taus = 2.0*np.cumsum(f)-1.0
        windows[d] = _auto_window(taus, c)
        tau_est[d] = taus[windows[d]]

    # Check convergence
    flag = tol * tau_est > n_t

    # Warn or raise in the case of non-convergence
    if np.any(flag):
        msg = (
            "The chain is shorter than {0} times the integrated "
            "autocorrelation time for {1} parameter(s). Use this estimate "
            "with caution and run a longer chain!\n"
        ).format(tol, np.sum(flag))
        msg += "N/{0} = {1:.0f};\ntau: {2}".format(tol, n_t/tol, tau_est)
        if not quiet:
            raise ValueError(msg)
        logging.warning(msg)

    return tau_est


def acor(x, tol=10, quiet=True):
    """ Acor algorithm, as proposed by Sokal and implemented in EMCEE.

    Parameters
    ----------
    x : ndarray
        The time series of which to calculate the autocorrelation function.
    tol : int, optional
        The minimum number of autocorrelation times needed to trust the estimate.  Default is 10.
    quiet : bool, optional
        This argument controls the behavior when the chain is too short. If True, gives a warning instead of raising an error.  Default is True

    Returns
    -------
    tau : float
        Estimate of the autocorrelation time.
    mean : float
        Average value of x
    sigma : float
        Estimate of the square root of the autocovariance of x
    """
    mean = np.average(x)
    tau = integrated_time(x, tol=tol, quiet=quiet)
    var = np.var(x)
    sigma = np.sqrt(var * tau / len(x))
    return tau, mean, sigma


def icce(x):
    """The initial convex correlation time estimator for the autocorrelation time, as proposed by Geyer.

    Parameters
    ----------
    x : ndarray
        The time series of which to calculate the autocorrelation function.

    Returns
    -------
    tau : float
        Estimate of the autocorrelation time.
    mean : float
        Average value of x
    sigma : float
        Estimate of the square root of the autocovariance of x

    """
    x = np.copy(x)
    lagmax = int(len(x) / 2)
    corrfxn = autocorrfxn(x)
    t = corrfxn[0] + corrfxn[1]
    i = 1
    gammapast = t
    gamma = corrfxn[2*i] = corrfxn[2*i+1]
    while i < 0.5*lagmax-2:
        gammafuture = corrfxn[2*i+2] + corrfxn[2*i+3]
        if gamma > 0.5*(gammapast+gammafuture):
            break
        else:
            t += gamma
            gammapast = gamma
            gamma = gammafuture
        i += 1
    tau = 2*t - 1
    var = np.var(x)
    mean = np.average(x)
    sigma = np.sqrt(var * tau / len(x))
    return tau, mean, sigma


def _get_iat_method(iat_method):
    """Control routine for selecting the method used to calculate integrated
    autocorrelation times (iat)

    Parameters
    ----------
    iat_method : string, optional
        Routine to use for calculating said iats.  Accepts 'ipce', 'acor', and 'icce'.

    Returns
    -------
    iatroutine : function
        The function to be called to estimate the integrated autocorrelation time.

    """
    if iat_method == 'acor':
        # from autocorrelation import acor
        iatroutine = acor
    elif iat_method == 'ipce':
        # from autocorrelation import ipce
        iatroutine = ipce
    elif iat_method == 'icce':
        # from autocorrelation import icce
        iatroutine = icce
    else:
        raise ValueError('Method for calculation iat not recognized.')
    return iatroutine
