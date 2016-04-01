#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module implementing MBAR algorithm
"""

import numpy as np
import argparse
import usroutines as usr

############# DEFAULT PARAMETERS #############
# Default values 
_DEFAULT_kB = 1.9872041*10**-3 # Boltzmann Constant, given in kcal/mol
_DEFAULT_TEMP = 310.0 # Default Temperature in Kelvin.
_DEFAULT_tol = 1E-8 # Relative Tolerance for MBAR iterations CURRENTLY UNUSED
_DEFAULT_NBR_SIGMA = 4.0 # Neighborlist radius for each harmonic bias potential
    #in units of standard deviations of the corresponding gaussian
##############################################


def parse_metafile(filestr,dim):
    """
    Parses the meta file located at filestr.

    Parameters
    ----------
    filestr : string
        The path to the meta file.
    dim : int
        The number of dimensions in the cv space.

    Returns
    -------
    ks : array of strings
        Array, element i is the location of the cv trajectory for 
        window i.

    #### FINISH THE DOCUMENTATION!!!!!

    """
    trajlocs = []
    ks = []
    cntrs = []
    corrts = []
    temps = []
    with open(filestr,'r') as f:
        for line in f:
            windowparams = line.split(' ')
            trajlocs.append(windowparams[0])
            cntrs.append(windowparams[1:1+dim])
            ks.append(windowparams[1+dim:1+2*dim])
            if len(windowparams) > 1+2*dim: # If Correlation Time provided
                corrts.append(windowparams[2+2*dim])
            if len(windowparams) > 2+2*dim: # If Temperature is provided
                temps.append(windowparams[3+2*dim])
    # Move to numpy arrays, convert to appropriate data types
    ks = np.array(ks).astype('float')
    cntrs = np.array(cntrs).astype('float')
    corrts = np.array(corrts).astype('float')
    temps = np.array(temps).astype('float')
    return trajlocs,ks,cntrs,corrts,temps

def _def_arg_parser():
    """
    Creates the parser argument used to interpret command line instructions.

    Returns
    -------
    parser : argparse ArgumentParser object
    """
    parser = argparse.ArgumentParser(description="Runs EMUS algorithm "
        "for recombining umbrella sampling data in 1D.  Code is set up "
        "to be similar to that for the WHAM1D code by Grossfield et. al. "
        "note that command line options with negative numbers "
        "must include zeros after decimals due to quirks in argparse.")
    parser.add_argument('metadatafile',type=str,help="File containing "
        "the locations and force constants of the umbrellas.")
    parser.add_argument('histinfo',nargs='+',type=float,help="Specifies "
        "the min, max values for the histogram, and number of histograms in "
        "that dimension.  Takes format xmin, xmax, nhistx, ymin, ymin, nhisty, "
        "and so forth for higher dimensions.")
    parser.add_argument('--period',nargs='+',type=float,help="Periodicity of "
        "the collective variable in each dimension.  If a collective variable "
        "is aperiodic, just set to zero.  If not set, it is assumed every "
        "collective variable is aperiodic.")
    parser.add_argument('--kB',type=float,help="Boltzmann constant.  Default is in "
        "kcal/mol; default can be changed in header .")
    parser.add_argument('--T',type=float,help="Temperature.  Default is 310.0K,"
        "default can be changed in header")  
    parser.add_argument('--neighbor',nargs='?',const=_DEFAULT_NBR_SIGMA,type=float,
        help="Truncation parameter for neighborlisting the umbrellas."
        "Corresponds to how many standard deviations "
        "of the corresponding Gaussian an window contributes to.  This introduces a "
        "slight bias into the estimator, but saves calculation time and memory."
        "If flag given w/o specifications, set to a default of 4.0.")
#    parser.add_argument('--statefefile',type=str,help="File to write the \
#        the final state free energies to.")
#    parser.add_argument('--pmffile',type=str,help="File to write the \
#        the potential of mean force to.")
#    parser.add_argument('-silent','--silent',type=str,help="File to write the \
#        the final state free energies to.")
    parser.add_argument('--zasymptoticvar',nargs=2,type=int,help="Calculates "
        "asymptotic variance and windows importances for the free energy "
        "difference between the two windows specified by the user.")
    #### FUNCTIONALITY TO BE IMPLEMENTED
#    parser.add_argument('--mem',type=str,help="Uses Slower, but less \
#        memory intensive routines for calculating values.  Can be \
#        considerably slower if many polishing iterations requested.")
#    parser.add_argument('--makepmf',nargs='+',type=float,help="If \
#        specified, computes the potential of mean force.  Takes in \
#        sequence of float of format \"min max nbins\".  Multiple \
#        dimensions can be specified, e.g. \"xmin xmax xnbins ymin \
#        ymax ynbins\" and so forth.")
##    parser.add_argument('hist_min',type=float,help="Maximum value on the \
##        histogram to calculate the PMF.  Note: if negative, requires a \
##        number after the decimal place:  '-180.0' is ok, but '180.' is \
##        not.")
##    parser.add_argument('hist_max',type=float,help="Maximum value on the \
##        histogram to calculate the PMF.")
#    parser.add_argument('--numpad',type=int,help="Number of padding \
#    histogram bins to add at the front and end of the
#    parser.add_argument('num_bins',type=int,help="Number of histogram \
#        bins to use for the PMF.")
    return parser

def _printmessage():
    messagestring = """#########################################################################
#                                                                       #
#                                             /-/-=/-=                  #
#                                             // ZX    "                #
#                                           / /XZX  (O)  ---,           #
#                                           // ZXZX     _____`\         #
#                                            /  ZXZXZX                  #
#                                             / ;,ZXZ                   #
#                                             /  :ZX%\                  #
#                                             /%  : X\                  #
#                                             :/:  %:X:                 #
#                                             /%:%   %\\                #
#                                              /:%:  %  \               #
#                                              /  :%  : \               #
#                                               /  :%  :%\              #
#                                               /%  %    \              #
#                                               /%:  %:  %\             #
#                                               //%    :  %\            #
#                          %%%%%%%%%%  %% %%   /  /     :   :           #
#                     %%%%%%%%       %%  %   %%%%           \:          #
#                   %%%%%%                                 : \          #
#                 %%%%                                    :%:\          #
#              % %                                       %  \           #
#             %                                         %% \            #
#           %                                  %%% %%%%%% \             #
#         %                                 %%%%%%%% %%%\               #
#        %                                %%  %%%%%% %\                 #
#        %                         %%%%%%%%%%%%% %%%%\\                 #
#        %%%%/%%%%%%%%%%%%%%%%%%%%%%% %%%%%%% %%% %%\                   #
#        /       %%%%%%%%%  % %% % %%% % %%%%% %%%\                     #
#                       (_____)%%(_____)%%%%%%\                         #
#                     ((_____) ((____)                                  #
#                       (___)    (___)                                  #
#                        (__)     (__)                                  #
#                         (__)     (__)                                 #
#                          (__)     (__)                                #
#                           (__)     (__)                               #
#                            (  )     (  )                V@V           #
#                            (  )     (  )                 |            #
#                           (    )   (    )           \    |  / /       #
#                         (__    |  (___  |            \   |  /         #
#                     ___(  /  \  \____ \  |---          \\|//          #
#                   <(____/      \_____(>\_____(>      -===-=           #
#                                                                       #
####################### STARTING EMUS CALCULATION #######################
"""
    print messagestring
 
def main():
    """
    Method that calls EMUS from command line.
    """
    # Read in command line args
    parser = _def_arg_parser()  #Container method for defining cmdline args.
    args = parser.parse_args()
    emus1d(args.metadatafile,args.histinfo,args.period,args.kB,
        args.T,args.neighbor,args.zasymptoticvar)

    
def emus1d(metadatafile,histinfo,period=None,kB=_DEFAULT_kB,
        T=_DEFAULT_TEMP,neighbor=None,zasymptoticvar=None):
    """
    Performs 1D EMUS calculation.
    """

    # Define variables for the purposes of code readibility
    dim = len(histinfo)/3 # Number of Dimensions in cv space.
    try:
        histdata = np.reshape(np.array(histinfo),(dim,3))
    except ValueError:
        raise IOError("Error reading in histogram specifications: "
             "check if values are missing.")

        
    print "histogram data",histdata


#    _printmessage()
#    with open('asciiemus.txt','r') as finalmessage:
#        print finalmessage.read()
    trajlocs, ks, cntrs, corrts, temps  = parse_metafile(metadatafile,dim)
    L = len(trajlocs) # Number of states
    
    # Set kT corresponding to user input.
    kB = 1.9872041*10**-3 # Boltzmann Constant, given in kcal/mol
    if len(temps) is 0: # If temperature values not provided in meta file.
        temps = np.ones(len(trajlocs))
        temps *= 310.0
    kTs = temps*kB

    # Calculate Umbrella neighbors is specified.
    if neighbor is not None:
        print period
        nbrs = usr.neighbors_harmonic(cntrs,ks,kTs,period=period,nsig=neighbor)
    else:
        nbrs = [np.arange(L) for i in range(L)] 

    # Load in data
    print "#### Loading in Data ####"
    trajs = []
    for i, trajloc in enumerate(trajlocs):
        trajs.append(np.loadtxt(trajloc)[:,1:]) 
    # Calculate psi values
    psis = []
    print "#### Calculating Bias Fxns ####"
    for i in xrange(L):
        nbcenters = cntrs[nbrs[i]]
        psis.append(usr.calc_harmonic_psis(trajs[i],nbcenters,ks,kTs,period=period))
    print "# Calculated biases"

    # Calculate the weights,
    print "#### Calculating Normalization Constants ####"
#    z,F = usr.EMUS_weights(psis,nbrs)
    from emus import emus
    EM = emus(psis)
    z = EM.calc_zs()
    z_MB1 = EM.calc_zs(max_iter=2)
    z_MB2 = EM.calc_zs(max_iter=3)
    np.save("z_emustest",z)
    np.save("z_MB1",z_MB1)
    np.save("z_MB2",z_MB2)

    return

    wfes = -np.log(z)
    print "# Unitless free energy for each window, -ln(z):"
    for i, wfes_i in enumerate(wfes):
        print "Window_FE: %d %f"%(i,wfes_i)

    # Calculate any window importances.
    if zasymptoticvar is not None:
        print "#### Calculating Window Importances ####"
        (um1, um2) = zasymptoticvar
        print "Calculation importances for FE difference between windows %i, %i"%(um1,um2)
        errs, taus = usr.avar_zfe(psis,nbrs,um1,um2)
        print "Estimated Free Energy, Asymptotic Variance:"
        print "Delta_FE: ",wfes[um2]-wfes[um1], np.dot(errs,errs)
        print "# Importance of each window::"
        imps = errs * np.array([len(traj) for traj in trajs])
        imps *= L/np.sum(imps)
        for i, imp_i in enumerate(imps):
            print "Window_imp: %d %f"%(i,imp_i)


    print "####################### EMUS FINISHED SUCCESFULLY #######################"

#    print trajlocs
#    print ks
#    print cntrs
    
    
if __name__ == "__main__":
    main()
