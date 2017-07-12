#!/usr/bin/env python
"""
EMUS Script runnable from the command line that uses WHAM-like input.
"""
import numpy as np                  
from emus import usutils as uu
from emus import emus, avar
import argparse
import os
from emus._defaults import *
import h5py

def main():
    a = _parse_args() # Get Dictionary of Arguments
    kT = a['k_B'] * a['T']
    
    # Load data
    psis, cv_trajs, neighbors = uu.data_from_meta(a['meta_file'],a['n_dim'],T=a['T'], k_B=a['k_B'],period=a['period'],nsig=a['sigma'])
    if a['fxn_file'] is not None:
        fdata = uu.fxn_data_from_meta(a['fxn_file'])
    else:
        fdata = None

    # Calculate the partition function for each window
    z, F= emus.calculate_zs(psis,neighbors=neighbors,n_iter=a['n_iter'])
    # Calculate the PMF
    pmf,edges = emus.calculate_pmf(cv_trajs,psis,a['domain'],z,neighbors=neighbors,nbins=a['nbins'],kT=kT)   

    # Calculate any averages of functions.
    if fdata is not None:
        favgs = []
        for n, fdata_i in enumerate(fdata):
            favgs.append(emus.calculate_obs(psis,z,fdata_i,neighbors=neighbors))

    # Perform Error analysis if requested.
    if a['error'] is not None:
        zEMUS, FEMUS= emus.calculate_zs(psis,neighbors=neighbors,n_iter=0)
        zvars, z_contribs, z_iats = avar.calc_partition_functions(psis,zEMUS,FEMUS,neighbors=neighbors,iat_method=a['error'])
        z_avg_taus = np.average(z_iats,axis=0)
        pmf_EMUS, pmf_EMUS_avar = avar.calc_pmf(cv_trajs,psis,a['domain'],zEMUS,FEMUS,neighbors=neighbors,nbins=a['nbins'],kT=kT,iat_method=z_avg_taus)
        # Perform analysis on any provided functions.
        if fdata is not None:
            favgs_EM = []
            ferrs = []
            fcontribs = []
            nfxns = len(fdata[0][0])
            for fdata_i in fdata:
                iat, mean, variances = avar.calc_avg_ratio(psis,zEMUS,FEMUS,fdata_i,neighbors=neighbors,iat_method=a['error'])
                print iat
                favgs_EM.append(mean)
                fcontribs.append(variances)
                ferrs.append(np.sum(variances))

    # Save Data

    f = h5py.File(a['output']+'_out.hdf5',"w")
    # Save PMF
    pmf_grp = f.create_group("PMF")
    pmf_dset = pmf_grp.create_dataset("pmf",pmf.shape,dtype='f')
    dmn_dset = pmf_grp.create_dataset("domain",np.array(a['domain']).shape,dtype='f')
    pmf_dset[...] = pmf
    dmn_dset[...] = np.array(a['domain'])
    for ie,edg in enumerate(edges):
        edg_dset = pmf_grp.create_dataset("edge_%d"%ie,edg.shape,dtype='f')
        edg_dset[...] = edg
    if a['error'] is not None:
        pmf_EMUS_dset = pmf_grp.create_dataset("pmf_no_iter",pmf_EMUS.shape,dtype='f')
        pmf_EMUS_dset[...] = pmf_EMUS
        pmf_EMUS_avar_dset = pmf_grp.create_dataset("pmf_no_iter_avars",pmf_EMUS_avar.shape,dtype='f')
        pmf_EMUS_avar_dset[...] = pmf_EMUS_avar

    # Save partition functions
    z_grp = f.create_group("partition_function")
    z_dset = z_grp.create_dataset("z",z.shape,dtype='f')
    z_dset[...] = z
    F_dset = z_grp.create_dataset("F",F.shape,dtype='f')
    F_dset[...] = F
    if a['error'] is not None:
        zerr_dset = z_grp.create_dataset("z_avars",np.array(zvars).shape,dtype='f')
        zerr_dset[...] = np.array(zvars)
        zEMUS_dset = z_grp.create_dataset("z_no_iter",zEMUS.shape,dtype='f')
        zEMUS_dset[...] = zEMUS
        FEMUS_dset = z_grp.create_dataset("F_no_iter",FEMUS.shape,dtype='f')
        FEMUS_dset[...] = FEMUS

    # Save function data.
    if fdata is not None:
        f_grp = f.create_group('function_averages')
        f_dset = f_grp.create_dataset("f",np.shape(favgs),dtype='f')
        f_dset[...] = np.array(favgs)
        if a['error'] is not None:
            fvar_dset = f_grp.create_dataset("f_avars",np.shape(ferrs),dtype='f')
            fvar_dset[...] = ferrs
    f.close()


def _parse_args():
    """
    Helper function to read in the arguments from the command line. 
    """
    ### Construct argparse Parser
    parser = argparse.ArgumentParser(description="Performs EMUS analysis on sampled data stored in a wham-like format.")
    parser.add_argument('n_dim',type=int,help='The number of collective variables of the data') 
    parser.add_argument('metafile',type=str,help='Path to the wham-like meta file')
    parser.add_argument('pmf',type=str, nargs='+',help="Information on the domain of the PMF.  Should be a list of 3*n_dim values, where each triple is the lowest value in that dimension, the highest value, and the number of histogram bins in that dimension respectively.  The parser will also interpret 'pi' for 3.14..., or 'npi' for negative pi, as well as 2pi and n2pi for two pi and negative two pi")
    parser.add_argument('-p','--period',type=str, nargs='+',help="Optional periodicity information for the collective variables.  Format is a list of n_dim values, where each float is either the period of the variable, 'pi' or '2pi' if the value has period of pi or 2 pi respectively (e.g. units are radians), or 0 if the collective variable is aperiodic.")
    parser.add_argument('-f','--fxnfile',type=str,help="Path to meta file containing paths to observable information.")
    parser.add_argument('-T','--temperature',type=float,default=DEFAULT_T,help='Temperature of the data, default is %.4f.  Any temperature that is provided in the meta file overrides this.'% DEFAULT_T)
    parser.add_argument('-k','--boltz',type=str,help="Units for the Boltzmann constant. Default is k_B=%.4f, other choices include 'kCal' for kCal/mol and 'kJ', or a numerical value of the Boltzmann constant."% DEFAULT_K_B)
    parser.add_argument('-n','--n_iter',type=int,default=0,help="Number of iterative EMUS iterations to perform on the data set.  Note that error analysis is only for the initial iteration.")
    parser.add_argument('-s','--sigma',type=float,default=6,help="Do not evaluate F_{ij} if window centers are more than this number of standard deviations apart.  Default is 6.")
    parser.add_argument('-e','--error',type=str,choices=['acor','ipce','icce'],help="If given, performs error analysis on z values and any function given.  The autocorrelation time is estimated using the selected function ('acor' is recommended, although this requires separate installation of the acor package).Note: currently no error analysis is available on the PMF, due to the expense in estimating the autocorrelation time for each histogram bin.")
    parser.add_argument('-o','--output',help="Base string for data to output.  Data is saved to [output].hdf5.  Default is the name of the meta file.") 
    args = parser.parse_args()
    newargs = {}
    ### Process the arguments
    newargs['meta_file'] = args.metafile
    newargs['fxn_file'] = args.fxnfile
    newargs['n_dim'] = args.n_dim
    n_dim = args.n_dim
    newargs['n_iter'] = args.n_iter
    newargs['error'] = args.error
    newargs['sigma'] = args.sigma
    newargs['output'] = args.output
    if newargs['output'] is None:
        newargs['output'] = os.path.splitext(newargs['meta_file'])[0]
    newargs['T'] = args.temperature

    # Boltzmann Constant
    if args.boltz is None:
        k_B = DEFAULT_K_B
    if (args.boltz == 'kCal') or (args.boltz == 'kcal'):
        k_B = 1.9872041E-3
    elif (args.boltz == 'kj') or (args.boltz == 'kJ'):
        k_B = 8.3144621E-3
    else:
        try:
            k_B = float(args.boltz)
        except ValueError:
            raise ValueError("Unable to convert input for k_B to a float.  Input string was %s"%args.boltz)
    newargs['k_B'] = k_B

    # Periodicity
    if args.period is None:
        period = None
    else:
        period = [] 
        for pndx,p in enumerate(args.period):
            if ((p == '0') or (p == '0.0')):
                period.append(None)
            elif ((p == 'pi') or (p == 'Pi') or (p == 'PI')):
                period.append(np.pi)
            elif ((p == '2pi') or (p == '2Pi') or (p == '2PI')):
                period.append(2.*np.pi)
            else:
                try:
                    pd = np.float(p)
                except ValueError:
                    raise ValueError("Unable to convert input for period in dimension %d to a float.  Input string was %s"%(i,args.boltz))
                period.append(pd)
        if len(period) != n_dim:
            raise ValueError("Periodicity information has length %d but %d dimensions were specified"%(len(period),n_dim))
    newargs['period'] = period

    # PMF domain information
    domain = []
    nbins = []
    if len(args.pmf) != 3*n_dim:
        raise ValueError("Mismatch between the no. of parameters provided for the PMF and the number of dimensions.  For %d dimensions %d parameters needed, but only %d were found.  PMF parameter input is "%(n_dim,3*n_dim,len(args.pmf)) + str(args.pmf))
    for nd in xrange(n_dim):
        # Lower bound to PMF value
        ndx = 3*nd
        if args.pmf[ndx] == 'pi':
            domain.append(np.pi)
        elif args.pmf[ndx] == 'npi':
            domain.append(-1*np.pi)
        elif args.pmf[ndx] == '2pi':
            domain.append(2*np.pi)
        elif args.pmf[ndx] == 'n2pi':
            domain.append(-2*np.pi)
        else:
            try:
                domain.append(float(args.pmf[ndx]))
            except ValueError:
                raise ValueError("Unable to parse first argument for PMF parameters in dimension %d."%nd)
        # Upper bound to pmf value
        if args.pmf[ndx+1] == 'pi':
            domain.append(np.pi)
        elif args.pmf[ndx+1] == 'npi':
            domain.append(-1*np.pi)
        else:
            try:
                domain.append(float(args.pmf[ndx+1]))
            except ValueError:
                raise ValueError("Unable to parse second argument for PMF parameters in dimension %d."%nd)
        # Number of Histogram Bins
        try:
            nbins.append(int(args.pmf[ndx+2]))
        except ValueError:
            raise ValueError("Unable to parse third argument for PMF parameters in dimension %d."%nd)
    newargs['domain'] = domain
    newargs['nbins'] = nbins
    
    return newargs


if __name__ == "__main__":
    main()
