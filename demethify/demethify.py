import os
import sys
import argparse
import numpy as np
import pandas as pd
from time import time
from .deconvolution import *
from .bootstrap import *
from .ic import *
from .plotting import *
import warnings
warnings.filterwarnings("ignore")

logo = """     
    ____                      __  __    _ ____     
   / __ \___  ____ ___  ___  / /_/ /_  (_) __/_  __
  / / / / _ \/ __ `__ \/ _ \/ __/ __ \/ / /_/ / / /
 / /_/ /  __/ / / / / /  __/ /_/ / / / / __/ /_/ / 
/_____/\___/_/ /_/ /_/\___/\__/_/ /_/_/_/  \__, /  
                                          /____/   
""" 
    

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description="DeMethify - Partial reference-based Methylation Deconvolution")

    # Add regular arguments
    parser.add_argument('--methfreq', nargs='+', type=str, required=True, help='Methylation frequency file path (values between 0 and 1)')
    parser.add_argument('--ref', nargs='?', type=str, help='Methylation reference matrix file path')
    parser.add_argument('--iterations', nargs=2, type=int, help='Numbers of iterations for outer and inner loops (default without purity = 10000, 20, with purity= 100, 500)')
    parser.add_argument('--nbunknown', nargs=1, type=int, help="Number of unknown cell types to estimate ")
    parser.add_argument('--purity', nargs='+', type=int, help="The purities of the samples in percent [0,100], if known")
    parser.add_argument('--termination', nargs=1, type=float, default=1e-2, help='Termination condition for cost function (default = 1e-2)')
    parser.add_argument('--init', nargs="?", default='uniform', help='Initialisation option (default = random uniform)')
    parser.add_argument('--outdir', nargs='?', required=True, help='Output directory')
    parser.add_argument('--fillna', action="store_true", help='Replace every NA by 0 in the given data')
    parser.add_argument('--ic', nargs="?", help='Select number of unknown cell types by minimising an information criterion (AIC or BIC)')
    parser.add_argument('--confidence', nargs=2, type=int, help='Outputs bootstrap confidence intervals, takes confidence level and boostrap iteration numbers as input.')
    parser.add_argument('--plot', action="store_true", help='Plot cell type proportions estimates for each sample, eventually with confidence intervals. ')

    group = parser.add_mutually_exclusive_group()

    # Add conflicting arguments to the group
    group.add_argument('--bedmethyl', action='store_true', help="Flag to indicate that the input will be bedmethyl files, modkit style")
    group.add_argument('--counts', nargs='+', type=str, help='Read counts file path')
    group.add_argument('--noreadformat', action="store_true", help="Flag to use when the data isn\'t using the read format (e.g Illumina epic arrays)")

    # Parse the arguments
    args = parser.parse_args()


    if not args.iterations:
        if args.purity:
            args.iterations = [100, 500]
        else:
            args.iterations = [10000, 20]

    if args.purity:
        purity = np.array(args.purity)
        if np.any((purity >= 0) & (purity <= 1)):
            print("Purity is between 0 and 1, are you sure that it's a percentage?")
        elif np.any((purity < 0) & (purity > 100)):
            sys.stderr.write("Error: Invalid value for purity, not within [0,100] bounds.")
            sys.exit(1)

        purity = purity / 100.0


    if args.ic:
        if args.nbunknown:
            sys.stderr.write("Error: --ic cannot be used with --nbunknown.\n")
            sys.exit(1)
        if not args.ref:
            sys.stderr.write("Error: --ic requires --ref to be specified.\n")
            sys.exit(1)
    
    print(logo)
    
    outdir = os.path.join(os.getcwd(), args.outdir)
    if not os.path.exists(outdir):
        print(f'Creating directory {outdir} to store results')
        os.mkdir(outdir)

    if args.nbunknown is None:
    	args.nbunknown = [0]
    
    # read bedmethyl files (modkit output)
    if(args.bedmethyl):
        if(args.ref):
            ref = pd.read_csv(args.ref, sep='\t').iloc[:, 3:]
            header = list(ref.columns)
            ref = ref.values
        list_meth_freq = []
        list_counts = []
        for bed in args.methfreq:
            temp = pd.read_csv(bed, sep='\t')
            if(args.fillna):
                temp = temp.fillna(0)
            list_meth_freq.append(temp["percent_modified"].values / 100)
            list_counts.append(temp["valid_coverage"].values)
        meth_f = np.column_stack(list_meth_freq)
        counts = np.column_stack(list_counts)
	    

    # read csv files
    else:
        meth_f = pd.read_csv(args.methfreq).values
        if(args.ref):
            ref = pd.read_csv(args.ref)
            header = list(ref.columns)
            ref = ref.values

        if(args.fillna):
            meth_f.fillna(0, inplace = True)
            if(args.ref):
                ref.fillna(0, inplace = True)
        if(not(args.noreadformat)):
            counts = pd.read_csv(args.counts).values
            if(args.fillna):
                counts.fillna(0, inplace = True)
        else:
            counts = np.ones_like(meth_f)


    args.methfreq = [bla.split("/")[-1] for bla in args.methfreq]

    # deconvolution
    time_start = time()

    if(args.confidence):
        bt_results = bt_ci(args.confidence[0], args.confidence[1], args.nbunknown[0], meth_f, counts, ref, args.init, args.iterations[0], args.iterations[1], args.termination, header, outdir, args.methfreq, args.purity)

    if(not args.ref):
        ref_estimate, proportions = unsupervised_deconv(meth_f, args.nbunknown[0], counts, args.init, n_iter1 = args.iterations[0], n_iter2 = args.iterations[1], tol = args.termination)
        unknown_header = ["unknown_cell_" + str(i + 1) for i in range(args.nbunknown[0])]
        header = unknown_header
        pd.DataFrame(ref_estimate).to_csv(outdir + '/methylation_profile_estimate.csv', index = False, header=unknown_header)
        
    elif(args.ic):
        ref_estimate, proportions, ic_n_u = evaluate_best_ic(meth_f, ref, counts, args.init, args.ic)
        unknown_header = ["unknown_cell_" + str(i + 1) for i in range(ic_n_u)]
        header += unknown_header
        pd.DataFrame(ref_estimate).to_csv(outdir + '/methylation_profile_estimate.csv', index = False, header=unknown_header)
        
    elif(args.nbunknown[0] > 0 and meth_f.shape[1] >= 1):
        if args.purity:
            u, R, alpha = init_BSSMF_md_p(args.init, meth_f, counts, ref, args.nbunknown[0], purity, rb_alg = fs_irls)
            ref_estimate, proportions = mdwbssmf_deconv_p(u, R, alpha, meth_f, counts, ref, args.nbunknown[0], purity,n_iter1 = args.iterations[0], n_iter2 = args.iterations[1], tol = args.termination)
        else:
            u, R, alpha = init_BSSMF_md(args.init, meth_f, counts, ref, args.nbunknown[0], rb_alg = fs_irls)
            ref_estimate, proportions = mdwbssmf_deconv(u, R, alpha, meth_f, counts, ref, args.nbunknown[0], n_iter1 = args.iterations[0], n_iter2 = args.iterations[1], tol = args.termination)
        unknown_header = ["unknown_cell_" + str(i + 1) for i in range(args.nbunknown[0])]
        header += unknown_header
        pd.DataFrame(ref_estimate).to_csv(outdir + '/methylation_profile_estimate.csv', index = False, header=unknown_header)
        
    elif(args.nbunknown[0] == 0 and meth_f.shape[1] >= 1):
        alpha_tab = []
        for k in range(meth_f.shape[1]):
            alpha_tab.append(fs_irls(counts[:,k:k+1] * meth_f[:,k:k+1], counts[:,k:k+1], ref))
        proportions = np.concatenate(alpha_tab, axis = 1)
    else:
        exit(f'Invalid number of unknown value! : "{args.nbunknown}" ')
        
    time_tot = time() - time_start
    
    # saving output files
    proportions = pd.DataFrame(proportions)
    proportions.index = header
    proportions.columns = args.methfreq
    proportions.index.name = "Cell types"
    proportions.to_csv(outdir + '/celltypes_proportions.csv', index = True)

    print("All demethified! Results in " + outdir)
    f = open(os.path.join(outdir, 'log.log'), "w+")
    f.write("Total execution time = " + str(time_tot) + " s" + '\n')
    if(args.ic):
        f.write("Number of unknowns that minimises " + args.ic + " : " + str(ic_n_u))
    f.close()
    
    if(args.plot):
        ci_df = pd.DataFrame()
        if(args.confidence):
            ci_df = bt_results[0]
        plot_proportions(proportions, ci_df, outdir)

if __name__ == "__main__":
	main()
