import os
import argparse
import numpy as np
import numpy.random as rd
import pandas as pd
import tqdm
from time import time
from numba import njit
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import FastICA
from sklearn.decomposition import TruncatedSVD
import colorcet as cc  
import seaborn as sns
import matplotlib.pyplot as plt
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

def compute_bic(cost, n_u, n_cpg, n_ct, n_samples):
    k = n_u * n_cpg + (n_ct + n_u - 1) * n_samples
    bic = 2 * np.log(cost) + k * np.log(n_cpg)
    return bic

def compute_aic(cost, n_u, n_cpg, n_ct, n_samples):
    k = n_u * n_cpg + (n_ct + n_u - 1) * n_samples
    aic = 2 * (np.log(cost) + k)
    return aic

# @njit
def evaluate_best_ic(meth_f, ref, counts, init_option, ic):
    # Example parameters
    max_range=meth_f.shape[1]
    n_u_values = range(0, max_range + 1)  
    n_cpg, n_samples = meth_f.shape
    n_ct = ref.shape[1]

    best_ic = float('inf')
    best_n_u = None
    best_u_overall = None
    best_alpha_overall = None

    for n_u in n_u_values:
        # Run the deconvolution for the current n_u
        if(n_u >= 1):
            u, R, alpha = init_BSSMF_md(init_option, meth_f, counts, ref, n_u, rb_alg = fs_irls)
            u, alpha = mdwbssmf_deconv(u, R, alpha, meth_f, counts, ref, n_u, 10000, 20, 1e-2)
            R = np.hstack((ref, u.reshape(-1, n_u)))

        else:
            alpha_tab = []
            for k in range(n_samples):
                alpha_tab.append(fs_irls(counts[:,k:k+1] * meth_f[:,k:k+1], counts[:,k:k+1], ref))
            alpha = np.concatenate(alpha_tab, axis = 1)
            R = ref

        cost = cost_f_w(meth_f, R, alpha, counts)
        # Compute the BIC
        if(ic == "BIC"):
            func = compute_bic
        elif(ic == "AIC"):
            func = compute_aic
            
        ic_result = func(cost, n_u, n_cpg, n_ct, n_samples)
    
        # Update the best BIC and best n_u
        if ic_result < best_ic:
            best_ic = ic_result
            best_alpha_overall = alpha
            best_n_u = n_u
            if(n_u>=1):
                best_u_overall = u
    return best_u_overall, best_alpha_overall, best_n_u


def wls_deconv(ref, samples, weights):
    reg = LinearRegression(fit_intercept = False, positive = True).fit(ref, samples, weights.ravel())
    temp = reg.coef_.T
               
    P_deconv = temp / temp.sum()
               
    return P_deconv



def fs_irls(x, d_x, R_full, tol = 1e-4, n_iter = 1000):
    nb_celltypes = R_full.shape[1]
    alpha = np.reshape(rd.dirichlet(np.ones(nb_celltypes)), (nb_celltypes, 1))
    
    for k in range(n_iter):
        gamma = R_full @ alpha
        W = np.divide(d_x, (gamma * (1 - gamma)) + 1e-16)
        W[np.isnan(W)] = 1e-16
        z = np.divide(x, d_x)
        z[np.isnan(z)] = 1e-16
        a =  wls_deconv(R_full, z, W)
        
        if(np.mean(abs(a - alpha)) / np.mean(abs(alpha)) < tol):
            break
        else:
            alpha = a
    
    return alpha


@njit
def cost_f_w(y, R, alpha, d_x):
    return np.linalg.norm((y - R @ alpha))**2
    # return np.linalg.norm(np.sqrt(d_x) * (y - R @ alpha))**2

@njit
def projection_simplex_sort_2d(v, z=1):
    p, n = v.shape
    w = np.zeros_like(v)
    
    for i in range(n):
        u = np.sort(v[:, i])[::-1]  
        pi = np.cumsum(u) - z
        rho = -1
        for j in range(p):
            if u[j] - pi[j] / (j + 1) > 0:
                rho = j

        theta = pi[rho] / (rho + 1)
        for j in range(p):
            w[j, i] = max(v[j, i] - theta, 0)
    
    return w


def init_BSSMF_md(init_option, meth_frequency, d_x, R_trunc, n_u, rb_alg = fs_irls):
    alpha_tab = []
    nb = meth_frequency.shape[1]

    if(init_option != "uniform" and n_u > nb):
        print("The number of unknowns is greater than the number of samples, we'll go with a uniform initialisation. ")
        init_option = 'uniform'
    
    if(init_option == 'uniform'): ## Random uniform
        u = rd.uniform(size = (R_trunc.shape[0], n_u)) 
    elif(init_option == 'ICA'): ## Independent component analysis
        tt = FastICA(n_components=n_u, tol=1e-2, max_iter=200)
        u = tt.fit_transform(meth_frequency)
        u_ = (u - np.min(u)) 
        u = u_ / np.max(u_)
    elif(init_option == 'SVD'):
        tt = TruncatedSVD(n_components = n_u)
        u = tt.fit_transform(meth_frequency)
        u_ = (u - np.min(u)) 
        u = u_ / np.max(u_)

    R = np.c_[R_trunc, u]


    for k in range(nb):
        alpha_tab.append(rb_alg(d_x[:,k:k+1] * meth_frequency[:,k:k+1], d_x[:,k:k+1], R))
    alpha = np.concatenate(alpha_tab, axis = 1)
    
    if(alpha[-n_u:][0].all() == 0.0):
        alpha[-n_u:][0] = 1e-10
        alpha[:-n_u] = (1 - 1e-10) * alpha[:-n_u]
        
    return u, R, alpha

@njit
def update_u(u, alpha, n_iter2, a1, l_w_, l_w, u_, meth_frequency, R_trunc, n_u, d_x):
    for i in range(n_iter2):
        a0 = a1
        a1 = (1 + np.sqrt(1 + 4 * a0 * a0)) / 2
        beta_w = min((a0 - 1) / a1, 0.9999 *  np.sqrt(l_w_ / l_w))
        u_temp = u + beta_w * (u - u_)
        u_ = u
        u = np.clip((u_temp + (d_x * ((meth_frequency - R_trunc @ alpha[:-n_u] - u_temp @ alpha[-n_u:])) @ alpha[-n_u:].T) / l_w), 0, 1)
        l_w_ = l_w
    return u, u_, a1, l_w_
    
@njit
def update_alpha(n_iter2, alpha, a2, l_h_, l_h, alpha_, R, d_x, meth_frequency):
    for j in range(n_iter2):
        a0 = a2
        a2 = (1 + np.sqrt(1 + 4 * a0 * a0)) / 2
        beta_h = min((a0 - 1) / a2, 0.9999 *  np.sqrt(l_h_ / l_h))
        alpha_temp = alpha + beta_h * (alpha - alpha_)
        alpha_ = alpha
        alpha = projection_simplex_sort_2d(alpha_temp + (R.T @ (d_x * (meth_frequency - R @ alpha_temp))) / l_h)
        l_h_ = l_h
    return alpha, alpha_, a2, l_h_

def unsupervised_deconv(meth_frequency, n_u, d_x, init_option, n_iter1=100000, n_iter2=50, tol=1e-3):

    if(init_option != "uniform" and n_u > meth_frequency.shape[1]):
        print("The number of unknowns is greater than the number of samples, we'll go with a uniform initialisation. ")
        init_option = 'uniform'
    
    if(init_option == 'uniform'): ## Random uniform
        u = rd.uniform(size = (meth_frequency.shape[0], n_u)) 
    elif(init_option == 'ICA'): ## Independent component analysis
        tt = FastICA(n_components = n_u)
        u = tt.fit_transform(meth_frequency)
        u_ = (u - np.min(u)) 
        u = u_ / np.max(u_)
    elif(init_option == 'SVD'):
        tt = TruncatedSVD(n_components = n_u)
        u = tt.fit_transform(meth_frequency)
        u_ = (u - np.min(u)) 
        u = u_ / np.max(u_)
	    
    alpha = rd.dirichlet(np.ones(n_u), meth_frequency.shape[1]).T
    
    a1 = 1.0
    a2 = 1.0
    u_ = u.copy()
    alpha_ = alpha.copy()
    
    l_w = (np.linalg.norm(alpha[-n_u:]) * 
           np.linalg.norm(alpha[-n_u:].T) * 
           np.linalg.norm(d_x.astype(np.float64)))
    l_w_ = l_w
    
    l_h = (np.linalg.norm(u.T) * 
           np.linalg.norm(d_x.astype(np.float64)) * 
           np.linalg.norm(u))
    l_h_ = l_h
    
    
    cf = cost_f_w(meth_frequency, u, alpha, d_x)  
    
    for k in range(n_iter1):
        cf_0 = cf

        for i in range(n_iter2):
            a0 = a1
            a1 = (1 + np.sqrt(1 + 4 * a0 * a0)) / 2
            beta_w = min((a0 - 1) / a1, 0.9999 *  np.sqrt(l_w_ / l_w))
            u_temp = u + beta_w * (u - u_)
            u_ = u
            u = np.clip((u_temp + (d_x * ((meth_frequency -  u @ alpha)) @ alpha.T) / l_w), 0, 1)
            l_w_ = l_w

        l_h = (np.linalg.norm(u.T) * 
               np.linalg.norm(d_x.astype(np.float64)) * 
               np.linalg.norm(u))

        for j in range(n_iter2):
            a0 = a2
            a2 = (1 + np.sqrt(1 + 4 * a0 * a0)) / 2
            beta_h = min((a0 - 1) / a2, 0.9999 *  np.sqrt(l_h_ / l_h))
            alpha_temp = alpha + beta_h * (alpha - alpha_)
            alpha_ = alpha
            alpha = projection_simplex_sort_2d(alpha_temp + (u.T @ (d_x * (meth_frequency - u @ alpha_temp))) / l_h)
            l_h_ = l_h

        l_w = (np.linalg.norm(alpha[-n_u:]) * 
               np.linalg.norm(alpha[-n_u:].T) * 
               np.linalg.norm(d_x.astype(np.float64)))

        cf = cost_f_w(meth_frequency, u, alpha, d_x)

        if abs(cf - cf_0) < tol:
            break

    return u, alpha
    


# @njit
def mdwbssmf_deconv(u, R, alpha, meth_frequency, d_x, R_trunc, n_u, n_iter1=100000, n_iter2=50, tol=1e-3):
    nb_cpg, nb_celltypes = R_trunc.shape
    a1 = 1.0
    a2 = 1.0
    u_ = u.copy()
    alpha_ = alpha.copy()
    
    l_w = (np.linalg.norm(alpha[-n_u:]) * 
           np.linalg.norm(alpha[-n_u:].T) * 
           np.linalg.norm(d_x.astype(np.float64)))
    l_w_ = l_w
    
    l_h = (np.linalg.norm(R.T) * 
           np.linalg.norm(d_x.astype(np.float64)) * 
           np.linalg.norm(R))
    l_h_ = l_h
    
    cf = cost_f_w(meth_frequency, R, alpha, d_x)  
    
    for k in range(n_iter1):
        cf_0 = cf

        u, u_, a1, l_w_ = update_u(u, alpha, n_iter2, a1, l_w_, l_w, u_, meth_frequency, R_trunc, n_u, d_x)
        R = np.hstack((R_trunc, u.reshape(-1, n_u)))

        l_h = (np.linalg.norm(R.T) * 
               np.linalg.norm(d_x.astype(np.float64)) * 
               np.linalg.norm(R))

        alpha, alpha_, a2, l_h_ = update_alpha(n_iter2, alpha, a2, l_h_, l_h, alpha_, R, d_x, meth_frequency)

        l_w = (np.linalg.norm(alpha[-n_u:]) * 
               np.linalg.norm(alpha[-n_u:].T) * 
               np.linalg.norm(d_x.astype(np.float64)))

        cf = cost_f_w(meth_frequency, R, alpha, d_x)

        if abs(cf - cf_0) < tol:
            break

    return u, alpha


def bt_ci(confidence_level, n_bootstrap, n_u, meth_f, counts, ref, init_option, n_iter1, n_iter2, tol, header, outdir, samples):
    supervised = n_u == 0
    a = 1 - confidence_level / 100
    lower_percentile = 100 * (a / 2)  # Lower percentile for confidence interval
    upper_percentile = 100 * (1 - (a / 2))  # Upper percentile for confidence interval

    proportions_list = [[] for k in range(meth_f.shape[1])]
    if not supervised:
        ref_estimate_list = [[] for k in range(n_u)]

    for i in tqdm.tqdm(range(n_bootstrap)):
        meth_f_resampled, counts_resampled, ref_resampled = resample(meth_f, counts, ref)

        if(not supervised):
            u, R, alpha = init_BSSMF_md(init_option, meth_f_resampled, counts_resampled, ref_resampled, n_u, rb_alg=fs_irls)
            ref_estimate, proportions = mdwbssmf_deconv(u, R, alpha, meth_f_resampled, counts_resampled, ref_resampled, n_u, n_iter1, n_iter2, tol)
            for k in range(n_u):
                ref_estimate_list[k].append(ref_estimate[:,k:k+1])
        else:
            alpha_tab = []
            for k in range(meth_f.shape[1]):
                alpha_tab.append(fs_irls(counts_resampled[:,k:k+1] * meth_f_resampled[:,k:k+1], counts_resampled[:,k:k+1], ref_resampled))
            proportions = np.concatenate(alpha_tab, axis = 1)
    
        for j in range(meth_f.shape[1]):
            proportions_list[j].append(proportions[:,j:j+1])

    
    results = []

    proportions_array = [np.array(proportions_list[k]) for k in range(meth_f.shape[1])]

    lower_bounds_proportions = [np.percentile(proportions_array[k], lower_percentile, axis=0) for k in range(meth_f.shape[1])]
    upper_bounds_proportions = [np.percentile(proportions_array[k], upper_percentile, axis=0) for k in range(meth_f.shape[1])]

    unknown_header = []
    if not supervised:
        unknown_header = ["unknown_cell_" + str(i + 1) for i in range(n_u)]
    cell_types = header + unknown_header
    proportions_dict = {}
    for i in range(meth_f.shape[1]): 
        sample_column = []
        for k in range(ref.shape[1] + n_u):  
            sample_column.append((lower_bounds_proportions[i][k][0], upper_bounds_proportions[i][k][0]))
        proportions_dict[f'Sample_{i+1}'] = sample_column

    proportions_df = pd.DataFrame(proportions_dict, index=cell_types)
    proportions_df.columns = samples
    proportions_df.index.name = 'Cell Type'
    proportions_df.to_csv(outdir + '/confidence_interval_celltypes_proportions.csv', index = True)

    results.append(proportions_df)
        
    if(not supervised):
        ref_estimate_array = [np.array(ref_estimate_list[k]) for k in range(n_u)]

        lower_bounds_ref = [np.percentile(ref_estimate_array[k], lower_percentile, axis=0) for k in range(n_u)]
        upper_bounds_ref = [np.percentile(ref_estimate_array[k], upper_percentile, axis=0) for k in range(n_u)]

        ref_estimate_dict = {}
        for k in range(n_u): 
            unknown_column = []
            for j in range(ref.shape[0]): 
                unknown_column.append((lower_bounds_ref[k][j][0], upper_bounds_ref[k][j][0]))
            ref_estimate_dict[unknown_header[k]] = unknown_column

        ref_estimate_df = pd.DataFrame(ref_estimate_dict)

        ref_estimate_df.to_csv(outdir + '/confidence_interval_methylation_estimate.csv', index = False)
        results.append(ref_estimate_df)


    return results


def plot_proportions(df, ci_df, outdir):

    unique_ct = list(df.index)

    colors = sns.color_palette(cc.glasbey, len(unique_ct))

    color_mapping = {barcode: color for barcode, color in zip(unique_ct, colors)}
    clrs = [color_mapping[barcode] for barcode in unique_ct]
    plt.figure(figsize=(12, 8))
    ax = df.T.plot(kind='bar', stacked=True, figsize=(10, 6), color=clrs)

    plt.title('Proportion of Cell Types in Each Sample')
    plt.ylabel('Proportion')
    plt.xlabel('Samples')

    plt.legend(title='Cell Types', bbox_to_anchor=(1.05, 1), loc='upper left')

    outdir_plots = outdir + '/plots'
    if not os.path.exists(outdir_plots):
        os.mkdir(outdir_plots)
    plt.savefig(outdir_plots + '/proportions_stackedbar.png', dpi=300, bbox_inches='tight')

    sns.set(style="whitegrid")

    for sample in df.columns: 
        plt.figure(figsize=(12, 8))
        if(not ci_df.empty):
            ax = sns.barplot(x=df.index, y=df[sample], palette=clrs, ci=None)

            ci_values = ci_df[sample].apply(lambda x: (x[0], x[1])) 

            lower_bounds = np.array([ci[0] for ci in ci_values])
            upper_bounds = np.array([ci[1] for ci in ci_values])

            lower_error = abs(df[sample].values - lower_bounds)  
            upper_error = abs(upper_bounds - df[sample].values)  

            ax.errorbar(x=np.arange(len(df.index)), 
                        y=df[sample], 
                        yerr=[lower_error, upper_error],  
                        fmt='none',  
                        ecolor='black',  
                        capsize=5,  
                        capthick=2)  
        else:
            sns.barplot(x=df.index, y=df[sample], palette=clrs)

        plt.xlabel('Cell Types')
        plt.ylabel('Proportion')
        plt.title(f'Proportion of Cell Types in {sample}')
    
        plt.xticks(rotation=90)
    
        plt.savefig(outdir_plots + '/proportions_bar_' + sample[:-4] + '.png', dpi=300, bbox_inches='tight')

    print("Plots generated in " + outdir_plots)
    
    

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description="DeMethify - Partial reference-based Methylation Deconvolution")

    # Add regular arguments
    parser.add_argument('--methfreq', nargs='+', type=str, required=True, help='Methylation frequency file path (values between 0 and 1)')
    parser.add_argument('--ref', nargs='?', type=str, help='Methylation reference matrix file path')
    parser.add_argument('--iterations', default=[10000, 20], nargs=2, type=int, help='Numbers of iterations for outer and inner loops (default = 10000, 20)')
    parser.add_argument('--nbunknown', nargs=1, type=int, help="Number of unknown cell types to estimate ")
    parser.add_argument('--termination', nargs=1, type=float, default=1e-2, help='Termination condition for cost function (default = 1e-2)')
    parser.add_argument('--init', nargs="?", default='uniform', help='Initialisation option (default = random uniform)')
    parser.add_argument('--outdir', nargs='?', required=True, help='Output directory')
    parser.add_argument('--fillna', action="store_true", help='Replace every NA by 0 in the given data')
    parser.add_argument('--ic', nargs="?", help='Select number of unknown cell types by minimising an information criterion (AIC or BIC)')
    parser.add_argument('--confidence', nargs=2, type=int, help='Outputs bootstrap confidence intervals, takes confidence level and boostrap iteration numbers as input.')
    parser.add_argument('--plot', action="store_true", help='Plot cell type proportions estimates for each sample, eventually with confidence intervals. ')

    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group()

    # Add conflicting arguments to the group
    group.add_argument('--bedmethyl', action='store_true', help="Flag to indicate that the input will be bedmethyl files, modkit style")
    group.add_argument('--counts', nargs='+', type=str, help='Read counts file path')
    group.add_argument('--noreadformat', action="store_true", help="Flag to use when the data isn\'t using the read format (e.g Illumina epic arrays)")

    # Parse the arguments
    args = parser.parse_args()

    if args.ic:
        if args.nbunknown:
            sys.stderr.write("Error: --ic cannot be used with --nbunknown.\n")
            sys.exit(1)
        if not args.ref:
            sys.stderr.write("Error: --ic requires --ref to be specified.\n")
            sys.exit(1)
    
    print(logo)
    
    # create output dir if it doesn't exist
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

    # print(args.methfreq)
    # args.methfreq = [bla.split("/")[-1] for bla in args.methfreq]
    # print(args.methfreq)
        
    # deconvolution
    time_start = time()

    if(args.confidence):
        bt_results = bt_ci(args.confidence[0], args.confidence[1], args.nbunknown[0], meth_f, counts, ref, args.init, args.iterations[0], args.iterations[1], args.termination, header, outdir, args.methfreq)

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
