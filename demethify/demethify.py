import os
import argparse
import numpy as np
import numpy.random as rd
import pandas as pd
from time import time
from numba import njit
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import FastICA
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


def wls_deconv(ref, samples, weights):
    reg = LinearRegression(fit_intercept = False, positive = True).fit(ref, samples, weights.ravel())
    temp = reg.coef_.T
               
    P_deconv = temp / temp.sum()
               
    return P_deconv



def fs_irls(x, d_x, R_full, tol = 1e-14, n_iter = 10000):
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
    return np.linalg.norm(d_x * (y - R @ alpha))

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
    if(init_option == 'uniform'): ## Random uniform
        u = rd.uniform(size = (R_trunc.shape[0], n_u)) 
    elif(init_option == 'ICA'): ## Independent component analysis
        tt = FastICA(n_components = n_u)
        u = tt.fit_transform(meth_frequency)
        u_ = (u - min(u)) 
        u = u_ / max(u_)
        
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
    


@njit
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

        R = np.hstack((R_trunc, u.reshape(-1, 1)))

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

def main():

    # Create the argument parser
    parser = argparse.ArgumentParser(description="DeMethify - Partial reference-based Methylation Deconvolution")

    # Add regular arguments
    parser.add_argument('--methfreq', nargs='+', type=str, required=True, help='Methylation frequency file path (values between 0 and 1)')
    parser.add_argument('--ref', nargs='?', type=str, required=True, help='Methylation reference matrix file path')
    parser.add_argument('--iterations', default=[50000, 50], nargs=2, type=int, help='Numbers of iterations for outer and inner loops (default = 50000, 50)')
    parser.add_argument('--nbunknown', nargs=1, type=int, help="Number of unknown cell types to estimate ")
    parser.add_argument('--termination', nargs=1, type=float, default=1e-2, help='Termination condition for cost function (default = 1e-2)')
    parser.add_argument('--init', nargs="?", help='Initialisation option')
    parser.add_argument('--outdir', nargs='?', required=True, help='Output directory (must be empty)')
    parser.add_argument('--fillna', action="store_true", help='Replace every NA by 0 in the given data')

    # Create a mutually exclusive group
    group = parser.add_mutually_exclusive_group()

    # Add conflicting arguments to the group
    group.add_argument('--bedmethyl', action='store_true', help="Flag to indicate that the input will be bedmethyl files, modkit style")
    group.add_argument('--counts', nargs='+', type=str, help='Read counts file path')
    group.add_argument('--noreadformat', action="store_true", help="Flag to use when the data isn\'t using the read format (e.g Illumina epic arrays)")

    # Parse the arguments
    args = parser.parse_args()
    
    print(logo)
    
    # create output dir if it doesn't exist
    outdir = os.path.join(os.getcwd(), args.outdir)
    if not os.path.exists(outdir):
        print(f'Creating directory {outdir} to store results')
        os.mkdir(outdir)
    
    # read bedmethyl files (modkit output)
    if(args.bedmethyl):
        ref = pd.read_csv(args.ref, sep='\t').iloc[:, 3:]
        header = list(ref.columns)
        ref = ref.values
        list_meth_freq = []
        list_counts = []
        for bed in args.methfreq:
            temp = pd.read_csv(bed, sep='\t')
            if(args.fillna):
                temp = temp.fillna(0, inplace = True)
            list_meth_freq.append(temp["percent_modified"].values / 100)
            list_counts.append(temp["valid_coverage"].values)
        meth_f = np.column_stack(list_meth_freq)
        counts = np.column_stack(list_counts)

    # read csv files
    else:
        meth_f = pd.read_csv(args.methfreq).values
        ref = pd.read_csv(args.ref)
        header = list(ref.columns)
        ref = ref.values
        if(args.fillna):
            meth_f.fillna(0, inplace = True)
            ref.fillna(0, inplace = True)
        if(not(args.noreadformat)):
            counts = pd.read_csv(args.counts).values
            if(args.fillna):
                counts.fillna(0, inplace = True)
        else:
            counts = np.ones_like(meth_f)
        
    
    # deconvolution
    time_start = time()
    if(args.nbunknown[0] > 0 and meth_f.shape[1] > 1):
        u, R, alpha = init_BSSMF_md(args.init, meth_f, counts, ref, args.nbunknown[0], rb_alg = fs_irls)
        ref_estimate, proportions = mdwbssmf_deconv(u, R, alpha, meth_f, counts, ref, args.nbunknown[0], n_iter1 = args.iterations[0], n_iter2 = args.iterations[1], tol = args.termination)
        unknown_header = ["unknown_cell_" + str(i + 1) for i in range(args.nbunknown[0])]
        header += unknown_header
        pd.DataFrame(ref_estimate).to_csv(outdir + '/methylation_profile_estimate.csv', index = False, header=unknown_header)
    elif(args.nbunknown[0] == 0 or meth_f.shape[1] == 1):
        prop_tab = []
        for k in range(ref.shape[1] - 1):
            prop_tab.append(fs_irls(counts[:,k:k+1] * meth_f[:,k:k+1], counts[:,k:k+1], ref))
        proportions = np.concatenate(prop_tab, axis = 1)
    else:
        exit(f'Invalid number of unknown value! : "{args.nbunknown}" ')
        
    time_tot = time() - time_start
    
    # saving output files
    proportions = pd.DataFrame(proportions)
    proportions.index = header
    proportions.to_csv(outdir + '/celltypes_proportions.csv', index = True, header = args.methfreq)
    
    f = open(os.path.join(outdir, 'time.log'), "w+")
    f.write("Total execution time = " + str(time_tot) + " s")
    f.close()

    print("All demethified! Results in " + outdir)
    
    
    
if __name__ == "__main__":
	main()
