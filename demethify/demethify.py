import os
import argparse

import numpy as np
import numpy.random as rd
import pandas as pd
from time import time
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import FastICA


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
        W = np.divide(d_x, (gamma * (1 - gamma) + 1e-16))
        z = np.divide(x, d_x)
        z[np.isnan(z)] = 1e-16
        a =  wls_deconv(R_full, z, W)
        
        if(np.mean(abs(a - alpha)) / np.mean(abs(alpha)) < tol):
            break
        else:
            alpha = a
    
    return alpha



def cost_f_w(y, R, alpha, d_x):
    return np.linalg.norm(d_x * (y - R @ alpha))**2



def foo(tab):
    n = tab.shape[0]
    for k in range(n):
        if(tab[k][0] > 0):
            num = k
            
    return num



def simplex_proj(v):
    n = v.shape[0]
    v_ = v[(-v[:, 0]).argsort()]
    v_temp = (1 - np.reshape(np.cumsum(v_), (n,1))) * (1 / np.reshape(np.array(range(1,n + 1)), (n, 1))) + v_
    rho = foo(v_temp)
    lamb = (1 - np.cumsum(v_)[rho]) / (rho + 1)
    w = (v + lamb).clip(min=0)
    return w



def simplex_proj_md(v):
    x, y = v.shape
    tab_v = []
    for k in range(y):
        tab_v.append(simplex_proj(np.reshape(v[:,k], (x, 1))))
        
    v = np.concatenate(tab_v, axis = 1)
    
    return v



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
# =============================================================================
#     elif(init_option == 'PRmeth'): ## Hierarchichal clustering by PRmeth
#         r['source']('PRMeth-main/PRMeth/R/prmeth.R')
#         RefFreeCellMixInitialize = robjects.globalenv['RefFreeCellMixInitialize']
#         u = RefFreeCellMixInitialize(meth_frequency, n_u,"binary")
# =============================================================================
    # elif(init_option == ''): ## Hierarchichal clustering with sklearn
    R = np.c_[R_trunc, u]
    
    for k in range(nb):
        alpha_tab.append(rb_alg(d_x[:,k:k+1] * meth_frequency[:,k:k+1], d_x[:,k:k+1], R))
    alpha = np.concatenate(alpha_tab, axis = 1)
    
    if(alpha[-n_u:][0].all() == 0.0):
        alpha[-n_u:][0] = 1e-10
        alpha[:-n_u] = (1 - 1e-10) * alpha[:-n_u]
        
    return u, R, alpha



def mdwbssmf_deconv(meth_frequency, d_x, R_trunc, n_u, init_option, n_iter1 = 1000, n_iter2 = 50, tol = 1e-3):
    nb_cpg, nb_celltypes = R_trunc.shape
    a1 = 1
    a2 = 1
    u, R, alpha = init_BSSMF_md(init_option, meth_frequency, d_x, R_trunc, n_u, rb_alg = fs_irls)
    u_ = u
    alpha_ = alpha
    l_w = np.linalg.norm(alpha[-n_u:]) * np.linalg.norm(alpha[-n_u:].T) * np.linalg.norm(d_x)
    l_w_ = l_w
    l_h = np.linalg.norm(R.T) * np.linalg.norm(d_x) * np.linalg.norm(R)
    l_h_ = l_h
    cf = cost_f_w(meth_frequency, R, alpha, d_x)
    
    for k in range(n_iter1):
        cf_0 = cf
        for i in range(n_iter2):
            a0 = a1
            a1 = (1 + np.sqrt(1 + 4 * a0 * a0)) / 2
            beta_w = min((a0 - 1) / a1, 0.9999 *  np.sqrt(l_w_ / l_w))
            u_temp = u + beta_w * (u - u_)
            u_ = u
            u = (u_temp + (d_x * ((meth_frequency - R_trunc @ alpha[:-n_u] - u_temp @ alpha[-n_u:])) @ alpha[-n_u:].T) / l_w).clip(min = 0, max = 1)
            l_w_ = l_w
        
        R = np.c_[R_trunc, u]
        l_h = np.linalg.norm(R.T) * np.linalg.norm(d_x) * np.linalg.norm(R)
        
        for j in range(n_iter2):
            a0 = a2
            a1 = (1 + np.sqrt(1 + 4 * a0 * a0)) / 2
            beta_h = min((a0 - 1) / a2, 0.9999 *  np.sqrt(l_h_ / l_h))
            alpha_temp = alpha + beta_h * (alpha - alpha_)
            alpha_ = alpha
            alpha = simplex_proj_md((alpha_temp + (R.T @ (d_x * (meth_frequency - R @ alpha_temp))) / l_h))
            l_h_ = l_h
        
        l_w = np.linalg.norm(alpha[-n_u:]) * np.linalg.norm(alpha[-n_u:].T) * np.linalg.norm(d_x)
        
        cf = cost_f_w(meth_frequency, R, alpha, d_x)
        
        if(k % 100 == 0):
            print(str(k) + ' iterations, ', abs(cf - cf_0))
        
        if(abs(cf - cf_0) < tol):
            break
        
    return R, alpha



def main():
    
    parser = argparse.ArgumentParser(description="DeMethify - Partial reference-based Methylation Deconvolution")
    parser.add_argument('--methfreq', nargs='?', type=str, required=True, help='Methylation frequency CSV file (values between 0 and 1)')
    parser.add_argument('--noreadformat', action ="store_true", help="Flag to use when the data isn't using the read format (like Illumina epic arrays)")
    parser.add_argument('--counts', nargs='?', type=str, help='Read counts CSV file')
    parser.add_argument('--ref', nargs='?', type=str, required=True, help='Reference methylation matrix CSV file')
    parser.add_argument('--iterations', default = [50000,50], nargs= 2, type=int, help='Numbers of iterations for outer and inner loops (default = 50000, 50)')
    parser.add_argument('--nbunknown', nargs= 1, type=int, help="Number of unknown cell types to estimate ")
    parser.add_argument('--termination', nargs= 1, type=float, default = 1e-2 , help='Termination condition for cost function (default = 1e-2)')
    parser.add_argument('--outdir', nargs='?', required=True, help='Output directory (must be empty)')
    args = parser.parse_args()
    
    print(logo)
    
    # create output dir if it doesn't exist
    outdir = os.path.join(os.getcwd(), args.outdir)
    if not os.path.exists(outdir):
        print(f'Creating directory {outdir} to store results')
        os.mkdir(outdir)
    elif os.listdir(outdir):
        exit(f'Output directory "{outdir}" already exists and contains files. Please remove the files or supply a different directory name.')
        
    
    # read csv files
    meth_f = pd.read_csv(args.methfreq).values
    ref = pd.read_csv(args.ref).values
    if(not(args.noreadformat)):
        counts = pd.read_csv(args.counts).values
    else:
        counts = np.ones_like(meth_f)
        
    
    # deconvolution
    time_start = time()
    if(args.nbunknown[0] > 0):
        ref_estimate, proportions = mdwbssmf_deconv(meth_f, counts, ref, args.nbunknown[0], 'ICA', n_iter1 = args.iterations[0], n_iter2 = args.iterations[1], tol = args.termination)
        pd.DataFrame(ref_estimate).to_csv(outdir + '/methylation_profile_estimate.csv', index = False)
    elif(args.nbunknown[0] == 0):
        prop_tab = []
        for k in range(ref.shape[1] - 1):
            prop_tab.append(fs_irls(counts[:,k:k+1] * meth_f[:,k:k+1], counts[:,k:k+1], ref))
        proportions = np.concatenate(prop_tab, axis = 1)
    else:
        exit(f'Invalid number of unknown value! : "{args.nbunknown}" ')
        
    time_tot = time() - time_start
    
    # saving output files
    pd.DataFrame(proportions).to_csv(outdir + '/celltypes_proportions.csv', index = False)
    
    f = open(os.path.join(outdir, 'time.log'), "w+")
    f.write("Total execution time = " + str(time_tot) + " s")
    f.close()
    
    
    
if __name__ == "__main__":
	main()