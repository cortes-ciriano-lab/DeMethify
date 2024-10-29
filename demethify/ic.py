import numpy as np
import pandas as pd
from .deconvolution import *


def compute_bic(cost, n_u, n_cpg, n_ct, n_samples):
    k = n_u * n_cpg + (n_ct + n_u - 1) * n_samples
    bic = 2 * np.log(cost) + k * np.log(n_cpg)
    return bic

def compute_aic(cost, n_u, n_cpg, n_ct, n_samples):
    k = n_u * n_cpg + (n_ct + n_u - 1) * n_samples
    aic = 2 * (np.log(cost) + k)
    return aic

# @njit
def evaluate_best_ic(meth_f, ref, counts, init_option, ic, seed):
    # Example parameters
    max_range=meth_f.shape[1]
    n_u_values = range(0, max_range + 1)  
    n_cpg, n_samples = meth_f.shape
    if ref != None:
        n_ct = ref.shape[1]
    else:
        n_ct = 0

    best_ic = float('inf')
    best_n_u = None
    best_u_overall = None
    best_alpha_overall = None

    for n_u in n_u_values:
        # Run the deconvolution for the current n_u
        if(n_u >= 1):
            if ref != None:
                u, R, alpha = init_BSSMF_md(init_option, meth_f, counts, ref, n_u, rb_alg = fs_irls, seed=seed)
                u, alpha = mdwbssmf_deconv(u, R, alpha, meth_f, counts, ref, n_u, 10000, 20, 1e-2, seed=seed)
                R = np.hstack((ref, u.reshape(-1, n_u)))
            else:
                u, alpha = unsupervised_deconv(meth_f, n_u, counts, init_option, 10000, 20, 1e-2, seed=seed)
                R = u

        else:
            if ref != None:
                alpha_tab = []
                for k in range(n_samples):
                    alpha_tab.append(fs_irls(counts[:,k:k+1] * meth_f[:,k:k+1], counts[:,k:k+1], ref, seed=seed))
                alpha = np.concatenate(alpha_tab, axis = 1)
                R = ref
            else:
                continue

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