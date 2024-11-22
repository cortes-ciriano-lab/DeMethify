import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import pdist
import tqdm
from .deconvolution import *

def compute_bic(cost, n_u, n_cpg, n_ct, n_samples):
    l = n_samples * n_cpg
    k = n_u * n_cpg + (n_ct + n_u - 1) * n_samples
    bic = 2 * np.log(cost) * k * np.log(l) + (k * np.log(l) * (k + 1)) / (l - k - 1)
    return bic

def compute_aic(cost, n_u, n_cpg, n_ct, n_samples):
    l = n_samples * n_cpg
    k = n_u * n_cpg + (n_ct + n_u - 1) * n_samples
    aic = l * np.log(cost / l) + 2 * k + (2 * k * (k + 1)) / (l - k - 1)
    return aic

def compute_consensus_matrix(alpha_runs):
    n_samples = alpha_runs[0].shape[1]  
    n_runs = len(alpha_runs)  
    consensus_matrix = np.zeros((n_samples, n_samples))

    for alpha in alpha_runs:
        cluster_assignments = np.argmax(alpha, axis=0)
        for i in range(n_samples):
            for j in range(n_samples):
                if cluster_assignments[i] == cluster_assignments[j]:
                    consensus_matrix[i, j] += 1

    consensus_matrix /= n_runs
    return consensus_matrix

def compute_ccc(alpha_runs):
    consensus_matrix = compute_consensus_matrix(alpha_runs)
    distance_matrix = pdist(consensus_matrix, metric="euclidean")
    linkage_matrix = linkage(distance_matrix, method="average")
    ccc, _ = cophenet(linkage_matrix, distance_matrix)
    return ccc

def run_deconvolution(meth_f, counts, ref, n_u, init_option, seed):
    if ref is not None:
        u, R, alpha = init_BSSMF_md(init_option, meth_f, counts, ref, n_u, rb_alg=wls_intercept, seed=seed)
        u, alpha = mdwbssmf_deconv(u, R, alpha, meth_f, counts, ref, n_u, 10000, 20, 1e-2)
        R = np.hstack((ref, u.reshape(-1, n_u)))
    else:
        u, alpha = unsupervised_deconv(meth_f, n_u, counts, init_option, 10000, 20, 1e-2, seed=seed)
        R = u
    return u, R, alpha
    

def evaluate_best_ic(meth_f, ref, counts, init_option, ic, seed, n_restarts=5):
    max_range = meth_f.shape[1]
    n_u_values = range(1, 30 + 1)
    n_cpg, n_samples = meth_f.shape

    if ref is not None:
        n_ct = ref.shape[1]
    else:
        n_ct = 0

    best_ic = float("inf") 
    best_n_u = None
    best_u_overall = None
    best_alpha_overall = None
    list_result = []

    for n_u in tqdm.tqdm(n_u_values):
        if ic == "CCC":
            alpha_runs = []
            for restart in range(n_restarts):
                u, R, alpha = run_deconvolution(meth_f, counts, ref, n_u, init_option, seed + restart)
                alpha_runs.append(alpha)

            ic_result = -compute_ccc(alpha_runs)


        else:  
            u, R, alpha = run_deconvolution(meth_f, counts, ref, n_u, init_option, seed)
            cost = cost_f_w(meth_f, R, alpha, counts)
            ic_result = compute_bic(cost, n_u, n_cpg, n_ct, n_samples) if ic == "BIC" else compute_aic(cost, n_u, n_cpg, n_ct, n_samples)


        list_result.append(ic_result)

        if ic_result < best_ic:
            best_ic = ic_result
            best_n_u = n_u
            best_alpha_overall = alpha
            best_u_overall = u

    return best_u_overall, best_alpha_overall, best_n_u, list_result

