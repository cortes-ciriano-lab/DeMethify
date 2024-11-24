import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.special import gammaln
from scipy.spatial.distance import pdist
from sklearn.model_selection import KFold
import tqdm
from .deconvolution import *

# Computes the corrected Bayesian Information Criterion (BIC) for selecting the best model based on the trade-off between complexity and goodness of fit.
def compute_bic(cost, n_u, n_cpg, n_ct, n_samples):
    l = n_samples * n_cpg
    k = n_u * n_cpg + (n_ct + n_u - 1) * n_samples
    bic = 2 * np.log(cost) * k * np.log(l) + (k * np.log(l) * (k + 1)) / (l - k - 1)
    return bic

# Computes the corrected Akaike Information Criterion (AIC) to compare different models and select the best one.
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

# Calculates Brunet's Cophenetic Correlation Coefficient (CCC) to assess the stability of clustering across different factorization runs.
def compute_ccc(alpha_runs):
    consensus_matrix = compute_consensus_matrix(alpha_runs)
    distance_matrix = pdist(consensus_matrix, metric="euclidean")
    linkage_matrix = linkage(distance_matrix, method="average")
    ccc, _ = cophenet(linkage_matrix, distance_matrix)
    return ccc

def run_deconvolution(meth_f, counts, ref, n_u, init_option, seed, iter1, iter2, tol):
    if ref is not None:
        u, R, alpha = init_BSSMF_md(init_option, meth_f, counts, ref, n_u, rb_alg=wls_intercept, seed=seed)
        u, alpha = mdwbssmf_deconv(u, R, alpha, meth_f, counts, ref, n_u,  n_iter1=iter1, n_iter2=iter2, tol= tol)
        R = np.hstack((ref, u.reshape(-1, n_u)))
    else:
        u, alpha = unsupervised_deconv(meth_f, n_u, counts, init_option,  n_iter1=iter1, n_iter2=iter2, tol=tol, seed=seed)
        R = u
    return u, R, alpha
    
# Implements Owen and Perry's bi-cross validation technique for selecting the best model rank.
def bicross_validation(meth_f, n_u, counts, iter1, iter2, tol, n_folds=10, seed=None, ref=None, init_option="uniform_", fraction=0.3):
    np.random.seed(seed)
    n_cpg, n_samples = meth_f.shape

    total_press = 0
    best_u = None
    best_alpha = None
    min_error = float('inf')

    for _ in range(n_folds):
        train_mask = np.random.rand(*meth_f.shape) < fraction
        test_mask = ~train_mask

        if np.sum(test_mask) == 0 or np.sum(train_mask) == 0:
            continue  


        u, R, alpha = run_deconvolution(meth_f * train_mask, counts * train_mask, ref, n_u, init_option, seed,  iter1, iter2, tol)

        Y_pred = R @ alpha  

        # Evaluate test error only for unknown components
        test_error = np.linalg.norm((meth_f - Y_pred) * test_mask, "fro") ** 2 / np.sum(test_mask)
        total_press += test_error

        if test_error < min_error:
            min_error = test_error
            best_u = u
            best_alpha = alpha

    mean_press = total_press / n_folds if n_folds > 0 else float('inf')
    return total_press, best_u, best_alpha


def estimate_H1(Y, W1, counts):
    if W1 is None:
        return None

    n_features, n_samples = Y.shape
    H1 = np.zeros((W1.shape[1], n_samples))  
    for i in range(n_samples):
        H1[:, i] = wls_intercept(Y[:, i], counts[:, i] ,W1)
    return H1


# Selects the rank for matrix factorization using Minka's PCA method, adapted to handle cases with partial reference information.
def select_rank_minka(Y, counts, W1=None):
    n_features, n_samples = Y.shape

    if W1 is not None:
        H1 = estimate_H1(Y, W1, counts)

        Residual = Y - W1 @ H1
    else:
        Residual = Y

    
    U, svals, Vt = np.linalg.svd(Residual, full_matrices=False)

    cov_evals = svals ** 2 / n_samples

    ranks = np.arange(1, len(svals))
    log_liks = np.empty_like(ranks, dtype=float)

    for idx, rank in enumerate(ranks):
        log_liks[idx] = get_log_lik_partial(cov_evals, rank, (n_samples, n_features))

    log_liks = pd.Series(log_liks, index=ranks)
    log_liks.name = 'log_lik'
    log_liks.index.name = 'rank'

    rank_est = log_liks.idxmax()
    return rank_est, {'log_liks': log_liks, 'cov_evals': cov_evals}


def get_log_lik_partial(cov_evals, rank, shape):
    n_samples, n_features = shape
    if not 1 <= rank <= n_features - 1:
        raise ValueError("The tested rank should be in [1, n_features - 1]")

    eps = 1e-15
    if cov_evals[rank - 1] < eps:
        return -np.inf

    pu = -rank * np.log(2.0)
    for i in range(1, rank + 1):
        pu += gammaln((n_features - i + 1) / 2.0) - np.log(np.pi) * (n_features - i + 1) / 2.0

    pl = np.sum(np.log(cov_evals[:rank]))
    pl = -pl * n_samples / 2.0

    v = max(eps, np.sum(cov_evals[rank:]) / (n_features - rank))
    pv = -np.log(v) * n_samples * (n_features - rank) / 2.0

    m = n_features * rank - rank * (rank + 1.0) / 2.0
    pp = np.log(2.0 * np.pi) * (m + rank) / 2.0

    pa = 0.0
    spectrum_ = cov_evals.copy()
    spectrum_[rank:n_features] = v
    for i in range(rank):
        for j in range(i + 1, len(cov_evals)):
            pa += np.log((cov_evals[i] - cov_evals[j]) * (1.0 / spectrum_[j] - 1.0 / spectrum_[i])) + np.log(n_samples)

    ll = pu + pl + pv + pp - pa / 2.0 - rank * np.log(n_samples) / 2.0
    return ll




# Evaluates different criteria to determine the optimal number of unknown cell types, then performs deconvolution using this selected number.
def evaluate_best_ic(meth_f, ref, counts, init_option, ic, seed, iter1, iter2, tol, n_restarts=5):
    max_range = meth_f.shape[1]
    n_u_values = range(1, 25 + 1)
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


    if ic == "minka":
        best_n_u, minka_result = select_rank_minka(meth_f, counts, ref)
        best_ic = -minka_result['log_liks'][best_n_u]
        best_u_overall, _ , best_alpha_overall = run_deconvolution(meth_f, counts, ref, best_n_u, init_option, seed)
        return best_u_overall, best_alpha_overall, best_n_u, (-minka_result['log_liks']).to_list()

    for n_u in tqdm.tqdm(n_u_values):
        if ic == "CCC":
            alpha_runs = []
            for restart in range(n_restarts):
                u, R, alpha = run_deconvolution(meth_f, counts, ref, n_u, init_option, seed + restart,  iter1, iter2, tol)
                alpha_runs.append(alpha)
            ic_result = - compute_ccc(alpha_runs)

        elif ic == "BCV":
            ic_result, u, alpha = bicross_validation(meth_f, n_u, counts, iter1, iter2, tol, fraction=0.3, n_folds=n_restarts, seed=seed, ref=ref, init_option=init_option)


        else:  
            u, R, alpha = run_deconvolution(meth_f, counts, ref, n_u, init_option, seed,  iter1, iter2, tol)
            cost = cost_f_w(meth_f, R, alpha, counts)
            ic_result = compute_bic(cost, n_u, n_cpg, n_ct, n_samples) if ic == "BIC" else compute_aic(cost, n_u, n_cpg, n_ct, n_samples)


        list_result.append(ic_result)

        if ic_result < best_ic:
            best_ic = ic_result
            best_n_u = n_u
            best_alpha_overall = alpha
            best_u_overall = u

    return best_u_overall, best_alpha_overall, best_n_u, list_result

