import numpy as np
import pandas as pd
import tqdm
from sklearn.utils import resample
from .deconvolution import *


def bt_ci(confidence_level, n_bootstrap, n_u, meth_f, counts, ref, init_option, n_iter1, n_iter2, tol, header, outdir, samples, purity):
    supervised = n_u == 0
    a = 1 - confidence_level / 100
    lower_percentile = 100 * (a / 2)  # Lower percentile for confidence interval
    upper_percentile = 100 * (1 - (a / 2))  # Upper percentile for confidence interval
    yes_purity= False
    if purity:
        yes_purity = True
        purity = np.array(purity)/100.0

    proportions_list = [[] for k in range(meth_f.shape[1])]
    if not supervised:
        ref_estimate_list = [[] for k in range(n_u)]

    for i in tqdm.tqdm(range(n_bootstrap)):
        meth_f_resampled, counts_resampled, ref_resampled = resample(meth_f, counts, ref)

        if(not supervised):
            if yes_purity:
                
                u, R, alpha = init_BSSMF_md_p(init_option, meth_f, counts, ref, n_u, purity, rb_alg = fs_irls)
                ref_estimate, proportions = mdwbssmf_deconv_p(u, R, alpha, meth_f_resampled, counts_resampled, ref_resampled, n_u, purity, n_iter1, n_iter2, tol)
            else:
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