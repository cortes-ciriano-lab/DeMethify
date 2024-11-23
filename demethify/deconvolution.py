import numpy as np
import numpy.random as rd
import pandas as pd
from numba import njit
from .init_func import *

def set_seed(seed=None):
    if seed is not None:
        rd.seed(seed)

@njit
def cost_f_w(y, R, alpha, d_x):
    # return np.linalg.norm((y - R @ alpha))**2
    return np.linalg.norm(np.sqrt(d_x) * (y - R @ alpha))**2

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

def init_BSSMF_md(init_option, meth_frequency, d_x, R_trunc, n_u, seed=None, rb_alg=wls_intercept):
    set_seed(seed)
    nb = meth_frequency.shape[1]

    if init_option != "uniform_" and n_u > nb:
        init_option = 'uniform_'

    if init_option == 'uniform':
        u = rd.uniform(size=(R_trunc.shape[0], n_u)) 
        alpha_tab = []
        for k in range(nb):
                alpha_tab.append(wls_intercept(meth_frequency[:,k:k+1], d_x[:,k:k+1], np.c_[R_trunc, u]))
        alpha = np.concatenate(alpha_tab, axis = 1)

    elif init_option == 'uniform_':
        u = rd.uniform(size=(R_trunc.shape[0], n_u)) 
        alpha = rd.dirichlet(np.ones(R_trunc.shape[1] + n_u), nb).T

    elif init_option == 'beta': 
        temp = np.ones((R_trunc.shape[0], n_u))
        u = rd.beta(temp * 0.5, temp * 0.5) 
        alpha = rd.dirichlet(np.ones(R_trunc.shape[1] + n_u), nb).T

    elif init_option == 'ICA':
        W, alpha = constrained_nn_ica(meth_frequency, R_trunc,  d_x, rank=n_u, t_tol=1e-1, verbose=0)
        alpha = projection_simplex_sort_2d(alpha)
        u = W[:, R_trunc.shape[1]:]

    elif init_option == 'SVD':
        W, alpha = constrained_nndsvd(meth_frequency, R_trunc,  d_x, rank=n_u, flag=0)
        alpha = projection_simplex_sort_2d(alpha)
        u = W[:, R_trunc.shape[1]:]

    R = np.c_[R_trunc, u]
    if alpha[-n_u:][0].all() == 0.0:
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



def unsupervised_deconv(meth_frequency, n_u, d_x, init_option, n_iter1=100000, n_iter2=20, tol=1e-3, seed=None):
    set_seed(seed)
    nb = meth_frequency.shape[1]
    if init_option != "uniform_" and n_u > nb:
        init_option = 'uniform_'

    if init_option == 'uniform':
        u = rd.uniform(size=(meth_frequency.shape[0], n_u)) 
        alpha_tab = []
        for k in range(nb):
                alpha_tab.append(wls_intercept(meth_frequency[:,k:k+1], d_x[:,k:k+1], np.c_[R_trunc, u]))
        alpha = np.concatenate(alpha_tab, axis = 1)

    elif init_option == 'uniform_':
        u = rd.uniform(size=(meth_frequency.shape[0], n_u)) 
        alpha = rd.dirichlet(np.ones(n_u), nb).T

    elif init_option == 'beta': 
        temp = np.ones((meth_frequency.shape[0], n_u))
        u = rd.beta(temp * 0.5, temp * 0.5) 
        alpha = rd.dirichlet(np.ones(n_u), nb).T

    elif init_option == 'ICA':
        u, alpha = run_nn_ica(meth_frequency, rank=n_u, t_tol=1e-1, verbose=0)
        u = u.clip(0, 1)
        alpha = projection_simplex_sort_2d(alpha)

    elif init_option == 'SVD':
        u, alpha = nndsvd_initialize(meth_frequency, rank=n_u)
        u = u.clip(0, 1)
        alpha = projection_simplex_sort_2d(alpha)
    
    a1 = 1.0
    a2 = 1.0
    u_ = u.copy()
    alpha_ = alpha.copy()

    d = d_x.max() ** 2
    l_w = (np.linalg.norm(alpha[-n_u:])**2) *  d
    l_w_ = l_w
    
    l_h = (np.linalg.norm(u)**2) *d 
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

        l_h = (np.linalg.norm(u)**2) * d

        for j in range(n_iter2):
            a0 = a2
            a2 = (1 + np.sqrt(1 + 4 * a0 * a0)) / 2
            beta_h = min((a0 - 1) / a2, 0.9999 *  np.sqrt(l_h_ / l_h))
            alpha_temp = alpha + beta_h * (alpha - alpha_)
            alpha_ = alpha
            alpha = projection_simplex_sort_2d(alpha_temp + (u.T @ (d_x * (meth_frequency - u @ alpha_temp))) / l_h)
            l_h_ = l_h

        l_w = (np.linalg.norm(alpha[-n_u:])**2) *  d

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

    d = d_x.max() ** 2
    l_w = (np.linalg.norm(alpha[-n_u:])**2) * d
    l_w_ = l_w
    
    l_h = (np.linalg.norm(R)**2) * d
    l_h_ = l_h
    
    cf = cost_f_w(meth_frequency, R, alpha, d_x)  
    
    for k in range(n_iter1):
        cf_0 = cf

        u, u_, a1, l_w_ = update_u(u, alpha, n_iter2, a1, l_w_, l_w, u_, meth_frequency, R_trunc, n_u, d_x)
        R = np.hstack((R_trunc, u.reshape(-1, n_u)))

        l_h = (np.linalg.norm(R)**2) * d

        alpha, alpha_, a2, l_h_ = update_alpha(n_iter2, alpha, a2, l_h_, l_h, alpha_, R, d_x, meth_frequency)

        l_w = (np.linalg.norm(alpha[-n_u:])**2) * d

        cf = cost_f_w(meth_frequency, R, alpha, d_x)

        if abs(cf - cf_0) < tol:
            break

    return u, alpha



def init_BSSMF_md_p(init_option, meth_frequency, d_x, R_trunc, n_u, purity, rb_alg = wls_intercept, seed=None):
    set_seed(seed)
    nb = meth_frequency.shape[1]

    if(init_option != "uniform" and n_u > nb):
        print("The number of unknowns is greater than the number of samples, we'll go with a uniform initialisation. ")
        init_option = 'uniform'
    
    if init_option != "uniform_" and n_u > nb:
        init_option = 'uniform_'

    if init_option == 'uniform':
        u = rd.uniform(size=(R_trunc.shape[0], n_u)) 
        alpha_tab = []
        for k in range(nb):
                alpha_tab.append(wls_intercept(meth_frequency[:,k:k+1], d_x[:,k:k+1], np.c_[R_trunc, u]))
        alpha = np.concatenate(alpha_tab, axis = 1)

    elif init_option == 'uniform_':
        u = rd.uniform(size=(R_trunc.shape[0], n_u)) 
        alpha = rd.dirichlet(np.ones(R_trunc.shape[1] + n_u), nb).T

    elif init_option == 'beta': 
        temp = np.ones((R_trunc.shape[0], n_u))
        u = rd.beta(temp * 0.5, temp * 0.5) 
        alpha = rd.dirichlet(np.ones(R_trunc.shape[1] + n_u), nb).T

    elif init_option == 'ICA':
        W, alpha = constrained_nn_ica(meth_frequency, R_trunc,  d_x, rank=n_u, t_tol=1e-1, verbose=0)
        alpha = np.vstack((purity * projection_simplex_sort_2d(alpha[:-n_u]), (1 - purity) * projection_simplex_sort_2d(alpha[-n_u:])))
        u = W[:, R_trunc.shape[1]:]

    elif init_option == 'SVD':
        W, alpha = constrained_nndsvd(meth_frequency, R_trunc,  d_x, rank=n_u, flag=0)
        alpha = np.vstack((purity * projection_simplex_sort_2d(alpha[:-n_u]), projection_simplex_sort_2d(alpha[-n_u:])))
        u = W[:, R_trunc.shape[1]:]

    R = np.c_[R_trunc, u]
        
    return u, R, alpha
    
@njit
def argmin_vertex_in_simplex(grad_alpha, purity):
    i_min = np.argmin(grad_alpha)
    
    s_alpha = np.zeros_like(grad_alpha)
    
    s_alpha[i_min] = purity
    
    return s_alpha
    
@njit
def frank_wolfe_nmf(W1, W2, meth_frequency, alpha1_init, alpha2_init, purity, max_iter, d_x):
    num_columns = alpha1_init.shape[1]  
    alpha1 = alpha1_init.copy()
    alpha2 = alpha2_init.copy()

    for k in range(max_iter):
        grad_alpha1 = - W1.T @  (d_x * (meth_frequency - W1 @ alpha1 - W2 @ alpha2))
        grad_alpha2 = - W2.T @ (d_x * (meth_frequency - W1 @ alpha1 - W2 @ alpha2))

        s_alpha1 = np.zeros_like(alpha1)
        s_alpha2 = np.zeros_like(alpha2)

        for col in range(num_columns):
            s_alpha1[:, col] = argmin_vertex_in_simplex(grad_alpha1[:, col], purity[col])
            s_alpha2[:, col] = argmin_vertex_in_simplex(grad_alpha2[:, col], 1 - purity[col])

        gamma_k = 2 / (k + 2) 

        alpha1 = (1 - gamma_k) * alpha1 + gamma_k * s_alpha1
        alpha2 = (1 - gamma_k) * alpha2 + gamma_k * s_alpha2


    return alpha1, alpha2

@njit
def mdwbssmf_deconv_p(u, R, alpha, meth_frequency, d_x, R_trunc, n_u, purity, n_iter1=100, n_iter2=500, tol=1e-3):
    nb_cpg, nb_celltypes = R_trunc.shape
    a1 = 1.0
    u_ = u.copy()
    alpha1, alpha2 = alpha[:-n_u], alpha[-n_u:]
    alpha1_ = alpha1.copy()
    alpha2_ = alpha2.copy()

    d = d_x.max() ** 2
    l_w = (np.linalg.norm(alpha2)**2) * d
    l_w_ = l_w
    
    cf = cost_f_w(meth_frequency, R, alpha, d_x)  
    
    for k in range(n_iter1):
        cf_0 = cf
        u, u_, a1, l_w_ = update_u(u, alpha, n_iter2, a1, l_w_, l_w, u_, meth_frequency, R_trunc, n_u, d_x)
        R = np.hstack((R_trunc, u.reshape(-1, n_u)))

        alpha1, alpha2 = frank_wolfe_nmf(R_trunc, u, meth_frequency, alpha1, alpha2, purity, n_iter2, d_x)
        
        l_w = (np.linalg.norm(alpha2)**2) * d

        alpha = np.vstack((alpha1, alpha2))

        cf = cost_f_w(meth_frequency, R, alpha, d_x)

        if abs(cf - cf_0) < tol:
            break


    return u, alpha

