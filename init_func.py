import numpy as np
from scipy.linalg import svd
from scipy.optimize import nnls
from numpy.linalg import eig
from scipy.optimize import minimize_scalar, nnls

def constrained_nndsvd(Y, W1, rank, flag=0):
    n_features, n_samples = Y.shape
    k1 = W1.shape[1] 

    H1 = np.zeros((k1, n_samples))
    for i in range(n_samples):
        H1[:, i], _ = nnls(W1, Y[:, i])

    Y_residual = np.maximum(Y - W1 @ H1, 0)

    W2, H2 = nndsvd_initialize(Y_residual, rank=rank, flag=flag)

    W2 = np.clip(W2, 0, 1)

    W = np.hstack([W1, W2])

    H = np.vstack([H1, H2])

    return W, H


def nndsvd_initialize(V, rank, flag=0):
    if np.any(V < 0):
        raise ValueError("The input matrix contains negative elements.")

    U, S, E = svd(V, full_matrices=False)
    E = E.T

    W = np.zeros((V.shape[0], rank))
    H = np.zeros((rank, V.shape[1]))

    W[:, 0] = np.sqrt(S[0]) * np.abs(U[:, 0])
    H[0, :] = np.sqrt(S[0]) * np.abs(E[:, 0].T)

    for i in range(1, rank):
        uu = U[:, i]
        vv = E[:, i]
        uup, uun = _pos_neg(uu)
        vvp, vvn = _pos_neg(vv)
        n_uup, n_vvp = np.linalg.norm(uup, 2), np.linalg.norm(vvp, 2)
        n_uun, n_vvn = np.linalg.norm(uun, 2), np.linalg.norm(vvn, 2)
        termp = n_uup * n_vvp
        termn = n_uun * n_vvn

        if termp >= termn:
            W[:, i] = np.sqrt(S[i] * termp) / n_uup * uup
            H[i, :] = np.sqrt(S[i] * termp) / n_vvp * vvp.T
        else:
            W[:, i] = np.sqrt(S[i] * termn) / n_uun * uun
            H[i, :] = np.sqrt(S[i] * termn) / n_vvn * vvn.T

    W[W < 1e-11] = 0
    H[H < 1e-11] = 0

    if flag == 1:  
        avg = np.mean(V)
        W[W == 0] = avg
        H[H == 0] = avg
    elif flag == 2:  
        avg = np.mean(V)
        W[W == 0] = avg * np.random.uniform(0, 1, size=W[W == 0].shape) / 100
        H[H == 0] = avg * np.random.uniform(0, 1, size=H[H == 0].shape) / 100

    return W, H


def _pos_neg(X):
    X_pos = np.maximum(X, 0)
    X_neg = np.maximum(-X, 0)
    return X_pos, X_neg


def loss(Y):
    n_samples = Y.shape[1]
    Y_neg = np.where(Y < 0, Y, 0)

    return 1 / (2 * n_samples) * np.linalg.norm(Y_neg, ord='fro')**2


def constrained_nn_ica(Y, W1, rank, t_tol=1e-1, t_neg=None, verbose=1, i_max=1e3):
    n_features, n_samples = Y.shape
    k1 = W1.shape[1]

    H1 = np.zeros((k1, n_samples))
    for i in range(n_samples):
        H1[:, i] = nnls(W1, Y[:, i])[0]

    Y_residual = np.maximum(Y - W1 @ H1, 0)

    W2, H2 = run_nn_ica(Y_residual, rank=rank, t_tol=t_tol, t_neg=t_neg, verbose=verbose, i_max=i_max)


    W = np.hstack([W1, W2])
    H = np.vstack([H1, H2])

    return W, H


def run_nn_ica(X, rank, t_tol=1e-1, t_neg=None, verbose=1, i_max=1e3):
    def whiten(X, epsilon=1e-8):
        C_X = np.cov(X, rowvar=True)
        D, E = eig(C_X)
        D = np.real(D)  
        D = np.maximum(D, epsilon)  
        E = np.real(E) 
        V = E @ np.diag(1 / np.sqrt(D)) @ E.T
        return V @ X

    def rotation(phi):
        return np.array([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])

    def torque(Y):
        Y = np.real(Y)  
        Y_pos = np.maximum(Y, 0)
        Y_neg = np.maximum(-Y, 0)
        n = Y.shape[0]
        G = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                G[i, j] = np.dot(Y_pos[i, :], Y_neg[j, :]) - np.dot(Y_neg[i, :], Y_pos[j, :])
        t_max = np.amax(np.abs(G))
        if t_max == 0:
            return 0, [], G
        result = np.where(np.abs(G) == t_max)
        ixs = [result[i][0] for i in range(len(result)) if len(result[i]) > 0]
        return t_max, ixs, G

    n_features, n_samples = X.shape
    Z = whiten(X)
    W = np.eye(n_features)
    Y = W @ Z

    i = 0
    while i < i_max:
        t_max, ixs, _ = torque(Y)
        if t_max < t_tol:
            break
        Y_red = Y[ixs, :]
        opt_res = minimize_scalar(lambda phi: loss(rotation(phi) @ Y_red), bounds=(0, 2 * np.pi), method='bounded')
        R = givens(n_features, ixs[0], ixs[1], opt_res['x'])
        W = R @ W
        Y = R @ Y
        i += 1

    if i >= i_max and t_max >= t_tol:
        print("Warning: Algorithm did not converge.")

    H = np.maximum(W @ Z, 0)
    return np.clip(W[:, :rank], 0, 1), H[:rank, :]



def givens(n, i, j, phi):
    R = np.eye(n)
    R[i, i], R[j, j] = np.cos(phi), np.cos(phi)
    R[i, j], R[j, i] = np.sin(phi), -np.sin(phi)
    return R