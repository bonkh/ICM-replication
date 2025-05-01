import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
from numpy.linalg import solve
from scipy.io import loadmat
from sklearn.metrics.pairwise import rbf_kernel
from utils import *
import gc



def dica(Kx, Ky, Kt, groupIdx, lambd, epsilon, M):
    """
    DICA - Domain-Invariant Component Analysis

    Parameters
    ----------
    Kx : (N, N) ndarray
        Kernel matrix between training data points.
    Ky : (N, N) ndarray
        Kernel matrix between outputs.
    Kt : (Nt, N) ndarray
        Kernel matrix between test and training samples.
    groupIdx : (N,) ndarray
        Group membership vector (domain labels).
    lambd : float
        Regularization parameter for input.
    epsilon : float
        Regularization parameter for output.
    M : int
        Number of components to extract (dimensionality of subspace).

    Returns
    -------
    V : (N, M) ndarray
        Eigenvectors (each column is an eigenvector).
    D : (M, M) ndarray
        Diagonal matrix with eigenvalues.
    X : (M, N) ndarray
        Projection of training data.
    Xt : (M, Nt) ndarray
        Projection of test data.
    """
    start = get_memory_usage_gb()
    Kx = Kx.astype(np.float32)
    Ky = Ky.astype(np.float32)

    N = Kx.shape[0]
    Nt = Kt.shape[0] if Kt is not None else 0

    if Kx.shape != (N, N) or Ky.shape != (N, N):
        raise ValueError("Kx and Ky must be square matrices of the same size.")

    unique_groups = np.unique(groupIdx)
    G = len(unique_groups)
    NG = np.array([np.sum(groupIdx == g) for g in unique_groups])

    # Centering matrix
    H = np.eye(N) - np.ones((N, N)) / N

    # Construct L matrix
    L = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            gi = groupIdx[i]
            gj = groupIdx[j]
            if gi == gj:
                groupSize = NG[unique_groups == gi][0]
                L[i, j] = 1 / (G * groupSize**2) - 1 / (G**2 * groupSize**2)
            else:
                groupSize_i = NG[unique_groups == gi][0]
                groupSize_j = NG[unique_groups == gj][0]
                L[i, j] = -1 / (G**2 * groupSize_i * groupSize_j)

    del groupIdx, unique_groups, NG
    gc.collect()

    # Center kernel matrices
    Ky = H @ Ky @ H
    Kx = H @ Kx @ H
    
    A_left = Kx @ L @ Kx + Kx + lambd * np.eye(N)
    del L
    gc.collect()

    Ky_eps = Ky + N * epsilon * np.eye(N)
    mid = solve(Ky_eps, Kx @ Kx)
    del Ky_eps

    A_right = Ky @ mid
    del mid
    del Ky
    gc.collect()

    A = solve(A_left, A_right)
    del A_left, A_right
    gc.collect()

    eigvals, eigvecs = eigs(A, k=M)
    del A

    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    for i in range(M):
        eigvecs[:, i] /= np.sqrt(eigvals[i])
    gc.collect()

    V = eigvecs
    D = np.diag(eigvals)
    del eigvecs, eigvals
    gc.collect()


    X = V.T @ Kx
    del Kx
    gc.collect()

    if Kt is not None and Nt > 0:
        Ht = np.eye(Nt) - np.ones((Nt, Nt)) / Nt
        Kt_c = Ht @ Kt @ H
        Xt = V.T @ Kt_c.T
        del Ht, Kt_c
    else:
        Xt = None
    del Kt
    gc.collect()

    end = get_memory_usage_gb()
    print(f"RAM Used: {end - start:.2f} GB") 

    return V, D, X, Xt


def dica_projection(train_x, train_y, domain_idx, lambda_, epsilon, m, gamma_x=None, gamma_y=None, supervised=True):
    N = train_x.shape[0]
    
    # Set gamma values if not provided
    gamma_x = gamma_x or (1.0 / train_x.shape[1])
    gamma_y = gamma_y or 1.0

    # Compute input and output kernels
    K = compute_rbf_kernel_blockwise(train_x, train_x, gamma=gamma_x)
# 
    # K = rbf_kernel(train_x, train_x, gamma=gamma_x)
    if supervised:
        # L = rbf_kernel(train_y.reshape(-1, 1), train_y.reshape(-1, 1), gamma=gamma_y)
        L = compute_rbf_kernel_blockwise(train_y.reshape(-1, 1), train_y.reshape(-1, 1), gamma=gamma_y)
    else:
        L = np.eye(N)

    # Domain-wise Q matrix
    unique_domains = np.unique(domain_idx)
    Q = np.zeros((N, N))
    for d in unique_domains:
        idx = np.where(domain_idx == d)[0]
        nd = len(idx)
        Q[np.ix_(idx, idx)] += 1.0 / nd
    Q /= len(unique_domains)

    # Compute matrix C
    if supervised:
        C = L @ np.linalg.inv(L + N * epsilon * np.eye(N)) @ (K @ K)
    else:
        C = K @ K

    # Solve generalized eigenvalue problem: (1/n) * C * B = (K Q K + K + λI) * B * Γ
    A = C / N
    B_mat = K @ Q @ K + K + lambda_ * np.eye(N)

    # Solve B eigenvectors: A B = B_mat B Γ
    eigvals, eigvecs = eigh(A, B_mat, subset_by_index=[N - m, N - 1])
    B = eigvecs  # shape: (N, m)
    return B, K

def project_kernel(K, B):
    """Project full kernel through B."""
    return K @ B @ B.T @ K

def project_test_kernel(Kt, B, K_train):
    """Project test kernel using B and K."""
    return Kt @ B @ B.T @ K_train
