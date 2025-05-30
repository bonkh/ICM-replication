import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
from numpy.linalg import solve
from scipy.io import loadmat
from sklearn.metrics.pairwise import rbf_kernel
from utils import *
import torch
import gc



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
def dica(Kx_path, Ky_path, Kt_path,N, Nt, groupIdx_path, lambd, epsilon, M):
    
    Kx = np.memmap(Kx_path, dtype='float32', mode='r', shape=(N, N))
    Ky = np.memmap(Ky_path, dtype='float32', mode='r', shape=(N, N))
    Kt = np.memmap(Kt_path, dtype='float32', mode='r', shape=(Nt, N))
    groupIdx = np.memmap(groupIdx_path, dtype='int32', mode='r', shape=(N,))

    if Kx.shape != (N, N) or Ky.shape != (N, N):
        raise ValueError("Kx and Ky must be square matrices of the same size.")

    unique_groups = np.unique(groupIdx)
    NG = np.array([np.sum(groupIdx == g) for g in unique_groups])

    # Centering matrix
    H = np.eye(N) - np.ones((N, N)) / N
  
    L_path = build_L_memmap(groupIdx=groupIdx, N=N, L_path='Experiment_01_top_left/L.dat')
    del groupIdx, unique_groups, NG
    gc.collect()

    # Center kernel matrices
    # Ky = H @ Ky @ H
    Ky_centered_path = center_kernel_memmap(Ky_path, shape=(N, N))

    # Kx = H @ Kx @ H
    Kx_centered_path = center_kernel_memmap(Kx_path, shape=(N, N))

    # Calculate A left
    A_left_path = "Experiment_01_top_left/A_left.dat"
    A_left_path = compute_A_left_blockwise(
        Kx_path = Kx_centered_path,
        L_path = L_path,  
        out_path = A_left_path,
        N = N,
        lambd = 0.01,
        block_size = 512 
        )
    
    gc.collect()

    # Calculate mid
    Ky_eps = Ky.copy()
    Ky_eps[np.diag_indices_from(Ky_eps)] += N * epsilon

    mid_path = "Experiment_01_top_left/mid.dat"
    mid = np.memmap(mid_path, dtype=Kx.dtype, mode='w+', shape=(N, N))
    block_size = 512

    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)

        # rhs_block = Kx @ Kx[:, i:i_end]
        rhs_block = np.dot(Kx, Kx[:, i:i_end])


        # Solve the linear system
        mid[:, i:i_end] = solve(Ky_eps, rhs_block)

        mid.flush()
        gc.collect()

    del Ky_eps, rhs_block
    gc.collect()

    mid = np.memmap(mid_path, dtype=np.float32, mode='r', shape=(N, N))

    # A_right_1 = Ky @ mid
    A_right_path = "Experiment_01_top_left/A_right.dat"
    A_right_path = compute_A_right_blockwise(
        Ky_path = Ky_centered_path,
        mid_path = mid_path,
        out_path = A_right_path,
        N=N
    )

    del mid
    del Ky
    gc.collect()

    # A = solve(A_left, A_right)
    A_path = "Experiment_01_top_left/A.dat"

    A_path = solve_blockwise(
        A_left_path = A_left_path,
        A_right_path = A_right_path,
        out_path = A_path,
        N=N,
        block_size=512  # adjust if needed
    )
    #

    # del A_left, A_right
    gc.collect()

    A = np.memmap(A_path, dtype=np.float32, mode='r', shape=(N, N))
    eigvals, eigvecs = eigs(A, k=M)
    del A

    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)


    positive_idx = eigvals > 1e-8  # tránh các trị gần 0
    eigvals = eigvals[positive_idx]
    eigvecs = eigvecs[:, positive_idx]


    for i in range(M):
        eigvecs[:, i] /= np.sqrt(eigvals[i])
    gc.collect()

    V = eigvecs
    D = np.diag(eigvals)
    del eigvecs, eigvals
    gc.collect()

    block_size = 500

    X = np.empty((M, N), dtype=np.float32)
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        X[:, i:i_end] = V.T @ Kx[:, i:i_end]

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


def dica_torch(Kx_path, Ky_path, Kt_path, N, Nt, groupIdx_path, lambd, epsilon, M, dtype='float32'):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Kx = torch.from_numpy(np.memmap(Kx_path, dtype=dtype, mode='r', shape=(N, N)).copy()).to(device)
    Ky = torch.from_numpy(np.memmap(Ky_path, dtype=dtype, mode='r', shape=(N, N)).copy()).to(device)
    Kt = torch.from_numpy(np.memmap(Kt_path, dtype=dtype, mode='r', shape=(Nt, N)).copy()).to(device)
    groupIdx = torch.from_numpy(np.memmap(groupIdx_path, dtype='int32', mode='r', shape=(N,)).copy()).to(device)
   
    torch.cuda.empty_cache()
    gc.collect()

    Nt = Kt.shape[0] if Kt is not None else 0

    I = safe_eye(N)
    # ones = safe_ones((N, N)) / N
    ones = safe_ones((N, N))
    ones_divided = try_gpu_safe(divide_fn, ones, N)
    del ones
    # ones_divided = ones_divided.to(I.device)
    # H = I - ones_divided
    H = try_gpu_safe(subtract_fn, I, ones_divided)


    del ones_divided, I
    torch.cuda.empty_cache()
    gc.collect()

    # Build L matrix
    # unique_groups = torch.unique(groupIdx)
    unique_groups = safe_unique(groupIdx)
    L = try_gpu_safe (build_L_matrix, groupIdx, N, dtype=Kx.dtype, device=device)

    del groupIdx, unique_groups
    torch.cuda.empty_cache()
    gc.collect()

    Kx = try_gpu_safe(center_kernel_blockwise, Kx, block_size=512, device=device)
    Ky = try_gpu_safe(center_kernel_blockwise, Ky, block_size=512, device=device)

    Kx_L = try_gpu_safe(blockwise_mm, Kx, L, block_size=512, device=device)
    Kx_L_Kx = try_gpu_safe(blockwise_mm, Kx_L, Kx, block_size=512, device=device)

    del Kx_L, L
    torch.cuda.empty_cache()
    gc.collect()

    Kx_Kx = try_gpu_safe(blockwise_mm, Kx, Kx, block_size=512)

    eye_N = safe_eye(N, dtype=Kx.dtype)
    eye_scaled = try_gpu_safe(scale_fn, eye_N, lambd)

    A_left = try_gpu_safe(add_fn, Kx_L_Kx, Kx)

    A_left = try_gpu_safe(add_fn, A_left, eye_scaled)
    del eye_scaled
    gc.collect()

    eye_scaled = try_gpu_safe(scale_fn, eye_N, N * epsilon)
    Ky_eps = try_gpu_safe(add_fn, Ky, eye_scaled)
    del eye_scaled
    gc.collect()

    mid = try_gpu_safe(solve_fn, Ky_eps, Kx_Kx)
    A_right = try_gpu_safe(matmul_fn, Ky, mid)

    A = try_gpu_safe(solve_fn, A_left, A_right)


    del A_left, A_right, mid, Ky_eps, Ky
    torch.cuda.empty_cache()
    gc.collect()


    if torch.allclose(A, A.T, atol=1e-6):
        eigvals, eigvecs = try_gpu_safe(safe_eigh, A)
        eigvals = eigvals[-M:]
        eigvecs = eigvecs[:, -M:]
    else:
        eigvals, eigvecs = try_gpu_safe(safe_eig, A)
        eigvals = eigvals.real
        eigvecs = eigvecs.real
        
        topk_vals, indices = try_gpu_safe(safe_topk, eigvals, M)
        eigvals = topk_vals
        indices = indices.to(eigvecs.device)
        eigvecs = eigvecs[:, indices]

    sqrt_vals = try_gpu_safe(safe_sqrt, eigvals)
    sqrt_vals = sqrt_vals.to(eigvecs.device)
    # eigvecs = eigvecs / sqrt_vals.unsqueeze(0)
    # eigvecs = safe_divide(eigvecs, sqrt_vals, unsqueeze_dim=0)
    eigvecs = try_gpu_safe(divide_fn, eigvecs, sqrt_vals, unsqueeze_dim = 0)

    V = eigvecs
    D = try_gpu_safe(safe_diag, eigvals)

    del A, eigvecs, eigvals
    torch.cuda.empty_cache()
    gc.collect()

    X = try_gpu_safe(matmul_fn, V.T, Kx)
    del Kx
    torch.cuda.empty_cache()
    gc.collect()

    if Kt is not None and Nt > 0:
        eye_Nt = safe_eye(Nt, dtype=Kt.dtype)
        ones_Nt = try_gpu_safe(safe_ones, (Nt, Nt), dtype=Kt.dtype)
        scaled_ones = try_gpu_safe(scale_fn, ones_Nt, 1.0 / Nt)
        del ones_Nt
        Ht = try_gpu_safe(add_fn, eye_Nt, -scaled_ones)  # Ht = I - 1/N * 1_1^T
        del eye_Nt, scaled_ones
       
        Ht_Kt = try_gpu_safe(matmul_fn, Ht, Kt)
        del Ht, Kt
        Kt_c = try_gpu_safe(matmul_fn, Ht_Kt, H)
        del Ht_Kt, H 
        # Xt = V.T @ Kt_c.T
        Xt = try_gpu_safe(matmul_fn, V.T, Kt_c.T)
        del Kt_c

    else:
        Xt = None
    torch.cuda.empty_cache()
    gc.collect()

    return V, D, X, Xt