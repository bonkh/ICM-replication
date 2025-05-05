import numpy as np

from sklearn import linear_model
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import solve
import psutil
import gc
import inspect
import os
import torch

def train_linear_and_eval(x, y, x_test, y_test):
    model = linear_model.LinearRegression()
    model.fit(x, y)
    result = mse(model, x_test, y_test)
    return result


def get_memory_usage_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3

def get_color_dict():

  colors = {
    'pool' : 'red',
    'lasso' : 'red',
    'shat' : 'green',
    'sgreed' : 'green',
    'ssharp' : 'green',
    'strue' : 'blue',
    'cauid' : 'blue',
    'causharp': 'blue',
    'cauul' : 'blue',
    'mean' : 'black',
    'msda' : 'orange',
    'mtl' : 'orange',
    'dica' : 'orange',
    'dom' : 'k',
    'naive' : 'magenta'
  }

  markers = {
    'pool' : 'o',
    'lasso' : '^',
    'shat' : 'o',
    'sgreed' : '^',
    'strue' : '^',
    'ssharp' : 'd',
    'cauid' : 'd',
    'causharp' : 'h',
    'cauul' : '^',
    'mean' : 'o',
    'msda' : 'o',
    'mtl' : '^',
    'dica' : 'd',
    'dom' : 'o',
    'naive' : 'o'
  }

  legends = {
              'pool' : r'$\beta^{CS}$',
              'lasso' : r'$\beta^{CS(\hat S Lasso)}$',
              'shat' : r'$\beta^{CS(\hat S)}$',
              'ssharp' : r'$\beta^{CS(\hat S \sharp)}$',
              'strue' : r'$\beta^{CS(cau)}$',
              'cauid' : r'$\beta^{CS(cau+,id)}$',
              'causharp' : r'$\beta^{CS(cau\sharp)}$',
              'cauul' : r'$\beta^{CS(cau\sharp UL)}$',
              'sgreed' :r'$\beta^{CS(\hat{S}_{greedy})}$',
              'mean'   : r'$\beta^{mean}$',
              'msda'   : r'$\beta^{mSDA}$',
              'mtl'   : r'$\beta^{MTL}$',
              'dica'   : r'$\beta^{DICA}$',
              'naive'   : r'$\beta^{naive}$',
              'dom'   : r'$\beta^{dom}$'
            }

  return colors, markers, legends

def mse(model, x, y):
  return np.mean((model.predict(x)-y)**2)


def compute_rbf_kernel_blockwise(X, Y=None, gamma=1.0, block_size=500, dtype=np.float32):
    """
    Compute RBF kernel in memory-efficient blockwise fashion.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
        First input dataset.
    Y : ndarray of shape (n_samples_Y, n_features), optional
        Second input dataset. If None, Y = X.
    gamma : float
        RBF kernel coefficient.
    block_size : int
        Number of rows to compute per block.
    dtype : np.dtype
        Output dtype to control memory usage.

    Returns
    -------
    K : ndarray of shape (n_samples_X, n_samples_Y)
        Computed RBF kernel matrix.
    """
    if Y is None:
        Y = X

    n_X = X.shape[0]
    n_Y = Y.shape[0]
    K = np.empty((n_X, n_Y), dtype=dtype)

    for i in range(0, n_X, block_size):
        i_end = min(i + block_size, n_X)
        Xi = X[i:i_end]
        K[i:i_end] = rbf_kernel(Xi, Y, gamma=gamma).astype(dtype)

    return K


def compute_rbf_kernel_blockwise_to_memmap(X, Y=None, gamma=1.0, block_size=500, 
                                           dtype=np.float32, mmap_path='kernel.dat'):
    """
    Compute RBF kernel in blockwise fashion and store in memmap to save RAM.

    Parameters
    ----------
    X : ndarray of shape (n_samples_X, n_features)
    Y : ndarray of shape (n_samples_Y, n_features), optional
        If None, Y = X.
    gamma : float
        RBF kernel coefficient.
    block_size : int
        Block size for memory-efficient computation.
    dtype : np.dtype
        Data type of the output kernel matrix.
    mmap_path : str
        Path to the memmap file to store the kernel.

    Returns
    -------
    K : np.memmap
        Memory-mapped RBF kernel matrix.
    """
    if Y is None:
        Y = X

    # n_X, n_Y = X.shape[0], Y.shape[0]

    # # Create or overwrite memory-mapped file
    # K = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=(n_X, n_Y))

    # for i in range(0, n_X, block_size):
    #     i_end = min(i + block_size, n_X)
    #     Xi = X[i:i_end]

    #     # Compute and store block row
    #     K_block = rbf_kernel(Xi, Y, gamma=gamma).astype(dtype)
    #     K[i:i_end, :] = K_block
    #     K.flush()

    # return K

    if Y is None:
        Y = X

    n_X = X.shape[0]
    n_Y = Y.shape[0]

    # Initialize memmap file
    K = np.memmap(mmap_path, dtype=dtype, mode='w+', shape=(n_X, n_Y))

    for i in range(0, n_X, block_size):
        i_end = min(i + block_size, n_X)
        Xi = X[i:i_end]
        
        # Compute RBF kernel for block
        K_block = rbf_kernel(Xi, Y, gamma=gamma).astype(dtype)

        # Store to memmap file
        K[i:i_end] = K_block
        K.flush()  # Ensure data is written to disk

    return K


def center_kernel_memmap(K_path, shape):
    N = shape[0]
    K = np.memmap(K_path, dtype='float32', mode='r', shape=shape)
    
    # Compute row and column means and total mean
    row_mean = np.zeros(N, dtype='float32')
    for i in range(N):
        row_mean[i] = np.mean(K[i, :])

    col_mean = np.zeros(N, dtype='float32')
    for j in range(N):
        col_mean[j] = np.mean(K[:, j])

    total_mean = np.mean(row_mean)

    # Allocate new memmap for centered matrix
    centered_path = K_path.replace('.dat', '_centered.dat')
    K_centered = np.memmap(centered_path, dtype='float32', mode='w+', shape=(N, N))

    for i in range(N):
        for j in range(N):
            K_centered[i, j] = K[i, j] - row_mean[i] - col_mean[j] + total_mean

    del K, K_centered
    return centered_path

import numpy as np
import os


def build_L_memmap(groupIdx, N, L_path='L.dat', dtype=np.float32):
    """
    Constructs the matrix L in a memory-mapped file.

    Parameters
    ----------
    groupIdx : ndarray of shape (N,)
        Domain/group labels for each sample.
    N : int
        Total number of samples.
    L_path : str
        Path to save the memory-mapped L matrix.
    dtype : data-type
        Data type for the memmap array.

    Returns
    -------
    str
        Path to the memory-mapped L file.
    """
    unique_groups = np.unique(groupIdx)
    G = len(unique_groups)
    NG = np.array([np.sum(groupIdx == g) for g in unique_groups])

    L = np.memmap(L_path, dtype=dtype, mode='w+', shape=(N, N))

    for i in range(N):
        gi = groupIdx[i]
        groupSize_i = NG[unique_groups == gi][0]

        for j in range(N):
            gj = groupIdx[j]
            groupSize_j = NG[unique_groups == gj][0]

            if gi == gj:
                L[i, j] = 1 / (G * groupSize_i**2) - 1 / (G**2 * groupSize_i**2)
            else:
                L[i, j] = -1 / (G**2 * groupSize_i * groupSize_j)

        if i % 100 == 0:
            # print(f"Row {i}/{N} written")
            gc.collect()

    L.flush()
    del L
    return L_path


def compute_A_left_blockwise(Kx_path, L_path, out_path, N, lambd, block_size=512, dtype=np.float32):
    # Load memory-mapped inputs
    Kx = np.memmap(Kx_path, dtype=dtype, mode='r', shape=(N, N))
    L = np.memmap(L_path, dtype=dtype, mode='r', shape=(N, N))
    A_left = np.memmap(out_path, dtype=dtype, mode='w+', shape=(N, N))

    # Step 1: Compute T1 = Kx @ L
    base, ext = os.path.splitext(out_path)
    t1_path = base + "_t1" + ext
    T1 = np.memmap(t1_path, dtype=dtype, mode='w+', shape=(N, N))

    # print("Step 1: Computing T1 = Kx @ L ...")

    # T1_a = Kx @ L

    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        T1[i:i_end, :] = Kx[i:i_end, :] @ L
        gc.collect()
    T1.flush()

    # Step 2: Compute T2 = T1 @ Kx
    # print("Step 2: Computing T2 = T1 @ Kx ...")
    # T2_a = T1_a @ Kx

    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        A_left[i:i_end, :] = T1[i:i_end, :] @ Kx
        gc.collect()
    


    # Step 3: Add Kx to A_left
    # print("Step 3: Adding Kx to A_left ...")
    # T2_a += Kx

    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        A_left[i:i_end, :] += Kx[i:i_end, :]
        gc.collect()

   

    # Step 4: Add lambd * I
    # print("Step 4: Adding λ * I to A_left ...")
    # T2_a = T2_a + lambd*np.eye(N)
    for i in range(0, N):
        A_left[i, i] += lambd

    # A = Kx @ L @ Kx + Kx + lambd * np.eye(N)

    # are_equal = np.allclose(A, A_left, rtol=1e-4, atol=1e-6)
    # print("Are equal ?:", are_equal)


    A_left.flush()
    del A_left
    gc.collect()

    # A_left_1 = np.memmap(out_path, dtype='float32', mode='r', shape=(N, N))
    # are_equal_1 = np.allclose(A_left_1, A, rtol=1e-4, atol=1e-6)
    # print("Are equal huh ?:", are_equal_1)

    # backup_path = out_path.replace(".dat", "_backup.dat")

    # # Write A_left to this new file
    # A_left_backup = np.memmap(backup_path, dtype='float32', mode='w+', shape=(N, N))
    # A_left_backup[:] = A_left_1[:]
    # A_left_backup.flush()


    # Cleanup
    del Kx, L, T1
    gc.collect()
    # print(out_path)


    return out_path

def compute_A_right_blockwise(Ky_path, mid_path, out_path, N, block_size=512, dtype=np.float32):
    Ky = np.memmap(Ky_path, dtype=dtype, mode='r', shape=(N, N))
    mid = np.memmap(mid_path, dtype=dtype, mode='r', shape=(N, N))
    A_right = np.memmap(out_path, dtype=dtype, mode='w+', shape=(N, N))  # output

    # print("Computing A_right = Ky @ mid blockwise...")
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        
        # Use float64 internally to reduce numerical error
        Ky_block = Ky[i:i_end, :].astype(np.float64)
        mid_64 = mid.astype(np.float64)
        A_right[i:i_end, :] = (Ky_block @ mid_64).astype(dtype)
        
        gc.collect()

    A_right.flush()
    del Ky, mid, A_right
    gc.collect()
    return out_path

def solve_blockwise(A_left_path, A_right_path, out_path, N, block_size=512, dtype=np.float32):
    A_left = np.memmap(A_left_path, dtype=dtype, mode='r', shape=(N, N))
    A_right = np.memmap(A_right_path, dtype=dtype, mode='r', shape=(N, N))
    A_out = np.memmap(out_path, dtype=dtype, mode='w+', shape=(N, N))  # Output A

    # print("Solving A_left @ A = A_right blockwise...")
    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        rhs_block = A_right[:, i:i_end]
        A_block = solve(A_left, rhs_block)  # or 'pos' if A_left is PSD
        A_out[:, i:i_end] = A_block.astype(dtype)
        gc.collect()

    A_out.flush()
    del A_left, A_right, A_out
    gc.collect()
    return out_path

def blockwise_mm(A, B, block_size=512, device=None):
    """Blockwise matrix multiplication: A (n x m) @ B (m x p)"""
    n, m = A.shape
    _, p = B.shape
    device = A.device
    dtype = A.dtype
    result = torch.zeros((n, p), device=device, dtype=dtype)

    for i in range(0, n, block_size):
        i_end = min(i + block_size, n)
        for j in range(0, p, block_size):
            j_end = min(j + block_size, p)
            for k in range(0, m, block_size):
                k_end = min(k + block_size, m)
                result[i:i_end, j:j_end] += A[i:i_end, k:k_end] @ B[k:k_end, j:j_end]

    return result
def center_kernel_blockwise(K, block_size=512, device=None):
    """
    Centers a kernel matrix blockwise to reduce memory usage.
    Args:
        K (torch.Tensor): NxN kernel matrix.
        block_size (int): Size of blocks.
    Returns:
        K_centered (torch.Tensor): Centered kernel matrix.
    """
    N = K.shape[0]
    K_mean_row = K.mean(dim=1, keepdim=True)
    K_mean_col = K.mean(dim=0, keepdim=True)
    K_mean_total = K.mean()

    K_centered = torch.zeros_like(K)

    for i in range(0, N, block_size):
        i_end = min(i + block_size, N)
        for j in range(0, N, block_size):
            j_end = min(j + block_size, N)

            block = K[i:i_end, j:j_end]
            block -= K_mean_row[i:i_end]
            block -= K_mean_col[:, j:j_end]
            block += K_mean_total
            K_centered[i:i_end, j:j_end] = block

    return K_centered


# def try_gpu_then_cpu(fn, *args, **kwargs):
#     try:
#         return fn(*args, **kwargs)
#     except RuntimeError as e:
#         if torch.cuda.is_available() and "CUDA out of memory" in str(e):
#             print(f"[OOM] Falling back to CPU for: {fn.__name__}")
#             torch.cuda.empty_cache()
#             gc.collect()

#             # Move tensor arguments to CPU
#             args_cpu = tuple(a.cpu() if isinstance(a, torch.Tensor) and a.is_cuda else a for a in args)
#             kwargs_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) and v.is_cuda else v for k, v in kwargs.items()}

#             return fn(*args_cpu, **kwargs_cpu)
#         else:
#             raise
# def try_gpu_then_cpu(fn, *args, **kwargs):
#     def to_cpu(x):
#         return x.cpu() if isinstance(x, torch.Tensor) and x.is_cuda else x

#     try:
#         return fn(*args, **kwargs)
#     except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
#         if "CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
#             print(f"[OOM] Falling back to CPU for: {fn.__name__}")
#             torch.cuda.empty_cache()
#             gc.collect()
#             args_cpu = tuple(to_cpu(a) for a in args)
#             kwargs_cpu = {k: to_cpu(v) for k, v in kwargs.items()}
#             return fn(*args_cpu, **kwargs_cpu)
#         raise
# def try_gpu_then_cpu(expr_fn):
#     try:
#         return expr_fn()
#     except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
#         if "CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
#             print(f"[OOM] Falling back to CPU for: {expr_fn}")
#             torch.cuda.empty_cache()
#             gc.collect()

#             def to_cpu(x):
#                 return x.cpu() if isinstance(x, torch.Tensor) and x.is_cuda else x

#             # Manually walk through closure variables
#             if hasattr(expr_fn, '__closure__') and expr_fn.__closure__:
#                 for cell in expr_fn.__closure__:
#                     val = cell.cell_contents
#                     if isinstance(val, torch.Tensor) and val.is_cuda:
#                         val_cpu = val.cpu()
#                         try:
#                             cell.cell_contents = val_cpu  # may fail, Python limitation
#                         except:
#                             pass

#             return expr_fn()
#         else:
#             raise
# def try_gpu_then_cpu(expr_fn, *args, **kwargs):
#     def to_cpu(x):
#         return x.cpu() if isinstance(x, torch.Tensor) and x.is_cuda else x

#     try:
#         return expr_fn(*args, **kwargs)
#     except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
#         if "CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
#             print(f"[OOM] Falling back to CPU for: {expr_fn.__name__ if hasattr(expr_fn, '__name__') else expr_fn}")
#             torch.cuda.empty_cache()
#             gc.collect()

#             args_cpu = tuple(to_cpu(a) for a in args)
#             kwargs_cpu = {k: to_cpu(v) for k, v in kwargs.items()}
#             return expr_fn(*args_cpu, **kwargs_cpu)
#         else:
#             raise


def try_gpu_then_cpu(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "CUDA out of memory" in str(e) or isinstance(e, torch.cuda.OutOfMemoryError):
            print(f"[OOM] Falling back to CPU for: {fn.__name__ if hasattr(fn, '__name__') else str(fn)}")
            torch.cuda.empty_cache()
            gc.collect()

            def to_cpu(x):
                if isinstance(x, torch.Tensor) and x.is_cuda:
                    return x.cpu()
                elif isinstance(x, (list, tuple)):
                    return type(x)(to_cpu(i) for i in x)
                elif isinstance(x, dict):
                    return {k: to_cpu(v) for k, v in x.items()}
                else:
                    return x

            args_cpu = to_cpu(args)
            kwargs_cpu = to_cpu(kwargs)

            return fn(*args_cpu, **kwargs_cpu)
        else:
            raise



def safe_solve(A_left, A_right, device=None):
    return torch.linalg.solve(A_left.to(device), A_right.to(device))


def build_L_matrix(groupIdx, N, dtype=torch.float32, device='cuda'):
    groupIdx = groupIdx.to(device)
    unique_groups = torch.unique(groupIdx)
    G = len(unique_groups)
    group_counts = torch.stack([(groupIdx == g).sum() for g in unique_groups])

    L = torch.zeros((N, N), dtype=dtype, device=device)

    for g, count in zip(unique_groups, group_counts):
        idx = (groupIdx == g).nonzero(as_tuple=True)[0]
        val = 1 / (G * count.item()**2) - 1 / (G**2 * count.item()**2)
        L[idx.unsqueeze(1), idx] = val

    for i, (g1, s1) in enumerate(zip(unique_groups, group_counts)):
        for j, (g2, s2) in enumerate(zip(unique_groups, group_counts)):
            if g1 != g2:
                idx1 = (groupIdx == g1).nonzero(as_tuple=True)[0]
                idx2 = (groupIdx == g2).nonzero(as_tuple=True)[0]
                val = -1 / (G**2 * s1.item() * s2.item())
                L[idx1.unsqueeze(1), idx2] = val

    return L