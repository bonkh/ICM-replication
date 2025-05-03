import numpy as np

from sklearn import linear_model
from sklearn.metrics.pairwise import rbf_kernel
from scipy.linalg import solve
import psutil
import gc
import os
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