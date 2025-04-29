import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy.linalg import eigh
from numpy.linalg import solve
from sklearn.metrics.pairwise import rbf_kernel


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
    N = Kx.shape[0]
    Nt = Kt.shape[0] if Kt is not None else 0

    if Kx.shape != (N, N) or Ky.shape != (N, N):
        raise ValueError("Kx and Ky must be square matrices of the same size.")

    # Precompute unique groups and their sizes
    unique_groups, group_counts = np.unique(groupIdx, return_counts=True)
    G = len(unique_groups)

    # Centering matrix
    H = np.eye(N) - np.ones((N, N)) / N

    # Construct L matrix
    L = np.zeros((N, N), dtype=np.float64)
    for g, group_size in zip(unique_groups, group_counts):
        indices = np.where(groupIdx == g)[0]
        L[np.ix_(indices, indices)] = 1 / (G * group_size**2) - 1 / (G**2 * group_size**2)

    for g1, size1 in zip(unique_groups, group_counts):
        for g2, size2 in zip(unique_groups, group_counts):
            if g1 != g2:
                indices1 = np.where(groupIdx == g1)[0]
                indices2 = np.where(groupIdx == g2)[0]
                L[np.ix_(indices1, indices2)] = -1 / (G**2 * size1 * size2)

    # Center kernel matrices
    Kx = H @ Kx @ H
    Ky = H @ Ky @ H

    # Matrix A
    A_left = Kx @ L @ Kx + Kx + lambd * np.eye(N)
    mid = solve(Ky + N * epsilon * np.eye(N), Kx @ Kx)
    A_right = Ky @ mid
    A = solve(A_left, A_right)

    # Eigendecomposition (optimized for symmetric matrices)
    if np.allclose(A, A.T, atol=1e-8):  # Check if A is symmetric
        eigvals, eigvecs = eigh(A, subset_by_index=[N - M, N - 1])
    else:
        eigvals, eigvecs = eigs(A, k=M, which='LM')

    # Ensure real parts only
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)

    # Normalize eigenvectors
    eigvecs /= np.sqrt(eigvals)[np.newaxis, :]

    # Extract eigenvectors and eigenvalues
    V = eigvecs
    D = np.diag(eigvals)

    # Project training data
    X = V.T @ Kx

    # Project test data if available
    if Kt is not None and Nt > 0:
        Ht = np.eye(Nt) - np.ones((Nt, Nt)) / Nt
        Kt_c = Ht @ Kt @ H
        Xt = V.T @ Kt_c.T
    else:
        Xt = None

    return V, D, X, Xt


def dica_projection(train_x, train_y, domain_idx, lambda_, epsilon, m, gamma_x=None, gamma_y=None, supervised=True):
    """
    Perform DICA projection on training data.

    Parameters
    ----------
    train_x : ndarray
        Training features.
    train_y : ndarray
        Training output/labels.
    domain_idx : ndarray
        Domain membership indices.
    lambda_ : float
        Regularization parameter.
    epsilon : float
        Regularization parameter for output.
    m : int
        Number of components to extract.
    gamma_x : float, optional
        RBF kernel parameter for input.
    gamma_y : float, optional
        RBF kernel parameter for output.
    supervised : bool, optional
        Whether to use supervised labels.

    Returns
    -------
    B : ndarray
        Projection matrix.
    K : ndarray
        Kernel matrix for training data.
    """
    N = train_x.shape[0]

    # Set gamma values if not provided
    gamma_x = gamma_x or (1.0 / train_x.shape[1])
    gamma_y = gamma_y or 1.0

    # Compute input and output kernels
    K = rbf_kernel(train_x, train_x, gamma=gamma_x)
    if supervised:
        L = rbf_kernel(train_y.reshape(-1, 1), train_y.reshape(-1, 1), gamma=gamma_y)
    else:
        L = np.eye(N)

    # Domain-wise Q matrix
    unique_domains, domain_counts = np.unique(domain_idx, return_counts=True)
    Q = np.zeros((N, N))
    for d, count in zip(unique_domains, domain_counts):
        idx = np.where(domain_idx == d)[0]
        Q[np.ix_(idx, idx)] += 1.0 / count
    Q /= len(unique_domains)

    # Compute matrix C
    if supervised:
        C = L @ solve(L + N * epsilon * np.eye(N), K @ K)
    else:
        C = K @ K

    # Solve generalized eigenvalue problem
    A = C / N
    B_mat = K @ Q @ K + K + lambda_ * np.eye(N)
    eigvals, eigvecs = eigh(A, B_mat, subset_by_index=[N - m, N - 1])
    B = eigvecs  # Shape: (N, m)
    return B, K


def project_kernel(K, B):
    """Project full kernel through B."""
    return K @ B @ B.T @ K


def project_test_kernel(Kt, B, K_train):
    """Project test kernel using B and K."""
    return Kt @ B @ B.T @ K_train