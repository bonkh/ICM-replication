import cupy as cp
import cupyx.scipy.linalg as cpx_linalg


def rbf_kernel_gpu(X, Y=None, gamma=None):
    """GPU-compatible RBF kernel using CuPy."""
    if Y is None:
        Y = X
    X_norm = cp.sum(X ** 2, axis=-1)
    Y_norm = cp.sum(Y ** 2, axis=-1)
    K = -2 * cp.dot(X, Y.T) + X_norm[:, None] + Y_norm[None, :]
    if gamma is None:
        gamma = 1.0 / X.shape[1]
    return cp.exp(-gamma * K)


def dica_gpu(Kx, Ky, Kt, groupIdx, lambd, epsilon, M):
    """DICA optimized for GPU with CuPy."""
    N = Kx.shape[0]
    Nt = Kt.shape[0] if Kt is not None else 0

    groupIdx = cp.asarray(groupIdx)

    unique_groups, group_counts = cp.unique(groupIdx, return_counts=True)
    G = len(unique_groups)

    H = cp.eye(N) - cp.ones((N, N)) / N

    L = cp.zeros((N, N), dtype=cp.float64)
    for g, group_size in zip(unique_groups, group_counts):
        indices = cp.where(groupIdx == g)[0]
        L[cp.ix_(indices, indices)] = 1 / (G * group_size**2) - 1 / (G**2 * group_size**2)

    for g1, size1 in zip(unique_groups, group_counts):
        for g2, size2 in zip(unique_groups, group_counts):
            if g1 != g2:
                idx1 = cp.where(groupIdx == g1)[0]
                idx2 = cp.where(groupIdx == g2)[0]
                L[cp.ix_(idx1, idx2)] = -1 / (G**2 * size1 * size2)

    Kx = H @ Kx @ H
    Ky = H @ Ky @ H

    A_left = Kx @ L @ Kx + Kx + lambd * cp.eye(N)
    mid = cpx_linalg.solve(Ky + N * epsilon * cp.eye(N), Kx @ Kx)
    A_right = Ky @ mid
    A = cpx_linalg.solve(A_left, A_right)

    eigvals, eigvecs = cpx_linalg.eigh(A)
    idx = cp.argsort(eigvals)[-M:]
    eigvals = cp.real(eigvals[idx])
    eigvecs = cp.real(eigvecs[:, idx])
    eigvecs /= cp.sqrt(eigvals)[cp.newaxis, :]

    V = eigvecs
    D = cp.diag(eigvals)
    X = V.T @ Kx

    if Kt is not None and Nt > 0:
        Ht = cp.eye(Nt) - cp.ones((Nt, Nt)) / Nt
        Kt_c = Ht @ Kt @ H
        Xt = V.T @ Kt_c.T
    else:
        Xt = None

    return V, D, X, Xt


def dica_projection_gpu(train_x, train_y, domain_idx, lambda_, epsilon, m, gamma_x=None, gamma_y=None, supervised=True):
    """Train DICA on GPU inputs."""
    train_x = cp.asarray(train_x)
    train_y = cp.asarray(train_y)
    domain_idx = cp.asarray(domain_idx)

    N = train_x.shape[0]
    gamma_x = gamma_x or (1.0 / train_x.shape[1])
    gamma_y = gamma_y or 1.0

    K = rbf_kernel_gpu(train_x, gamma=gamma_x)
    if supervised:
        L = rbf_kernel_gpu(train_y.reshape(-1, 1), gamma=gamma_y)
    else:
        L = cp.eye(N)

    unique_domains, domain_counts = cp.unique(domain_idx, return_counts=True)
    Q = cp.zeros((N, N))
    for d, count in zip(unique_domains, domain_counts):
        idx = cp.where(domain_idx == d)[0]
        Q[cp.ix_(idx, idx)] += 1.0 / count
    Q /= len(unique_domains)

    if supervised:
        C = L @ cpx_linalg.solve(L + N * epsilon * cp.eye(N), K @ K)
    else:
        C = K @ K

    A = C / N
    B_mat = K @ Q @ K + K + lambda_ * cp.eye(N)

    eigvals, eigvecs = cpx_linalg.eigh(A, B_mat)
    idx = cp.argsort(eigvals)[-m:]
    eigvecs = eigvecs[:, idx]

    return eigvecs, K


def project_kernel_gpu(K, B):
    """Project full kernel matrix on GPU."""
    return K @ B @ B.T @ K


def project_test_kernel_gpu(Kt, B, K_train):
    """Project test data through kernel projection on GPU."""
    return Kt @ B @ B.T @ K_train
