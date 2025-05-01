import numpy as np
from subset_search import *
from sklearn import linear_model
from sklearn.linear_model import LinearRegression


from sklearn.metrics.pairwise import rbf_kernel
from dica import *


from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

np.random.seed(2025)

n_samples_per_task = 4000 # (n_examples_task)
n_tasks = 3
n_test_tasks = 10 #n_test_tasks = 100
n_predictors = 3
n_ex = []

# Parameter of the SEM

alpha = np.random.uniform(-1, 2.5, 2)
sigma = 1.5
sx1 = 1
sx2 = 0.1
sx3 = 1

train_x = np.zeros((1, n_predictors))
print(train_x)
train_y = np.zeros(1)

use_hsic = True
return_mse = False
delta = 0.05

# Generate training tasks

for task in range(n_tasks):
    gamma_task = np.random.uniform(-1,1)
    x1 = np.random.normal(0, sx1, (n_samples_per_task, 1))
    x3 = np.random.normal(0, sx3, (n_samples_per_task, 1))
    y = alpha[0] * x1 + alpha[1] * x3 + np.random.normal(0, sigma, (n_samples_per_task, 1))

    x2 = gamma_task*y + np.random.normal(0, sx2, (n_samples_per_task, 1))

    x_task = np.concatenate([x1, x2, x3],axis = 1)
    train_x = np.append(train_x, x_task, axis = 0)
    train_y = np.append(train_y, y)
    n_ex.append(n_samples_per_task)

#     print(n_ex)

# print(train_x.shape)

n_ex = np.array(n_ex)
train_x =  train_x[1:, :]
train_y = train_y[1:, np.newaxis]

# Generate test tasks
test_x = np.zeros((1, n_predictors))
test_y = np.zeros(1)

for task in range (n_test_tasks):
    gamma_task = np.random.uniform(-1, 1)
    x1 = np.random.normal(0, sx1, (n_samples_per_task, 1))
    x3 = np.random.normal(0, sx3, (n_samples_per_task, 1))
    y = alpha[0]*x1 + alpha[1]*x3 + np.random.normal(0, sigma, (n_samples_per_task,1))

    x2 = gamma_task*y + np.random.normal(0,sx2,(n_samples_per_task,1))

    x_task = np.concatenate([x1, x2, x3],axis = 1)
    test_x = np.append(test_x, x_task, axis = 0)
    test_y = np.append(test_y, y)


test_x = test_x[1:,:]
test_y = test_y[1:,np.newaxis]

# import scipy.io

# data = scipy.io.loadmat('dataset.mat')

# train_x = data['train_x']
# train_y = data['train_y']
# test_x = data['test_x']
# test_y = data['test_y']
# n_ex = data['n_ex'].flatten()  # Flatten về vector 1D nếu cần

# print(train_x.shape)
# print(train_y.shape)
# print(test_x.shape)
# print(test_y.shape)
# print(n_ex)


s_hat = subset(train_x, train_y, n_ex, valid_split=0.5, 
                             delta=0.05, use_hsic=use_hsic)

print(f'S_HAT{s_hat}')

domain_idx = []
for domain, n in enumerate(n_ex):
    domain_idx.extend([domain] * n)
domain_idx = np.array(domain_idx)

# print (f'Domain index: {domain_idx}')

print(test_x.shape)


# Use RBF kernel with chosen bandwidth
gamma_x = 1.0 / train_x.shape[1]  # Or tune
gamma_y = 1.0


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

Kx1 = compute_rbf_kernel_blockwise(train_x, gamma=gamma_x)
# Ky = rbf_kernel(train_y, train_y, gamma=gamma_y)
Kt1 = compute_rbf_kernel_blockwise(test_x, train_x, gamma=gamma_x)

# Kx2 = rbf_kernel(train_x, train_x, gamma=gamma_x)
# # Ky = rbf_kernel(train_y, train_y, gamma=gamma_y)
# Kt2 = rbf_kernel(test_x, train_x, gamma=gamma_x)
# # print(f'Kx shape:   {Kx2.shape}')

# # if np.array_equal(Kx1, Kx2):
# if np.allclose(Kx1, Kx2, rtol=1e-5, atol=1e-7):
#     print('equal')
# else:
#     print('not equal')

# if np.allclose(Kt1, Kt2, rtol=1e-5, atol=1e-7):
#     print('equal')
# else:
#     print('not equal')



N = train_x.shape[0]
unique_domains = np.unique(domain_idx)
Q = np.zeros((N, N))

for d in unique_domains:
    idx = np.where(domain_idx == d)[0]
    nd = len(idx)
    Q[np.ix_(idx, idx)] += 1.0 / nd
Q /= len(unique_domains)



lambda_ = 1e-3
epsilon = 1e-3
m = 2  # Number of components to keep

V, D, Z_train, Z_test = dica(Kx=Kx, Ky=Ky, Kt=Kt , groupIdx=domain_idx, lambd=lambda_, epsilon=epsilon, M=m)

print(f'X train: {train_x.shape}')
print(f'Z train {Z_train.shape}')

print(f'X test: {test_x.shape}')
print(f'Z test: {Z_test.shape}')

Z_train = Z_train.T
Z_test = Z_test.T


# ----------- 1. Raw features (no transformation) ----------------
reg_raw = LinearRegression()
reg_raw.fit(train_x, train_y)
preds_raw = reg_raw.predict(test_x)
mse_raw = mean_squared_error(test_y, preds_raw)
print(f"Raw feature MSE: {mse_raw:.4f}")

# ----------- 2. Causal subset features using s_hat -------------
# train_x_s = train_x[:, s_hat]
# test_x_s = test_x[:, s_hat]

# reg_subset = LinearRegression()
# reg_subset.fit(train_x_s, train_y)
# preds_subset = reg_subset.predict(test_x_s)
# mse_subset = mean_squared_error(test_y, preds_subset)
# print(f"Causal subset (s_hat) MSE: {mse_subset:.4f}")


#-------------Greedy subset---------------
# s_hat_greedy = greedy_subset(train_x, train_y, n_ex, valid_split=0.5, 
#                              delta=0.05, use_hsic=use_hsic)

# print(f'Greedy subset: {s_hat_greedy}')

# train_x_greedy_s = train_x[:, s_hat_greedy]
# test_x_greedy_s = test_x[:, s_hat_greedy]

# reg_greedy_subset = LinearRegression()
# reg_greedy_subset.fit(train_x_greedy_s, train_y)
# preds_greedy_subset = reg_greedy_subset.predict(test_x_greedy_s)
# mse_greedy_subset = mean_squared_error(test_y, preds_greedy_subset)
# print(f"Raw feature MSE: {mse_raw:.4f}")
# print(f"Causal subset (s_hat) MSE: {mse_subset:.4f}")

# print(f"Greedy subset (s_hat) MSE: {mse_greedy_subset:.4f}")




# ----------- 3. DICA-transformed features -----------------------
# Nhớ rằng: X = V.T @ Kx_c → (m, N), Xt = V.T @ Kt_c.T → (m, Nt)


reg_dica = LinearRegression()
reg_dica.fit(Z_train, train_y)   # Z_train
preds_dica = reg_dica.predict(Z_test)  # Z_test
mse_dica = mean_squared_error(test_y, preds_dica)
print(f"DICA generalization MSE: {mse_dica:.4f}")


# B, K = dica_projection(train_x, train_y, domain_idx, lambda_=1.0, epsilon=1e-3, m=5)

# # Project training kernel to domain-invariant space
# K_proj = project_kernel(K, B)

# # Compute test kernel
# Kt = rbf_kernel(test_x, train_x, gamma=1.0 / train_x.shape[1])

# # Project test kernel
# Kt_proj = project_test_kernel(Kt, B, K)

# # Optionally use Ridge regression

# reg = LinearRegression()
# reg.fit(K_proj, train_y)
# preds = reg.predict(Kt_proj)
# mse_dica_2 = mean_squared_error(test_y, preds)
# print(f"DICA generalization MSE 2: {mse_dica_2:.4f}")
