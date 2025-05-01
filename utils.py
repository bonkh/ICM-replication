import numpy as np

from sklearn import linear_model
from sklearn.metrics.pairwise import rbf_kernel
import psutil
import os
def train_linear_and_eval(x, y, x_test, y_test):
    model = linear_model.LinearRegression()
    model.fit(x, y)
    result = mse(model, x_test, y_test)
    return result


def get_memory_usage_gb():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3  # In MB

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