from numpy import *
import numpy as np
import collections, re
import sys
from argparse import ArgumentParser
from sklearn.model_selection import KFold
from sklearn import linear_model
import utils

def mDA(X, p):
    X = vstack((X,ones((1,shape(X)[1]))))
    d = shape(X)[0]
    q = vstack((ones((d-1,1))*(1-p),1))
    S = dot(X,X.transpose())
    Q = S * dot(q,q.transpose())
    for i in range(d):
        Q[i,i] = q[i,0]*S[i,i]
    qrep = q.transpose()
    for j in range(d-1):
        qrep = vstack((qrep,q.transpose()))
    P = S * qrep
    W = linalg.solve((Q + 10**(-5)*eye(d)).transpose(), P[:d-1].transpose()).transpose()

    h = (dot(W,X))
    return (W,h)

def mSDA(X, p, l):
    d,n = shape(X)
    Ws = zeros((l,d,d+1))
    hs = zeros((l+1,d,n))
    hs[0] = X 
    for t in range(l):
        Ws[t], hs[t+1] = mDA(hs[t], p)
    return (Ws, hs)

def mSDA_features(w, x):

  for i in range(w.shape[0]):
    x = append(ones((1,x.shape[1])), x,0)
    x = dot(w[i], x)

  return x

def mSDA_cv(p, x, y, n_cv=5):

    kf = KFold(n_splits=n_cv)
    res = np.zeros((p.size, n_cv))

    for j, pj in enumerate(p):
        print(f"Type of pj: {type(pj)}, dtype: {np.array(pj).dtype}")  # Check type and dtype of pj
        i = 0
        for train, test in kf.split(x):
            x_temp, y_temp = x[train], y[train]
            x_test, y_test = x[test], y[test]

            fit_sda = mSDA(x_temp.T, pj, 1)
            x_sda = fit_sda[-1][-1].T
            w_sda = fit_sda[0]
            x_test_sda = mSDA_features(w_sda, x_test.T).T

            lr_sda = linear_model.LinearRegression()
            lr_sda.fit(x_sda, y_temp)
            mse_value = utils.mse(lr_sda, x_test_sda, y_test)
            print(f"Type of mse_value: {type(mse_value)}, dtype: {np.array(mse_value).dtype}")  # Check type and dtype of mse_value
            res[j, i] = mse_value

            i += 1
    res = np.mean(res, 1)
    print(f"Type of res: {type(res)}, dtype: {res.dtype}")  # Check type and dtype of res
    print(f"Type of res[0]: {type(res[0])}, dtype: {np.array(res[0]).dtype}")  # Check type and dtype of an element in res
    best_p = p[np.argmin(res)]
    print(f"Type of best_p: {type(best_p)}, dtype: {np.array(best_p).dtype}")  # Check type and dtype of best_p
    return best_p