import numpy as np
from sklearn import linear_model

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

def np_getDistances(x,y):
    K = (x[:,:, np.newaxis] - y.T)
    return np.linalg.norm(K,axis = 1)

    
#Select top 11 predictors from Lasso
def lasso_alpha_search_synt(X,Y):
    from sklearn import linear_model

    exit_loop = False
    alpha_lasso = 0.2
    step = 0.02
    num_iters = 1000
    count = 0
    n = 11

    while(not exit_loop and count < num_iters):
            count = count + 1

            regr = linear_model.Lasso(alpha = alpha_lasso)
            regr.fit(X,Y.flatten())
            zeros =  np.where(np.abs(regr.coef_) < 0.00000000001)

            nonzeros = X.shape[1]-zeros[0].shape[0]

            if(nonzeros >= n and nonzeros<n+1):
                    exit_loop = True
            if nonzeros<n:
                    alpha_lasso -= step
            else:
                    step /= 2
                    alpha_lasso += step


    mask = np.ones(X.shape[1],dtype = bool)
    mask[zeros] = False
    genes = []
    index_mask = np.where(mask == True)[0]

    return mask



def train_linear_and_eval(x, y, x_test, y_test):
    model = linear_model.LinearRegression()
    model.fit(x, y)
    result = mse(model, x_test, y_test)
    return result