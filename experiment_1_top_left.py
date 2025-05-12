
import numpy as np
from sklearn import linear_model

from sklearn.metrics.pairwise import rbf_kernel
import matplotlib.pyplot as plt
import argparse
import subset_search
import  pickle
import os
from data import *
from utils import *
from msda import *
from dica import *
from plotting import *
np.random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', default = 'Experiment_01_top_left')
parser.add_argument('--n_task', default=7)
parser.add_argument('--merge_dica', default=0)
parser.add_argument('--n', default=400)
parser.add_argument('--p', default = 6)
parser.add_argument('--p_s', default = 3)
parser.add_argument('--p_conf', default = 1)
parser.add_argument('--eps', default = 2)
parser.add_argument('--g', default = 1)
parser.add_argument('--lambd', default = 0.5)
parser.add_argument('--lambd_test', default = 0.99)
parser.add_argument('--use_hsic', default = 0)
parser.add_argument('--alpha_test', default = 0.05)
parser.add_argument('--n_repeat', default = 5)
parser.add_argument('--max_l', default = 100)
parser.add_argument('--n_ul', default = 100)
args = parser.parse_args()

save_dir = args.save_dir

if not os.path.exists(save_dir):
  os.makedirs(save_dir)

# save_dir = os.path.join(save_dir, 'fig4_tleft')
# if not os.path.exists(save_dir):
#   os.makedirs(save_dir)


n_task = int(args.n_task)
n = int(args.n)

p = int(args.p)
p_s = int(args.p_s)
p_conf = int(args.p_conf)
eps = float(args.eps)
g = float(args.g)

lambd = float(args.lambd)
lambd_test = float(args.lambd_test)

alpha_test = float(args.alpha_test)
use_hsic = bool(int(args.use_hsic))

dataset = gauss_tl(n_task, n, p, p_s, p_conf, eps ,g, lambd, lambd_test)
x_train = dataset.train['x_train'].astype(np.float32)
y_train = dataset.train['y_train'].astype(np.float32)
n_ex = dataset.train['n_ex']

lr = linear_model.LinearRegression()
lr.fit(x_train, y_train)

# Define test
x_test = dataset.test['x_test'].astype(np.float32)
y_test = dataset.test['y_test'].astype(np.float32)

n_train_tasks = np.arange(2, n_task)
n_repeat = int(args.n_repeat)

true_s = np.arange(p_s)

results = {}
methods = [
            'pool',
            'shat',
            'sgreed',
            'strue',
            'mean',
            'msda'
            # 'dica'
          ]

n_train_tasks = np.arange(n_train_tasks[0], n_train_tasks[-1] + 1, 1)
color_dict, markers, legends = get_color_dict()

for m in methods:
  results[m]  = np.zeros((n_repeat, n_train_tasks.size))

for rep in range(n_repeat):
  print (f'******** REPEAT: {rep}**********') 

    
 
  x_train, y_train = dataset.resample(n_task, n)
  # print(x_train.shape)
  # x_train_path = os.path.join(save_dir, 'x_train.dat')
  # y_train_path = os.path.join(save_dir, 'y_train.dat')
  # np.memmap(x_train_path, dtype='float32', mode='w+', shape=x_train.shape)[:] = x_train.astype('float32')
  # np.memmap(y_train_path, dtype='float32', mode='w+', shape=y_train.shape)[:] = y_train.astype('float32')




  x_test = dataset.test['x_test']
  y_test = dataset.test['y_test']

  # Load from disk without reading everything into memory
  # x_train = np.memmap(x_train_path, dtype='float32', mode='r', shape=(n_task * n, p))
  # y_train = np.memmap(y_train_path, dtype='float32', mode='r', shape=(n_task * n, 1))
  # end = get_memory_usage_gb()
  # print(f"RAM Used: {end - start:.2f} MB")  


  for index, t in np.ndenumerate(n_train_tasks):
    x_temp = x_train[0:np.cumsum(n_ex)[t], :]
    y_temp = y_train[0:np.cumsum(n_ex)[t], :]

    print (f'***** Number of tasks used: {t} ******')
    print (f'X_train shape: {x_temp.shape}')
    start = get_memory_usage_gb()


    # ************** 1. Pooled *********************
    print(f'1. Pooling the data')
    results['pool'][rep, index] = train_linear_and_eval(x_temp, y_temp, x_test, y_test)

    # *************** 2. Mean ************
    print (f'2. Mean prediction')
    error_mean = np.mean((y_test - np.mean(y_temp)) ** 2)
    results['mean'][rep, index] = error_mean

    # ************* 3. Estimated S_hat ***********
    print (f'3. Subset search')

    if p<10:
      s_hat = subset_search.subset(x_temp, y_temp, n_ex[0:t], 
                                   delta=alpha_test, valid_split=0.6, 
                                   use_hsic=use_hsic)
    
  
    if p<10:
      if s_hat.size> 0:
        lr_s_temp = linear_model.LinearRegression()
        lr_s_temp.fit(x_temp[:,s_hat], y_temp)
        results['shat'][rep, index] = mse(lr_s_temp, x_test[:,s_hat], 
                                                y_test)
        
        del lr_s_temp
        gc.collect()
      else:
        results['shat'][rep,index] = error_mean

      

    # ************** 4. Estimated greedy S ******************
    print(f'4. Greedy subset search')
    s_greedy = subset_search.greedy_subset(x_temp, y_temp, n_ex[0:t], 
                                           delta=alpha_test, 
                                           valid_split=0.8, 
                                           use_hsic=use_hsic)

    if s_greedy.size> 0:
      lr_sg_temp = linear_model.LinearRegression()
      lr_sg_temp.fit(x_temp[:,s_greedy], y_temp)
      results['sgreed'][rep, index] = mse(lr_sg_temp, x_test[:,s_greedy], y_test)

      del lr_sg_temp
      gc.collect()
    else:
      results['sgreed'][rep, index] = error_mean


    # ************ 5. True S **************
    print(f'5. True causal predictor')
    lr_true_temp = linear_model.LinearRegression()
    lr_true_temp.fit(x_temp[:,true_s], y_temp)
    results['strue'][rep, index] = mse(lr_true_temp,x_test[:,true_s], y_test)

    del lr_true_temp
    gc.collect()

    # ************ 6. mSDA *************
    print(f'6. mSDA')
    p_linsp = np.linspace(0,1,10)
    p_cv = mSDA_cv(p_linsp, x_temp, y_temp, n_cv = t)
    fit_sda = mSDA(x_temp.T,p_cv,1)
    x_sda = fit_sda[-1][-1].T
    w_sda = fit_sda[0]
    x_test_sda = mSDA_features(w_sda, x_test.T).T

    lr_sda = linear_model.LinearRegression()
    lr_sda.fit(x_sda, y_temp)
    results['msda'][rep, index] = utils.mse(lr_sda, x_test_sda, y_test)

    del lr_sda
    gc.collect()

    # *********** 7. DICA ********
    # print (f'7. DICA')
    
    # # domain_idx = []

    # # for domain, n in enumerate(n_ex):
    # #     domain_idx.extend([domain] * n)
    # # domain_idx = np.array(domain_idx)

    # domain_idx_path = os.path.join(save_dir, 'domain_idx.dat')
    
    # domain_idx_memmap = np.memmap(domain_idx_path, dtype='int32', mode='w+', shape=(sum(n_ex)))
    # offset = 0
    # for domain, n in enumerate(n_ex):
    #     domain_idx_memmap[offset:offset+n] = domain
    #     offset += n
    # domain_idx_memmap.flush()  # ensure it's written

    # # domain_idx = np.memmap (domain_idx_path, dtype='int32', mode='r', shape=(sum(n_ex),))

    # # are_equal = np.allclose(domain_idx, domain_idx_2)
    # # print(are_equal)

    # # Use RBF kernel with chosen bandwidth
    # gamma_x = 1.0 / x_temp.shape[1]
    # gamma_y = 1.0

    # Kx_path = os.path.join(save_dir, 'Kx.dat')
    # Ky_path = os.path.join(save_dir, 'Ky.dat')
    # Kt_path = os.path.join(save_dir, 'Kt.dat')

    # Kx = compute_rbf_kernel_blockwise_to_memmap(x_temp, x_temp, gamma=gamma_x, mmap_path=Kx_path)
    # Ky = compute_rbf_kernel_blockwise_to_memmap(y_temp, y_temp, gamma=gamma_y, mmap_path=Ky_path)
    # Kt = compute_rbf_kernel_blockwise_to_memmap(x_test, x_temp, gamma=gamma_x, mmap_path=Kt_path)
    # N = x_temp.shape[0]
    # # Kx = np.memmap(Kx_path, dtype='float32', mode='r', shape=(N, N))
    # # Ky = np.memmap(Ky_path, dtype='float32', mode='r', shape=(N, N))
    # # Kt = np.memmap(Kt_path, dtype='float32', mode='r', shape=(x_test.shape[0], N))

  
    # # unique_domains = np.unique(domain_idx)
    # lambda_ = 1e-3
    # epsilon = 1e-3
    # m = 2

    # V, D, Z_train, Z_test = dica_torch(
    # Kx_path = Kx_path, 
    # Ky_path = Ky_path, 
    # Kt_path = Kt_path, 
    # N = N,
    # Nt = x_test.shape[0],
    # groupIdx_path = domain_idx_path, 
    # lambd=lambda_, 
    # epsilon=epsilon, 
    # M=m
    # )

    # Z_train = Z_train.T.cpu().numpy()
    # Z_test = Z_test.T.cpu().numpy()


    # # V, D, Z_train, Z_test = dica(
    # #   Kx_path = Kx_path, 
    # #   Ky_path = Ky_path, 
    # #   Kt_path = Kt_path,
    # #   N = N,
    # #   Nt = x_test.shape[0],
    # #   groupIdx_path = domain_idx_path,
    # #   lambd = lambda_, 
    # #   epsilon = epsilon, 
    # #   M = m
    # #   )

    # # Z_train = Z_train.T
    # # Z_test = Z_test.T


    # reg_dica = linear_model.LinearRegression()
    # reg_dica.fit(Z_train, y_temp)
    # results['dica'][rep, index] = utils.mse(reg_dica, Z_test, y_test) 

    # del reg_dica
    # del x_temp, y_temp, Kx, Ky, Kt, V, D, Z_train, Z_test
    # gc.collect()
    # end = get_memory_usage_gb()
    # print(f"RAM Used: {end - start:.2f} GB")  

  del x_train, y_train, x_test, y_test
  gc.collect()
  end = get_memory_usage_gb()
  print(f"RAM Used: {end - start:.2f} GB")  
                 

save_all = {}
save_all['results'] = results
save_all['plotting'] = [methods, color_dict, legends, markers]
save_all['n_train_tasks'] = n_train_tasks

#Save pickle file
file_name = ['tl_norm_', str(n_repeat), str(eps), str(g), str(lambd)]
file_name = '_'.join(file_name)

with open(os.path.join(save_dir, file_name+'.pkl'),'wb') as f:
  pickle.dump(save_all, f)

#Create plot
plot_tl(os.path.join(save_dir, file_name + '.pkl'), ylim=4)

