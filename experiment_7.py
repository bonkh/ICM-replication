import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import argparse
import subset_search
from data import *
from utils import *
from msda import *
from dica import *
from icp import *
from plotting import *
import traceback
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import pickle
import os
from scipy.io import savemat


np.random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", default="Experiment_7")
parser.add_argument("--n_task", default=10)
parser.add_argument("--n", default=4000)
parser.add_argument("--p", default=6)
parser.add_argument("--p_s", default=3)
parser.add_argument("--p_conf", default=0)
parser.add_argument("--eps", default=2)
parser.add_argument("--g", default=1)
parser.add_argument("--lambd", default=0.5)
parser.add_argument("--lambd_test", default=0.5)
parser.add_argument("--use_hsic", default=0)
parser.add_argument("--alpha_test", default=0.05)
parser.add_argument("--n_repeat", default=100)
parser.add_argument("--max_l", default=100)
parser.add_argument("--n_ul", default=100)
args = parser.parse_args()

save_dir = args.save_dir

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


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

n_train_tasks = np.arange(2, n_task)
n_repeat = int(args.n_repeat)

true_s = np.arange(p_s)

results = {}
methods = ["mean", "causal" ,"icm","cicm"]

color_dict, markers, legends = utils.get_color_dict()



dif_inter = [[],[0], [0,1], [0,1,2]]

dif_inter_test = [[0,1, 2], [0, 1,2], [0,1,2], [0,1,2]]

# count = np.zeros((len(dif_inter), p))
count_subset = np.zeros((len(dif_inter), p))  # For subset search
count_icp = np.zeros((len(dif_inter), p))  

results = {}
results_ood = {}

for m in methods:
  results[m]  = np.zeros((n_repeat, len(dif_inter)))
  results_ood[m]  = np.zeros((n_repeat, len(dif_inter)))



for ind_l, l_d in enumerate(dif_inter):

    print(f"Intervened index: {l_d}")

    for rep in range(n_repeat):
        print(f"Repeat: {rep}")

        where_to_intervene = l_d
        mask = intervene_on_p(where_to_intervene, p - p_s)

        dataset = gauss_tl(n_task, n, p, p_s, p_conf, eps, g, lambd, lambd_test, mask )


       
        x_train = dataset.train["x_train"]
        y_train = dataset.train["y_train"]

        x_test = dataset.test['x_test']
        y_test = dataset.test['y_test']

        where_to_intervene_test = dif_inter_test[ind_l]
        mask_test = intervene_on_p(where_to_intervene_test, p - p_s)
        dataset_test = gauss_tl(n_task, n, p, p_s, p_conf, eps, g, lambd, lambd_test)
        x_test_ood = dataset_test.test['x_test']
        y_test_ood = dataset_test.test['y_test']

        n_ex = dataset.n_ex

        df_x = pd.DataFrame(x_train, columns=[f"X{i}" for i in range(x_train.shape[1])])
        df_y = pd.Series(y_train.ravel(), name="Y")

        # Compute correlations
        correlations = df_x.corrwith(df_y)
        correlations_sorted = correlations.reindex(correlations.abs().sort_values(ascending=False).index)
        print(correlations_sorted)

        print('-------------- 0. Mean prediction --------------')

        error_mean = np.mean((y_test - np.mean(y_train)) ** 2)
        results['mean'][rep, ind_l] = error_mean
        print(f'Error mean: {error_mean}')

        error_mean_ood = np.mean((y_test_ood - np.mean(y_train)) ** 2)
        results_ood['mean'][rep, ind_l] = error_mean_ood
        print(f'Error mean ood: {error_mean_ood}')

        # print('------------------- 1. True causal ------------------')
        # print(f'DATA ALPHA: {dataset.alpha}')
       
        
        # alpha = dataset.alpha
        # X = x_train[:, s_causal]

        # y_pred = np.dot(x_test[:, s_causal], alpha)
        # results['true_causal'][rep, ind_l] = np.mean((y_test - y_pred) ** 2)
        # print(f"Error: { results['true_causal'][rep, ind_l]}")


        # y_pred_ood = np.dot(x_test_ood[:, s_causal], alpha)
        # results_ood['true_causal'][rep, ind_l] = np.mean((y_test_ood - y_pred_ood) ** 2)
        # print(f"OOD Error: {results_ood['true_causal'][rep, ind_l]}")

        print ('------------- 1. Causal ----------------')
        s_causal =  np.arange(p_s)
     
        print(f'S causal: {s_causal}')

        lr_causal = linear_model.LinearRegression()
        lr_causal.fit(x_train[:,s_causal], y_train)

        results['causal'][rep, ind_l] = mse(lr_causal, x_test[:,s_causal], y_test)
        results_ood['causal'][rep, ind_l] = mse(lr_causal, x_test_ood[:,s_causal], y_test_ood)

        print (f"Causal error: {results['causal'][rep, ind_l]}")
        print(f"Causal error ood: { results_ood['causal'][rep, ind_l]}")


        print('----------- 2. Subset search - ICM ------------- ')

        s_hat = subset_search.subset(
            x_train, y_train, n_ex, valid_split=0.5, delta=alpha_test, use_hsic=use_hsic
        )
        print(f'S hat: {s_hat}')

        for pred in range(p):
            if pred in s_hat:
                count_subset[ind_l, pred] += 1

        if s_hat.size> 0:
            lr_subset_search = linear_model.LinearRegression()
            lr_subset_search.fit(x_train[:,s_hat], y_train)

            results['icm'][rep, ind_l] = mse(lr_subset_search, x_test[:,s_hat], y_test)
            results_ood['icm'][rep, ind_l] = mse(lr_subset_search, x_test_ood[:,s_hat], y_test_ood)

        else: 
            results['icm'][rep, ind_l] = error_mean
            results_ood['icm'][rep, ind_l] = error_mean_ood

        print(f"Error: {results['icm'][rep, ind_l]}")
        print(f"OOD error: {results_ood['icm'][rep, ind_l]}")


        print('------------ 2. cICM ----------------')
        envs = []
        start = 0
        for n in n_ex:
            end = start + n
            env_data = np.column_stack(
                [x_train[start:end], y_train[start:end]]
            )

            envs.append(env_data)
            start = end

        data_list = envs
        target_index = x_train.shape[1]
        alpha = 0.1
        verbose = False

        try:
            result = fit(data_list, target=target_index, alpha=alpha, verbose=False)
            print("ICP result:", result.estimate)


            accepted_features = list(result.estimate)

            for pred in range(p):
                if pred in accepted_features:
                    count_icp[ind_l, pred] += 1

            
        except Exception as e:
            print("ICP failed:")
            traceback.print_exc()

        if result.estimate is not None and len(result.estimate) > 0:

            selected_features = list(result.estimate)

            lr_cicm = linear_model.LinearRegression()
            lr_cicm.fit(x_train[:,selected_features], y_train)


            results['cicm'][rep, ind_l] = mse(lr_cicm, x_test[:,selected_features], y_test)
            results_ood['cicm'][rep, ind_l] = mse(lr_cicm, x_test_ood[:,selected_features], y_test_ood)


        else:
            results['cicm'][rep, ind_l] = error_mean
            results_ood['cicm'][rep, ind_l] = error_mean_ood
        

        print(f"Error: {results['cicm'][rep, ind_l]}")
        print(f"OOD error: {results_ood['cicm'][rep, ind_l]}")
                        


print(f' Count ICP: {count_icp}')


print(f' Count subset: {count_subset}')


save_all = {
    "count_subset": count_subset,
    "count_icp": count_icp,
    "n_repeat": n_repeat,
    "inter": dif_inter,
}

file_name = "icm_vs_cicm"


with open(os.path.join(save_dir, file_name + ".pkl"), "wb") as f:
    pickle.dump(save_all, f)


save_all_error_scen1 = {
    "results" : results,
    "plotting": [methods, color_dict, legends, markers],
    "n_repeat": n_repeat,
    "inter": dif_inter,
}

file_name = "mse_icm_vs_cicm_scen1"

with open(os.path.join(save_dir, file_name+'.pkl'),'wb') as f:
  pickle.dump(save_all_error_scen1, f)


save_all_error_scen2 = {
    "results" : results_ood,
    "plotting": [methods, color_dict, legends, markers],
    "n_repeat": n_repeat,
    "inter": dif_inter,
}

file_name = "mse_icm_vs_cicm_scen2"

with open(os.path.join(save_dir, file_name+'.pkl'),'wb') as f:
  pickle.dump(save_all_error_scen2, f)