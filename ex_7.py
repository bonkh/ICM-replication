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
parser.add_argument("--lambd_test", default=0.99)
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
methods = ["pool", "shat", "sgreed", "strue", "mean", "msda"]

color_dict, markers, legends = utils.get_color_dict()

for m in methods:
    results[m] = np.zeros((n_repeat, n_train_tasks.size))

# for rep in xrange(n_repeat):

save_all = {}
save_all["results"] = results
save_all["plotting"] = [methods, color_dict, legends, markers]
save_all["n_train_tasks"] = n_train_tasks

file_name = ["tl", str(n_repeat), str(eps), str(g), str(lambd)]
file_name = "_".join(file_name)


dif_inter = [[], [0], [0, 1], [0, 1, 2]]

count = np.zeros((len(dif_inter), p))

results = {}
methods = ["mean", "shat", "cicm"]

results = {}
for m in methods:
  results[m]  = np.zeros((n_repeat, len(dif_inter)))


count_subset = np.zeros((len(dif_inter), p))
count_icp = np.zeros((len(dif_inter), p))

for ind_l, l_d in enumerate(dif_inter):
    for rep in range(n_repeat):

        where_to_intervene = l_d
        mask = utils.intervene_on_p(where_to_intervene, p - p_s)

        dataset = data.gauss_tl(
            n_task, n, p, p_s, p_conf, eps, g, lambd, lambd_test, mask
        )
        x_train = dataset.train["x_train"]
        y_train = dataset.train["y_train"]
        x_test = dataset.test["x_test"]
        y_test = dataset.test["y_test"]

        n_ex = dataset.n_ex

        print("------------- 0. Mean error ----------------")

        error_mean = np.mean((y_test - np.mean(y_train)) ** 2)
        results["mean"][rep, ind_l] = error_mean
        print(error_mean)

        print("----------- 1. Subset search - ICM ------------- ")

        s_hat = subset_search.subset(
            x_train,
            y_train,
            dataset.n_ex,
            valid_split=0.6,
            delta=alpha_test,
            use_hsic=use_hsic,
        )

        # for pred in range(p):
        #   if pred in s_hat:
        #     count[ind_l, pred] += 1
        print(s_hat)

        for pred in range(p):
            if pred in s_hat:
                count_subset[ind_l, pred] += 1

        if s_hat.size > 0:
            lr_subset_search = linear_model.LinearRegression()
            lr_subset_search.fit(x_train[:, s_hat], y_train)
            results["shat"][rep, ind_l] = mse(
                lr_subset_search, x_test[:, s_hat], y_test
            )
        else:
            results["shat"][rep, ind_l] = error_mean

        print("------------ 2. cICM ----------------")
        envs = []
        start = 0
        for n in n_ex:
            end = start + n
            env_data = np.column_stack(
                [x_train[start:end], y_train[start:end]]
            )  # Combine X and y

            envs.append(env_data)
            start = end

        data_list = envs
        target_index = x_train.shape[1]
        alpha = 0.05  # Set your significance level
        verbose = False

        try:
            result = fit(data_list, target=target_index, alpha=0.05, verbose=True)
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
            lr_cicm.fit(x_train[:, selected_features], y_train)
            results["cicm"][rep, ind_l] = mse(
                lr_cicm, x_test[:, selected_features], y_test
            )

            del lr_cicm
            gc.collect()
        else:
            results["cicm"][rep, ind_l] = error_mean


# Save pickle
# save_all = {'count': count, 'n_repeat': n_repeat, 'inter' : dif_inter}

save_all = {
    "count_subset": count_subset,
    "count_icp": count_icp,
    "n_repeat": n_repeat,
    "inter": dif_inter,
}
with open(os.path.join(save_dir, file_name + ".pkl"), "wb") as f:
    pickle.dump(save_all, f)

# zplot_interv(os.path.join(save_dir, file_name + '.pkl'))


save_all_2 = {
    "results": results,
    "plotting": [methods, color_dict, legends, markers],
    "n_repeat": n_repeat,
    "inter": dif_inter,
}

file_name = "mse_icm_vs_cicm"

with open(os.path.join(save_dir, file_name + ".pkl"), "wb") as f:
    pickle.dump(save_all_2, f)
