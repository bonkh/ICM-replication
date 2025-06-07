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

import pickle
import os

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

save_all = {}
save_all["results"] = results
save_all["plotting"] = [methods, color_dict, legends, markers]
save_all["n_train_tasks"] = n_train_tasks

file_name = ["tl", str(n_repeat), str(eps), str(g), str(lambd)]
file_name = "_".join(file_name)


dif_inter = [[], [0], [0, 1], [0, 1, 2]]

# count = np.zeros((len(dif_inter), p))
count_subset = np.zeros((len(dif_inter), p))  # For subset search
count_icp = np.zeros((len(dif_inter), p))  

for ind_l, l_d in enumerate(dif_inter):
    print(f"Intervened index: {l_d}")
    for rep in range(n_repeat):
        print(f"Repeat: {rep}")

        where_to_intervene = l_d
        mask = intervene_on_p(where_to_intervene, p - p_s)

        dataset = gauss_tl(n_task, n, p, p_s, p_conf, eps, g, lambd, lambd_test, mask)
        x_train = dataset.train["x_train"]
        y_train = dataset.train["y_train"]
        n_ex = dataset.n_ex
        s_hat = subset_search.subset(
            x_train, y_train, n_ex, valid_split=0.6, delta=alpha_test, use_hsic=use_hsic
        )

        # for pred in range(p):
        #     if pred in s_hat:
        #         count[ind_l, pred] += 1
        for pred in range(p):
            if pred in s_hat:
                count_subset[ind_l, pred] += 1

        print(f'X train_shape: {x_train.shape}')
        print(f'y_train_shape: {y_train.shape}')

        envs = []
        start = 0
        for n in n_ex:
            end = start + n
            env_data = np.column_stack(
                [x_train[start:end], x_train[start:end]]
            )  # Combine X and y
            print(env_data.shape)
            envs.append(env_data)
            start = end

        data_list = envs
        target_index = x_train.shape[1]
        print(target_index)

        alpha = 0.05  # Set your significance level
        verbose = False

        try:
            result = fit(data_list, target=target_index, alpha=alpha, verbose=False)
            print("ICP result:", result.estimate)
            print('haha')

            accepted_features = list(result.estimate)
            for pred in accepted_features:
                count_icp[ind_l, pred] += 1
        except Exception as e:
            print("ICP failed:")
            traceback.print_exc()
                


print(count_icp)

# Save pickle
# save_all = {"count": count, "n_repeat": n_repeat, "inter": dif_inter}
# with open(os.path.join(save_dir, file_name + ".pkl"), "wb") as f:
#     pickle.dump(save_all, f)
save_all = {
    "count_subset": count_subset,
    "count_icp": count_icp,
    "n_repeat": n_repeat,
    "inter": dif_inter,
}

with open(os.path.join(save_dir, file_name + ".pkl"), "wb") as f:
    pickle.dump(save_all, f)

# Create plot
plot_interv(os.path.join(save_dir, file_name + ".pkl"),  'plot_ICM_cICM_1.pdf')
