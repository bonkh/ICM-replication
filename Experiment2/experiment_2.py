import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import pandas as pd
from sklearn.linear_model import LassoCV
import numpy as np
from sklearn.preprocessing import StandardScaler
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from ex_2_utils import *
import time

obs_data = pd.read_csv('observational_data.csv')
obs_data = obs_data.drop(columns=["Unnamed: 0"])

int_data = pd.read_csv('interventional_data.csv')
int_data = int_data.drop(columns=["Unnamed: 0"])

int_pos_data = pd.read_csv('interventional_position_data.csv')
edges = pd.read_csv("gene_network_edge.csv")
nodes = pd.read_csv('node.csv')

with open('top10_predictors_per_gene.json', 'r') as f:
    data = json.load(f)

print(f' data: {len(data)}')
intervened_genes = set(int_pos_data['Mutant'])
print(len(intervened_genes))

filtered_data = filter_predictors_by_intervention(data, intervened_genes)
print(len(filtered_data)) 


with open('icp_resample_causal_genes copy.json', 'r') as f:
    gene_causes = json.load(f)

causal_dict, non_causal_dict = split_by_causes_v2(filtered_data, gene_causes, int_pos_data, obs_data)

print("Causal scenarios:", len(causal_dict))
print("Non-causal scenarios:", len(non_causal_dict))

# non_causal_result = evaluate_gene_invariance(non_causal_dict, data, obs_data, int_data, int_pos_data, gene_causes)
# with open("non_causal_result.json", "w") as f:
#     json.dump(non_causal_result, f, indent=2)

# plot_all_errors(non_causal_result)



# target_gene = 'YAL061W'
# intervened_gene = 'YOR173W'
# int_rows = int_pos_data[int_pos_data['Mutant'] == intervened_gene].index
# held_out_idx = int_rows[0]
# top10_predictors = data[target_gene]

# X_obs = obs_data[top10_predictors]
# y_obs = obs_data[target_gene]

# int_data_subset = int_data.drop(index=held_out_idx)
# causal_cause = gene_causes[target_gene]
# y_int = int_data_subset[target_gene]


# X_obs_causal = obs_data[causal_cause]
# X_int_causal = int_data_subset[causal_cause]

# X_causal = pd.concat([X_obs_causal, X_int_causal], axis=0).to_frame()
# X_test = int_data.loc[[held_out_idx], causal_cause].to_frame()
# y = pd.concat([y_obs, y_int], axis=0).to_frame()

# y_test = int_data.loc[held_out_idx, target_gene]

# print(X_causal.shape)

# lr_true_causal = linear_model.LinearRegression()
# lr_true_causal.fit(X_causal, y)
# print(lr_true_causal)
# print(mse(lr_true_causal,X_test, y_test))