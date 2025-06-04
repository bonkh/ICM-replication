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
import ast


obs_data = pd.read_csv('observational_data.csv')
obs_data = obs_data.drop(columns=["Unnamed: 0"])

int_data = pd.read_csv('interventional_data.csv')
int_data = int_data.drop(columns=["Unnamed: 0"])

int_pos_data = pd.read_csv('interventional_position_data.csv')
edges = pd.read_csv("gene_network_edge.csv")
nodes = pd.read_csv('node.csv')

# with open('top10_predictors_per_gene.json', 'r') as f:
#     data_dict = json.load(f)

data = pd.read_csv('lasso_10.csv')
data['top10_predictors'] = data['top10_predictors'].apply(ast.literal_eval)


data_dict = dict(zip(data['target_gene'], data['top10_predictors']))

print(data.info())
# print(f' data: {len(data_dict)}')
intervened_genes = set(int_pos_data['Mutant'])
print(len(intervened_genes))

filtered_data = filter_predictors_by_intervention(data_dict, intervened_genes)

 
with open('icp_resample_causal_genes.json', 'r') as f:
    gene_causes = json.load(f)


causal_dict, non_causal_dict = split_by_causes_v2(filtered_data, gene_causes, int_pos_data, obs_data)

print("Causal scenarios:", len(causal_dict))
print("Non-causal scenarios:", len(non_causal_dict))

# non_causal_result, detailed_result = evaluate_gene_invariance(non_causal_dict, data_dict, obs_data, int_data, int_pos_data, gene_causes, scenario = 'non-causal')
# with open("non_causal_result.json", "w") as f:
#     json.dump(non_causal_result, f, indent=2)
# with open("detailed_results.json", "w") as f:
#     json.dump(detailed_result, f, indent=2)

# plot_all_errors(non_causal_result)

