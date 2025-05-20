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


intervened_genes = set(int_pos_data['Mutant'])
print(len(intervened_genes))

filtered_data = filter_predictors_by_intervention(data, intervened_genes)
print(len(filtered_data)) 


with open('top_correlation_causes.json', 'r') as f:
    gene_causes = json.load(f)

causal_dict, non_causal_dict = split_by_causes(filtered_data, gene_causes)

print("Causal targets:", len(causal_dict))
print("Non-causal targets:", len(non_causal_dict))

result = evaluate_gene_invariance(non_causal_dict, data, obs_data, int_data, int_pos_data, gene_causes)

plot_all_errors(result)
