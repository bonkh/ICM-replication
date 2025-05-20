from sklearn.linear_model import LinearRegression
import pandas as pd
import sys
import os
from collections import defaultdict
import gc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Add the parent directory (where subset_search.py is located) to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from subset_search import *
from utils import *

def filter_predictors_by_intervention(data, intervened_genes):
    """
    Filters the values of each key in the input dictionary, keeping only those
    predictors that are also in the intervened_genes set.

    Parameters:
    - data: dict, where each key is a gene and its value is a list of top predictors.
    - intervened_genes: set of genes that were intervened on.

    Returns:
    - new_dict: filtered dictionary with predictors that are in intervened_genes.
    """
    new_dict = {}

    for target_gene, predictors in data.items():
        filtered = [gene for gene in predictors if gene in intervened_genes]
        if filtered:
            new_dict[target_gene] = filtered

    return new_dict

def split_by_causes(filtered_data, gene_causes):
    causal_dict = {}
    non_causal_dict = {}

    for target, predictors in filtered_data.items():
        known_causes = set(gene_causes.get(target, []))
        causal = [g for g in predictors if g in known_causes]
        non_causal = [g for g in predictors if g not in known_causes]

        if causal:
            causal_dict[target] = causal
        if non_causal:
            non_causal_dict[target] = non_causal

    return causal_dict, non_causal_dict
alpha_test = 0.05
use_hsic = 0

result = defaultdict(list)



def evaluate_gene_invariance(intervened_gene_dict, top_10_gene_dict, obs_data, int_data, int_pos_data, algorithm1_fn=None):
    """
    gene_dict: {target_gene: [top10_predictors]}
    algorithm1_fn: function like def algorithm1_fn(X, y, predictors): return list of invariant predictors
    """

    results = []

    for target_gene, intervented_gene_list in intervened_gene_dict.items():

        print(f'Target_gene: {target_gene}')
       
        top10_predictors = top_10_gene_dict[target_gene]

        for intervened_gene in intervented_gene_list:
            print(f'Inter gene: {intervened_gene}')
            # Get rows where this predictor was intervened on
            int_rows = int_pos_data[int_pos_data['Mutant'] == intervened_gene].index
            print(int_rows)
            if len(int_rows) == 0:
                continue

            held_out_idx = int_rows[0]

            if held_out_idx not in int_data.index:
                continue

            # Create training data
            X_obs = obs_data[top10_predictors]
            y_obs = obs_data[target_gene]

            # Remove the held-out interventional point
            int_data_subset = int_data.drop(index=held_out_idx)
            X_int = int_data_subset[top10_predictors]
            y_int = int_data_subset[target_gene]

            # Combine data
            X = pd.concat([X_obs, X_int], axis=0)
            y = pd.concat([y_obs, y_int], axis=0)
            
            y_test = int_data.loc[held_out_idx, target_gene]

            error_mean = np.mean((y_test - np.mean(y)) ** 2)

            print(X.shape)
            n_obs = len(X_obs)
            n_int = len(X_int)

            n_ex = [n_obs, n_int]

            s_hat = subset(X, y, n_ex, delta=alpha_test, valid_split=0.6, use_hsic=use_hsic)
            if s_hat.size> 0:
                lr_s_hat = linear_model.LinearRegression()
                lr_s_hat.fit(X[:,s_hat], y)
                results['shat'].append(mse(lr_s_hat, int_data.loc[[held_out_idx], s_hat], 
                                                        y_test))
                
                del lr_s_hat
                gc.collect()
            else:
                results['shat'].append (error_mean)

    return results

def plot_shat_errors(results, output_pdf='shat_error_boxplot.pdf'):
    if 'shat' not in results or len(results['shat']) == 0:
        print("No 'shat' errors to plot.")
        return

    plt.figure(figsize=(8, 6))
    plt.boxplot(results['shat'], patch_artist=True, boxprops=dict(facecolor='lightblue'))

    plt.title('Boxplot of prediction errors for $\hat{S}$')
    plt.ylabel('Mean Squared Error')
    plt.grid(True)

    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()

    print(f"Boxplot saved to {output_pdf}")