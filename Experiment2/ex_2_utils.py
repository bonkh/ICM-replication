from sklearn.linear_model import LinearRegression
import pandas as pd
import sys
import os
from collections import defaultdict
import gc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json


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

from scipy.stats import pearsonr

def split_by_causes_v2(filtered_data, gene_causes, int_pos_data, obs_data):
    causal_dict = {}
    non_causal_dict = {}

    intervened_genes = set(int_pos_data['Mutant'])

    for target, predictors in filtered_data.items():
        known_causes = set(gene_causes.get(target, []))
        causal = [g for g in predictors if g in known_causes]

        if causal:
            causal_dict[target] = causal
            continue  # skip the rest, since it's a causal case

        # Now handle non-causal case with correlation check
        strong_non_causal = []

        for g in predictors:
            if g not in intervened_genes:
                continue  # only consider predictors that were intervened on

            if g not in obs_data.columns or target not in obs_data.columns:
                continue  # skip if missing in obs data

            try:
                corr, pval = pearsonr(obs_data[g], obs_data[target])
                # if pval == 0.0:
                if pval < 1e-10 and abs(corr) > 0.75:
                    strong_non_causal.append(g)
            except Exception:
                continue

        if strong_non_causal:
            non_causal_dict[target] = strong_non_causal

    return causal_dict, non_causal_dict


alpha_test = 0.05
use_hsic = 1

result = defaultdict(list)



def evaluate_gene_invariance(intervened_gene_dict, top_10_gene_dict, obs_data, int_data, int_pos_data, gene_causes):
    """
    gene_dict: {target_gene: [top10_predictors]}
    algorithm1_fn: function like def algorithm1_fn(X, y, predictors): return list of invariant predictors
    """
    for idx, (target_gene, intervented_gene_list) in enumerate(intervened_gene_dict.items(), start=1):

        print(f"************* Target Gene #{idx}: {target_gene} *******************")
       
        top10_predictors = top_10_gene_dict[target_gene]

        for intervened_gene in intervented_gene_list:
            print(f'Inter gene: {intervened_gene}')

            int_rows = int_pos_data[int_pos_data['Mutant'] == intervened_gene].index
    
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
            y = pd.concat([y_obs, y_int], axis=0).to_frame()
            
            y_test = int_data.loc[held_out_idx, target_gene]

            # ******** Pool the data ********
            lr_pool = linear_model.LinearRegression()
            lr_pool.fit(X, y)
            X_test = int_data[top10_predictors].loc[[held_out_idx],:]
            result['pool'].append (mse(lr_pool, X_test, y_test))

            #  ******* Mean ********
            error_mean = np.mean((y_test - np.mean(y)) ** 2)
            result['mean'].append (error_mean)

            # ****** Subset search ******
            n_obs = len(X_obs)
            n_int = len(X_int)

            n_ex = [n_obs, n_int]
            
            s_hat = subset(X, y, n_ex, delta=alpha_test, valid_split=0.6, use_hsic=use_hsic)

            if s_hat.size> 0:
                lr_s_hat = linear_model.LinearRegression()
                lr_s_hat.fit(X.iloc[:,s_hat], y)

                selected_features = X.columns[s_hat]
                X_test = int_data[top10_predictors].loc[[held_out_idx], selected_features]
                result['shat'].append(mse(lr_s_hat, X_test, y_test))
                
                del lr_s_hat
                gc.collect()
            else:
                result['shat'].append (error_mean)

            # ***** Causal ******
            if target_gene in gene_causes:
                causal_cause = gene_causes[target_gene]
                if isinstance(causal_cause, str):
                    causal_cause = [causal_cause]



                X_obs_causal = obs_data[causal_cause]
                X_int_causal = int_data_subset[causal_cause]

                X_causal = pd.concat([X_obs_causal, X_int_causal], axis=0)
                X_test = int_data.loc[[held_out_idx], causal_cause]


      
                lr_true_causal = linear_model.LinearRegression()
                lr_true_causal.fit(X_causal, y)
                
                result['strue'].append(mse(lr_true_causal,X_test, y_test))
            else:
                result['shat'].append (error_mean)

    return result

def plot_all_errors(results, output_pdf='all_error_boxplots.pdf'):
    if not results:
        print("No results to plot.")
        return

    # Filter out keys with empty lists
    filtered_results = {k: v for k, v in results.items() if len(v) > 0}
    if not filtered_results:
        print("No non-empty error lists to plot.")
        return

    plt.figure(figsize=(10, 6))

    # Create the boxplot
    plt.boxplot(
        filtered_results.values(),
        patch_artist=True,
        boxprops=dict(facecolor='lightblue'),
        medianprops=dict(color='red'),
        showfliers=False 
    )

    # Set labels and title
    plt.xticks(ticks=range(1, len(filtered_results) + 1), labels=filtered_results.keys())
    plt.title('Boxplot of prediction errors by method')
    plt.ylabel('Mean Squared Error')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save to PDF
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()

    print(f"Boxplot saved to {output_pdf}")