from sklearn.linear_model import LinearRegression
import pandas as pd
import sys
import os
from collections import defaultdict
import gc
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import pearsonr
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
    intervened_dict = {}

    for target_gene, predictors in data.items():
        filtered = [gene for gene in predictors if gene in intervened_genes]
        if filtered:
            intervened_dict[target_gene] = filtered

    return intervened_dict

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



def split_by_causes_v2(filtered_data, gene_causes, int_pos_data, obs_data):
    causal_dict = {}
    non_causal_dict = {}

    intervened_genes = set(int_pos_data['Mutant'])

    for target, predictors in filtered_data.items():
        known_causes = set(gene_causes.get(target, []))
        causal = [g for g in predictors if g in known_causes]

        if causal:
            causal_dict[target] = causal
            continue

        # Now handle non-causal case with correlation check
        strong_non_causal = []

        for g in predictors:

            if g not in obs_data.columns or target not in obs_data.columns:
                continue 

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

def evaluate_gene_invariance(intervened_gene_dict, top_10_gene_dict, obs_data, int_data, int_pos_data, gene_causes, scenario):

    detailed_result = []
    for idx, (target_gene, intervented_gene_list) in enumerate(intervened_gene_dict.items(), start=1):

        print(f"************* Target Gene #{idx}: {target_gene} *******************")
       
        top10_predictors = top_10_gene_dict[target_gene]

        for intervened_gene in intervented_gene_list:
            print(f'    Inter gene: {intervened_gene}')

            int_rows = int_pos_data[int_pos_data['Mutant'] == intervened_gene].index
            print(int_rows)
    
            if len(int_rows) == 0:
                continue

            held_out_idx = int_rows[0]


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
            print(y_test)

            # ******** Pool the data ********
            lr_pool = linear_model.LinearRegression()
            lr_pool.fit(X, y)
            X_test = int_data[top10_predictors].loc[[held_out_idx],:]
            linear_mse = mse(lr_pool, X_test, y_test)
            result['pool'].append (linear_mse)

            #  ******* Mean ********
            error_mean = np.mean((y_test - np.mean(y)) ** 2)
            result['mean'].append (error_mean)

            # ****** Subset search ******
            n_obs = len(X_obs)
            n_int = len(X_int)

            n_ex = [n_obs, n_int]
            
            s_hat = subset(X, y, n_ex, delta=alpha_test, valid_split=0.6, use_hsic=use_hsic)
            selected_genes = X.columns[s_hat]
            print(f"        Top 10 lasso genes: {X.columns} ---------")
            print(f"        S hat (selected genes): {list(selected_genes)} ---------")

            if s_hat.size> 0:
                lr_s_hat = linear_model.LinearRegression()
                lr_s_hat.fit(X.iloc[:,s_hat], y)

                selected_features = X.columns[s_hat]
                X_test = int_data[top10_predictors].loc[[held_out_idx], selected_features]
                mse_s_hat = mse(lr_s_hat, X_test, y_test)
                result['shat'].append(mse_s_hat)
                print(f'The error: {mse_s_hat}')
                
                del lr_s_hat
                gc.collect()
            else:
                result['shat'].append (error_mean)

            # ***** Causal ******

            if scenario == 'causal':
                mse_causal = linear_mse
                result['strue'].append(linear_mse)
            elif scenario == 'non-causal':
                if intervened_gene in top10_predictors:
                    predictors_wo_intervened = [g for g in top10_predictors if g != intervened_gene]
                else:
                    predictors_wo_intervened = top10_predictors

                X_obs_causal = obs_data[predictors_wo_intervened]
                X_int_causal = int_data_subset[predictors_wo_intervened]
                X_causal = pd.concat([X_obs_causal, X_int_causal], axis=0)
                X_test = int_data.loc[[held_out_idx], predictors_wo_intervened]

                print(X_causal.shape)

                lr_causal_wo_intervened = linear_model.LinearRegression()
                lr_causal_wo_intervened.fit(X_causal, y)
                mse_causal = mse(lr_causal_wo_intervened, X_test, y_test)
                result['strue'].append(mse_causal)

            detailed_result.append({
            'target_gene': target_gene,
            'intervened_gene': intervened_gene,
            'top10_predictors': list(top10_predictors),  
            's_hat_genes': list(selected_genes),     
            'mse_shat': mse_s_hat,    
            'mse_causal': mse_causal
            })


    return result, detailed_result


legends = {    'strue' : r'$\beta^{CS(cau)}$',
              'shat' : r'$\beta^{CS(\hat S Lasso)}$',
               'pool' : r'$\beta^{CS}$',
              'mean'   : r'$\beta^{mean}$',
            }
def plot_all_errors(results, output_pdf='all_error_boxplots.pdf'):
    if not results:
        print("No results to plot.")
        return

    ordered_keys = [k for k in legends if k in results and len(results[k]) > 0]
    if not ordered_keys:
        print("No valid results to plot.")
        return

    # Prepare the data in order
    data = [results[k] for k in ordered_keys]
    labels = [legends[k] for k in ordered_keys]

    # Plot
    plt.figure(figsize=(10, 6))
    plt.boxplot(
        data,
        patch_artist=True,
        boxprops=dict(facecolor='lightblue'),
        medianprops=dict(color='red'),
        showfliers=False
    )

    plt.xticks(ticks=range(1, len(labels) + 1), labels=labels)
    plt.title('Boxplot of prediction errors by method')
    plt.ylabel('Mean Squared Error')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save to PDF
    with PdfPages(output_pdf) as pdf:
        pdf.savefig()
        plt.close()

    print(f"Boxplot saved to {output_pdf}")
