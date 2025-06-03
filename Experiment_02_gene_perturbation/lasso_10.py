import numpy as np
import pandas as pd
import json
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

int_data = pd.read_csv("data/interventional_data.csv", index_col=0)
obs_data = pd.read_csv("data/observational_data.csv", index_col=0)
int_pos_data = pd.read_csv('data/interventional_position_data.csv', index_col=0)
gene_names = list(obs_data.columns)

def get_task_data_and_top10(i, obs_data, int_data, gene_names):
    """
    i: index of target gene
    obs_data: DataFrame chứa observational data (n_obs × p)
    int_data: DataFrame chứa intervention data (n_int × p)
    gene_names: list tên các gene (len = p)
    """
    target_gene = gene_names[i]
    predictor_genes = [g for g in gene_names if g != target_gene]

    X_obs = obs_data[predictor_genes]
    y_obs = obs_data[target_gene]

    X_int = int_data[predictor_genes]
    y_int = int_data[target_gene]

    X = pd.concat([X_obs, X_int], axis=0)
    y = pd.concat([y_obs, y_int], axis=0)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Target std: {np.std(y):.4f}")
    print(f"Any NaNs in X: {np.isnan(X_scaled).any()}")
    print(f"Any NaNs in y: {np.isnan(y).any()}")

    # === Fit LassoCV để chọn top 10 predictors ===
    lasso = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=np.logspace(-2, 2, 100)).fit(X_scaled, y)

    # Choose top 10 features
    coef_series = pd.Series(np.abs(lasso.coef_), index=predictor_genes)
    top10_genes = coef_series.sort_values(ascending=False).head(10).index.tolist()

    return {
        "target_gene": target_gene,
        "top10_genes": top10_genes}

top10_predictors_dict = {}

# for i in range(2500, 3000):
#     print(i)
#     try:
#         result = get_task_data_and_top10(i, obs_data, int_data, gene_names)
#         top10_predictors_dict[result["target_gene"]] = result["top10_genes"]
#     except Exception as e:
#         print(f"Error at index {i} ({gene_names[i]}): {e}")