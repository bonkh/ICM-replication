import pandas as pd
import json
import ast

# with open('data/lasso_10.txt', 'r') as f:
#     content = f.read()

# lasso_res_list = content.split('\n')
# id = lasso_res_list[0:-1:3]
# target_gene = lasso_res_list[1:-1:3]
# top10_predictors = lasso_res_list[2:-1:3]


# print(id[1])
# print(target_gene[1])
# print(top10_predictors[1])
# print(len(id))

# lasso_res = pd.DataFrame({
#     'id': id,
#     'target_gene': target_gene,
#     'top10_predictors': [list(ast.literal_eval(x)) for x in top10_predictors]
# })

# print(lasso_res.iloc[0])
# lasso_res.to_csv('data/lasso_10.csv', index=False)

# # Example: read JSON from a file
# with open("data/top10_predictors_per_gene_6000_6170.json") as f:
#     data_dict = json.load(f)

# # OR if you have a JSON string:
# # data_dict = json.loads(json_string)
# #
# # Convert the dict to a list of values (preserves order if you use Python 3.7+)
# top_10_lasso = [str(list(value)) for value in data_dict.values()]
# gene = list(data_dict.keys())

# print(top_10_lasso[-1])
# print(gene[0])
# print(len(top_10_lasso))
# new_df = pd.DataFrame({
#     'target_gene': gene,
#     'top10_predictors': top_10_lasso
# })

# print(new_df.shape)
# print(new_df.head())
# # If you already have a DataFrame:
# # df = pd.DataFrame(columns=data_dict.keys())  # create an empty one with correct columns
# # df.loc[len(df)] = new_row  # add the row

# df = pd.read_csv('data/lasso_10.csv', index_col=0)
# print(df.shape)
# df = pd.concat([df, new_df], axis=0)
# print(df.shape)
# print(df.tail())

# df.to_csv('data/lasso_10.csv')

df = pd.read_csv("data/lasso_10.csv", index_col=0)
df.to_csv("data/lasso_10.csv", index=False)
