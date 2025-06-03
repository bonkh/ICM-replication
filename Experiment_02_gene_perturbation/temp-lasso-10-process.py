import pandas as pd
import ast

with open('data/lasso_10.txt', 'r') as f:
    content = f.read()

lasso_res_list = content.split('\n')
id = lasso_res_list[0:-1:4]
target_gene = lasso_res_list[2:-1:4]
top10_predictors = lasso_res_list[3:-1:4]
print(id[0])
print(target_gene[0])
print(top10_predictors[0])  

lasso_res = pd.DataFrame({
    'id': id,
    'target_gene': target_gene,
    'top10_predictors': [list(ast.literal_eval(x)) for x in top10_predictors]
})

print(lasso_res.iloc[0])
lasso_res.to_csv('data/lasso_10.csv', index=False)