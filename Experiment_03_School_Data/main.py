# Things that need to be process:
# 1. Dvision of tasks (always categorical features)
# 2. Load and process of data 
# 3. Division of train and test data 
# 4. Model training and evaluation 

# TODO: Understand the MSE loss (And what s_greedy - alpha means?)
# TODO: Implement s_hat class too

import pandas as pd 
import numpy as np
from data_processor import DataProcessor
from method import SGreedy, Pooling, Mean, SHat, CLF_Pool, Mode
from experiment import Experiment
import matplotlib.pyplot as plt 
np.random.seed(1234)

# # ***** For Poverty dataset 
# features = pd.read_csv('dataset/train_values.csv', index_col=False)
# target = pd.read_csv('dataset/train_labels.csv', index_col=False)

# print("TARGET", target)

# data = pd.concat([features, target], axis=1)
# print("DATA", data.head())
# data.drop(columns=['row_id'], inplace=True)
# categorical_cols = [
#     'country', 'is_urban', 'female', 'married',
#     'religion', 'relationship_to_hh_head', 'education_level', 
#     'literacy', 'can_add', 'can_divide', 'can_calc_percents', 
#     'can_calc_compounding', 'employed_last_year', 'employment_category_last_year', 
#     'employment_type_last_year', 'income_ag_livestock_last_year', 
#     'income_friends_family_last_year', 'income_government_last_year', 
#     'income_own_business_last_year', 'income_private_sector_last_year',  
#     'income_public_sector_last_year', 
#     'formal_savings', 'informal_savings', 'cash_property_savings', 
#     'has_insurance', 'has_investment', 'bank_interest_rate', 
#     'mm_interest_rate', 'mfi_interest_rate', 'other_fsp_interest_rate', 
#     'num_shocks_last_year', 
#     'borrowed_for_emergency_last_year', 'borrowed_for_daily_expenses_last_year', 
#     'borrowed_for_home_or_biz_last_year', 'phone_technology', 'can_call', 
#     'can_text', 'can_use_internet', 'can_make_transaction', 'advanced_phone_use', 
#     'reg_bank_acct', 'reg_mm_acct', 'reg_formal_nbfi_account', 'financially_included', 
#     'active_bank_user', 'active_mm_user', 'active_formal_nbfi_user', 'active_informal_nbfi_user',
#     'nonreg_active_mm_user', 
# ]
# numerical_cols = [
#     'age', 'share_hh_income_provided', 'num_times_borrowed_last_year', 
#     'borrowing_recency', 'phone_ownership', 'num_formal_institutions_last_year', 
#     'num_informal_institutions_last_year', 'num_financial_activities_last_year', 
#     'avg_shock_strength_last_year'
# ]

# task_division = ['country']
# target = 'poverty_probability'

# with DataProcessor(data=data, task_division=task_division, target=target, 
#                    categorical_features=categorical_cols, 
#                    numerical_features=numerical_cols, fill_na=True) as dataset:
    
#     # dataset.plot_corr()
#     # dataset.plot_corr_tasks()
#     # dataset.plot_corr_tasks_stat()
#     dataset.train_test_split(test_split=0.5)

#     pooling = Pooling()
#     mean = Mean()
#     sgreedy = SGreedy().set_params(params={'use_hsic':False})
#     shat = SHat().set_params(params={'use_hsic':False})

#     methods = [pooling, mean, sgreedy, shat]
#     experiment = Experiment(dataset, methods)
#     experiment.run_experiment() 



# # ***** For School data (2016) *** (To fix classification)
# data = pd.read_csv('CollegeDistance.csv', index_col=False)
# def splitSchoolEnv(x):
#     if x< 1:
#         return 0 
#     elif x<2: 
#         return 1 
#     return 2

    
#     if x<= 0.5:
#         return 0
#     elif x<= 1: 
#         return 1
#     elif x<=2:
#         return 2 
#     elif x <= 3:
#         return 3
#     elif x <= 4:
#         return 4 
#     elif x <= 10: 
#         return 5
#     return 6

# def splitSchoolTarget(x):
#     if x >= 16:
#         return 1 
#     return 0

# data['env'] = data['dist'].apply(splitSchoolEnv)
# # print('MEDIAN DISTANCE' , data['dist'].median())


# data['target'] = data['ed'].apply(splitSchoolTarget)
# data.drop(columns=['dist', 'ed'], inplace=True)
# categorical_features = [
#     'female', 'black', 'hispanic', 'dadcoll', 'momcoll',
#     'incomehi', 'ownhome', 'urban'
# ]

# print(data.head())


# numerical_features = ['bytest','cue80', 'stwmfg80', 'tuition']


# with DataProcessor(data=data, task_division=['env'], target='target', categorical_features=categorical_features, numerical_features=numerical_features) as dataset:
#     dataset.plot_corr()
#     # dataset.plot_corr_tasks_stat()
#     # dataset.plot_corr_tasks()
#     dataset.train_test_split()

#     pooling = CLF_Pool()
#     mean = Mode()
#     sgreedy = SGreedy().set_params({'is_classification_task': True})
#     shat = SHat().set_params({'is_classification_task': True})

#     methods = [pooling, mean, sgreedy, shat]
#     experiment = Experiment(dataset, methods)
#     experiment.run_experiment() 
 
# # ***** For Air Quality data (2016) *** (To fix correlation)
data = pd.read_csv('china_air_quality_2.csv', index_col=False).rename(columns={'Unnamed: 0': 'Time'})
data = data.drop(columns=['Time'])
numerical_features = ['DewPt', 'Humidity', 'Press', 'WindSp', 'PrecipHr', 'PrecipCm', 'Season']

task_season_data = data.groupby('env')
# print("Testing with Season as problem")
for name, task_data in task_season_data:
    print("******* TASK CITY : ", name, " **********")
    task_data = task_data.reset_index().drop(columns=['env', 'index'])
    print(task_data)
    with DataProcessor(data=task_data, task_division=['Season'], target='target', numerical_features=numerical_features) as dataset:
        # dataset.plot_corr()
        # dataset.plot_corr_tasks_stat()
        # dataset.plot_corr_tasks()
        dataset.train_test_split()

        pooling = Pooling()
        mean = Mean()
        sgreedy = SGreedy().set_params()
        shat = SHat().set_params()

        methods = [pooling, mean, sgreedy, shat]
        experiment = Experiment(dataset, methods)
        experiment.run_experiment(f"CITY {name}") 
 