# Things that need to be process:
# 1. Dvision of tasks (always categorical features)
# 2. Load and process of data 
# 3. Division of train and test data 
# 4. Model training and evaluation 

# TODO: Understand the MSE loss (And what s_greedy - alpha means?)
# TODO: Implement s_hat class too

import pandas as pd 
from data_processor import DataProcessor
from method import SGreedy, Pooling, Mean
from experiment import Experiment
        
# # Prepare data 
# train = pd.read_csv('dataset/train_values.csv', index_col=False)
# target = pd.read_csv('dataset/train_labels.csv', index_col=False)

data = pd.read_csv('dataset/proc_ILEA567.csv', index_col=False)
print(data.index)
categorical_features = ['Year', 'School', 'Gender', 'VRbandOfStudent', 'EthicGroup', 'SchoolGender', 'SchoolDenomination'] 
numerical_features = ['ExamScore', '%VR1band', '%FSM']
task_divison = ['Year']

with DataProcessor(data, 
                       task_division = task_divison,
                       categorical_features=categorical_features,
                       numerical_features=numerical_features,
                       target='ExamScore') as dataset:

    # train_tasks, test_tasks = dataset.train_test_split(test_split=0.4, random_state=42)

    # # Get X, y for training and testing
    # X_train, y_train = dataset.get_xy_split(train_tasks)
    # X_test, y_test = dataset.get_xy_split(test_tasks)

    # # Define methods to be used in the experiment
    # methods = [
    #     SGreedy(name='sgreedy').set_params({'delta': 0.01, 'valid_split': 0.6}),
    #     Pooling(name='pooling'),
    #     Mean(name='mean')
    # ]

    # # Run the experiments 
    # experiment = Experiment(dataset, methods, n_repeats=20)
    # experiment.run_experiment()

    dataset.plot_corr_tasks_stat()

# # dataset.cleanup()