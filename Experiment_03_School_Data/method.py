
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score

import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import subset_search as subset_search  # Assuming subset_search is a custom module for model fitting and evaluation
from utils import mse, lasso_alpha_search_synt  # Assuming mse is a custom utility function for evaluation metrics

class Method:
    def __init__(self, name):
        self.name = name
    def  fit(self, X_train, y_train, params):
        """
        Fit the model to the training data.
        This is a placeholder for actual fitting logic.
        """
        pass

    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on the given data.
        This is a placeholder for actual evaluation logic.
        """
        # Return a dummy score for demonstration purposes
        return np.random.rand()
    
class SGreedy(Method):
    def __init__(self, name='sgreedy'):
        super().__init__(name)
        self.params = None
        
    def set_params(self, params=None):
        default_params = {
            'delta': 0.01,
            'valid_split': 0.6,
            'is_classification_task': False
        }

        # Merge with default params
        if params is not None:
            merged_params = {**default_params, **params}
        else:
            merged_params = default_params

        self.params = merged_params
        return self

    
    def fit(self, X_train, y_train, params):
        # Implement fitting logic for SGreedy
        # s_greedy = subset_search.greedy_search(
        #     X_train, y_train, 
        #     alpha=self.params['alpha'],
        #     valid_split=self.params['valid_split'],
        #     n_samples_per_task=params['n_samples_per_task'], 
        #     is_classification_task=self.params['is_classification_task']
        # )

        s_greedy = subset_search.greedy_subset(
            X_train, y_train, 
            delta=self.params['delta'],
            valid_split=self.params['valid_split'],
            n_ex=params['n_samples_per_task']
            # is_classification_task=self.params['is_classification_task']
        )


        if(self.params['is_classification_task']):
            model = linear_model.LogisticRegression()
            model.fit(X_train[:, s_greedy], y_train)
        else:
            model = linear_model.LinearRegression()
            model.fit(X_train[:, s_greedy], y_train)

        # print("Model coefficient", model.coef_)
        self.model = model  # Store the fitted model
        self.selected_features = s_greedy  # Store selected features
        return self

    def evaluate(self, X_test, y_test):
        if(self.params['is_classification_task']):
            return accuracy_score(y_test, self.model.predict(X_test[:, self.selected_features]))
        else:
            return mse(self.model, X_test[:, self.selected_features], y_test)


class SHat(Method): 
    def __init__(self, name='shat'):
        super().__init__(name)
        self.params = None 
        self.selected_features = None
        
    def set_params(self, params=None):
        default_params = {
            'delta': 0.01,
            'valid_split': 0.6,
            'use_hsic': False, 
            'is_classification_task': False
        }

        # Merge with default params
        if params is not None:
            merged_params = {**default_params, **params}
        else: 
            merged_params = default_params

        self.params = merged_params
        return self

    
    def fit(self, X_train, y_train, params):
        # Implement fitting logic for SGreedy
        self.lasso_mask = None
        if (len(X_train[1]) < 13): 
            print('---No Lasso ---')
            s_hat = subset_search.subset(
                X_train, y_train, 
                delta=self.params['delta'],
                valid_split=self.params['valid_split'],
                n_samples_per_task_list=params['n_samples_per_task'], 
                use_hsic=self.params['use_hsic']
                # is_classification_task = self.params['is_classification_task']
            )
        else:
            print("--Use Lasso--")
            lasso_mask = lasso_alpha_search_synt(X_train, y_train)
            self.lasso_mask = lasso_mask

            X_train = X_train[:, lasso_mask]
            
            s_hat = subset_search.full_search(
                X_train, y_train, 
                alpha=self.params['alpha'],
                valid_split=self.params['valid_split'],
                n_samples_per_task_list=params['n_samples_per_task'], 
                use_hsic=self.params['use_hsic'],
                is_classification_task = self.params['is_classification_task']
            )


        if(len(s_hat) != 0):
            if(self.params['is_classification_task']):
                model = linear_model.LogisticRegression()
                model.fit(X_train[:, s_hat], y_train)
            else:
                model = linear_model.LinearRegression()
                model.fit(X_train[:, s_hat], y_train)
            
            self.model = model  # Store the fitted model
            # print(model.coef_)
            self.selected_features = s_hat  # Store selected features
        else:
            self.selected_features = None
            from scipy import stats
            if(self.params['is_classification_task']):
                self.mode = stats.mode(y_train)[0]
            else:
                self.mean = np.mean(y_train)

    def evaluate(self, X_test, y_test):
        # Implement evaluation logic for SGreedy
        if self.lasso_mask is not None:
            X_test = X_test[:, self.lasso_mask]

        if(self.params['is_classification_task']):
            if self.selected_features is not None:
                print("Evaluate selected features")
                return accuracy_score(y_test, self.model.predict(X_test[:, self.selected_features]))
            else:
                return accuracy_score(y_test, np.full_like(y_test, self.mode))
        else:
            if self.selected_features is not None:
                return mse(self.model, X_test[:, self.selected_features], y_test)
            else:
                return np.mean((self.mean - y_test)**2)
                # return np.mean((y_test, self.model.predict(X_test[:,  self.selected_features]))**2)
    
class Pooling(Method):
    def __init__(self, name='pooling'):
        super().__init__(name)
    
    def fit(self, X_train, y_train, params):
        # Implement fitting logic for Pooling
        # This is a placeholder for actual fitting logic
        self.model = linear_model.LinearRegression()
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test):
        # Implement evaluation logic for Pooling
        return mse(self.model, X_test, y_test)
    
class Mean(Method):
    def __init__(self, name='mean'):
        super().__init__(name)
    
    def fit(self, X_train, y_train, params):
        # Implement fitting logic for Mean
        self.mean = np.mean(y_train)
        return self

    def evaluate(self, X_test, y_test):
        # Implement evaluation logic for Mean
        # return mse(np.full_like(y_test, self.mean), y_test)
        return np.mean((self.mean - y_test)**2)
    
class CLF_Pool(Method):
    def __init__(self, name='pool'):
        super().__init__(name)

    def fit(self, X_train, y_train, params):
        self.model = linear_model.LogisticRegression()
        self.model.fit(X_train, y_train)
        return self

    def evaluate(self, X_test, y_test):
        # return super().evaluate(X_test, y_test)
        return accuracy_score(y_test > 0.5, self.model.predict(X_test))

class Mode(Method):
    def __init__(self, name='mean'):
        super().__init__(name)

    def fit(self, X_train, y_train, params):
        from scipy import stats
        self.mode = stats.mode(y_train, keepdims=True)[0]

    def evaluate(self, X_test, y_test):
        return accuracy_score(y_test > 0.5, np.full_like(y_test, self.mode))