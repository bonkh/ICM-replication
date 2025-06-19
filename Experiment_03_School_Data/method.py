
import numpy as np
from sklearn import linear_model

import os 
import sys 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import subset_search  # Assuming subset_search is a custom module for model fitting and evaluation
from utils import mse  # Assuming mse is a custom utility function for evaluation metrics

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
        
    def set_params(self, params):
        """
        Set parameters for the SGreedy method.
        This is a placeholder for actual parameter setting logic.
        """
        self.params = params
        return self

    
    def fit(self, X_train, y_train, params):
        # Implement fitting logic for SGreedy
        s_greedy = subset_search.greedy_subset(
            X_train, y_train, 
            delta=self.params['delta'],
            valid_split=self.params['valid_split'],
            n_ex=params['n_ex']
        )

        lr_greedy = linear_model.LinearRegression()
        lr_greedy.fit(X_train[:, s_greedy], y_train)

        self.model = lr_greedy  # Store the fitted model
        self.selected_features = s_greedy  # Store selected features
        return self

    def evaluate(self, X_test, y_test):
        # Implement evaluation logic for SGreedy
        return mse(self.model, X_test[:, self.selected_features], y_test)
    
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
    