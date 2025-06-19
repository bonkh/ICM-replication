import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
import gc

class DataProcessor:
    def __init__(self, data, task_division, target, categorical_features=None, numerical_features=None):
        self.data = data
        self.categorical_features = list(set(categorical_features) - set(task_division))
        self.numerical_features = list(set(numerical_features) - set([target]))
        print(self.numerical_features)
        print(self.categorical_features)
        print(task_division)

        self.target = target
        self.task_division = task_division
        self.tasks = self._task_division(task_division)
        
        # Process features for each task
        for task_name, task_data in self.tasks.items():
            if categorical_features:
                self.tasks[task_name] = self.process_categorical_features(task_data, self.categorical_features)
            if numerical_features:
                self.tasks[task_name] = self.process_numerical_features(task_data, self.numerical_features)
        
        # self.train_tasks, self.test_tasks = self.train_test_split()

    
    def __del__(self):
        """
        Destructor method called when object is about to be garbage collected.
        Performs explicit cleanup of large data structures.
        """
        try:
            self.cleanup()
            print("DataProcessor object cleaned up successfully")
        except Exception as e:
            print(f"Error during cleanup: {e}")
    
    def cleanup(self):
        """
        Explicit cleanup method to free memory and resources.
        Call this when you're done with the object.
        """
        # Clear large data structures
        if hasattr(self, 'data'):
            del self.data
        
        if hasattr(self, 'tasks'):
            for task_name in list(self.tasks.keys()):
                del self.tasks[task_name]
            del self.tasks
        
        if hasattr(self, 'train_tasks'):
            for task_name in list(self.train_tasks.keys()):
                del self.train_tasks[task_name]
            del self.train_tasks
        
        if hasattr(self, 'test_tasks'):
            for task_name in list(self.test_tasks.keys()):
                del self.test_tasks[task_name]
            del self.test_tasks
        
        # Clear other attributes
        self.categorical_features = None
        self.numerical_features = None
        self.target = None
        self.task_division = None
        
        # Force garbage collection
        gc.collect()
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup automatically"""
        self.cleanup()

    def get_memory_usage(self):
        """
        Get approximate memory usage of the object's data structures.
        """
        total_memory = 0
        
        if hasattr(self, 'data') and self.data is not None:
            total_memory += self.data.memory_usage(deep=True).sum()
        
        if hasattr(self, 'tasks') and self.tasks:
            for task_data in self.tasks.values():
                if task_data is not None:
                    total_memory += task_data.memory_usage(deep=True).sum()
        
        if hasattr(self, 'train_tasks') and self.train_tasks:
            for task_data in self.train_tasks.values():
                if task_data is not None:
                    total_memory += task_data.memory_usage(deep=True).sum()
        
        if hasattr(self, 'test_tasks') and self.test_tasks:
            for task_data in self.test_tasks.values():
                if task_data is not None:
                    total_memory += task_data.memory_usage(deep=True).sum()
        
        return total_memory / (1024**2)  # Return in MB
    
    def _task_division(self, task_division):
        # Implement task division logic here
        tasks = {}
        groups = self.data.groupby(task_division)
        for name, group in groups:
            tasks[name] = group.copy()
        return tasks 
    
    def process_na(self):
        self.data.dropna(inplace=True, how="any")

    def process_categorical_features(self, data, features):
        # Implement categorical feature processing logic here
        if features:
            data = pd.get_dummies(data, columns=features, drop_first=True)
        return data

    def process_numerical_features(self, data, features):
        # Implement numerical feature processing logic here
        if features:
            data[features] = data[features].apply(pd.to_numeric, errors='coerce')
            
            # Normalize numerical features using sklearn
            scaler = StandardScaler()
            data[features] = scaler.fit_transform(data[features])
        return data
    
    def train_test_split(self, test_split=0.6, random_state=42):
        """
        Split tasks into train and test sets based on task division values.
        The latter (biggest combination values) go to test set.
        
        Parameters:
        test_split: float, proportion of tasks to use for testing
        random_state: int, random seed for reproducibility
        
        Returns:
        train_tasks: dict, training tasks
        test_tasks: dict, testing tasks
        """
        # Get all task names (keys) and sort them
        # This ensures consistent ordering for the "latter" tasks
        task_names = list(self.tasks.keys())
        
        # Sort task names to get consistent ordering
        # For tuples (Year, Gender), this will sort lexicographically
        task_names_sorted = sorted(task_names)
        
        # Calculate split point
        n_tasks = len(task_names_sorted)
        n_test_tasks = max(1, int(n_tasks * test_split))  # Ensure at least 1 test task
        
        # Split: latter tasks go to test
        train_task_names = task_names_sorted[:-n_test_tasks]
        test_task_names = task_names_sorted[-n_test_tasks:]
        
        # Create train and test task dictionaries
        train_tasks = {name: self.tasks[name] for name in train_task_names}
        test_tasks = {name: self.tasks[name] for name in test_task_names}
        
        print(f"Train tasks: {train_task_names}")
        print(f"Test tasks: {test_task_names}")

        self.train_tasks = train_tasks
        self.test_tasks = test_tasks
        
        return train_tasks, test_tasks
    
    def get_xy_split(self, tasks_dict):
        """
        Helper function to extract X and y from tasks dictionary
        
        Parameters:
        tasks_dict: dict, dictionary of tasks
        
        Returns:
        X: pd.DataFrame, features
        y: pd.Series, target (if specified)
        """
        # Concatenate all task data
        all_data = pd.concat(tasks_dict.values(), ignore_index=True)
        
        if self.target:
            X = all_data.drop(columns=[self.target]).values 
            y = all_data[self.target].values.reshape(-1, 1)
            return X, y
        else:
            return all_data, None
        
    def plot_corr(self):
        import matplotlib.pyplot as plt 
        import seaborn as sns 

        sns.heatmap(self.data.corr(), annot=True)
        plt.show()

    def plot_corr_tasks(self):
        import matplotlib.pyplot as plt 
        import seaborn as sns 
        import math 

        # construct axes 
        ncols = 3 
        nrows = math.ceil(len(self.tasks)/ncols)

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
        axes = axes.flatten()

        task_division = self.task_division
        for i, task_info in enumerate(self.tasks.items()):
            sns.heatmap(pd.DataFrame(data=task_info[1].drop(columns=task_division)).corr(), ax=axes[i], annot=True)
            title = ' '.join(['_'.join([task_division[i], str(task_info[0][i])]) for i in range(len(task_division))])
            axes[i].set_title(title)

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.show()

    def plot_corr_tasks_stat(self):
        import matplotlib.pyplot as plt 
        import seaborn as sns 
        import math 

        columns = tuple(set(self.data.columns) - set(self.task_division))

        corr_map = {}
        for column in columns:
            corr_map[column] = np.zeros(len(self.tasks))

        task_division = self.task_division
        
        target = self.target 
        for i, task_info in enumerate(self.tasks.items()):
            corr = task_info[1].corr()
            for column in columns:
                print(column, target)

                try:
                    corr_map[column][i] = corr.loc[column, target]
                except:
                    corr_map[column][i] = 0

        ncols = 3 
        nrows = math.ceil(len(columns)/ncols)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*nrows, 4*ncols))
        axes = axes.flatten()
        
        titles = ['_'.join(f"{col}_{val}" for col, val in zip(task_division, task_key)) for task_key in self.tasks]    
        for i, corr_info in enumerate(corr_map.items()):
            axes[i].barh(titles, corr_info[1])
            axes[i].set_title(str(corr_info[0]))

        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.show()

        
        
        



        