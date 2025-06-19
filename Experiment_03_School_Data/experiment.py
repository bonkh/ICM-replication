
import numpy as np
import gc

class Experiment:
    def __init__(self, data_processor, methods, n_repeats=10):
        self.data_processor = data_processor
        self.methods = methods  # List of models to train and evaluate
        self.n_repeats = n_repeats
        self.n_train_tasks = len(data_processor.train_tasks)
        self.n_samples_per_train_tasks = tuple([len(data_processor.train_tasks[task]) for task in data_processor.train_tasks])
        self.n_test_tasks = len(data_processor.test_tasks)
        self.results = {method.name: np.zeros((n_repeats, self.n_train_tasks)) for method in methods}   


    def _run(self):
        for repeat in range(self.n_repeats):
            print(f'Repeats {repeat}')
            for idx, n_tasks in enumerate(np.arange(2, self.n_train_tasks + 1)):
                print(f"Task: {n_tasks}")
                params = {
                    'n_ex': self.n_samples_per_train_tasks[0:n_tasks],
                }

                # Select a subset of training tasks
                selected_train_tasks = dict(list(self.data_processor.train_tasks.items())[:n_tasks])
                selected_test_tasks = dict(list(self.data_processor.test_tasks.items())[:n_tasks])
                
                # Get X, y for the selected tasks
                X_train, y_train = self.data_processor.get_xy_split(selected_train_tasks)
                X_test, y_test = self.data_processor.get_xy_split(selected_test_tasks)
                
                # Train each method and store results
                for method in self.methods:
                    method.fit(X_train, y_train, params)
                    self.results[method.name][repeat, idx] = method.evaluate(X_test, y_test)
                    print(f'{method.name}: {self.results[method.name][repeat, idx]:.3f}')

                # Clean up memory after each task iteration
                del X_train, y_train, X_test, y_test, selected_train_tasks, selected_test_tasks
                gc.collect()

        
        return self

    def run_experiment(self):
        self._run()
        print(f"Experiment result - Tasks: {self.data_processor.task_division}")

        for method in self.methods:
            mean_mse = np.mean(self.results[method.name])
            std_mse = np.std(self.results[method.name])
            print(f"  {method.name}: {mean_mse:.4f} ± {std_mse:.4f}")