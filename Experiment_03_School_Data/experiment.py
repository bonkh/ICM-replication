
import numpy as np
import gc

class Experiment:
    def __init__(self, data_processor, methods):
        self.data_processor = data_processor
        self.methods = methods  # List of models to train and evaluate
        # self.n_train_tasks = len(data_processor.train_tasks)
        # self.n_samples_per_train_tasks = tuple([len(data_processor.train_tasks[task]) for task in data_processor.train_tasks])
        # self.n_test_tasks = len(data_processor.test_tasks)
        self.results = {method.name: [] for method in methods} 
        self.all_results = []  # Store all results as a list of dicts
  

    def _run(self, annot):
        target = {}
        for i, (train_tasks, test_tasks, train_tasks_name, test_tasks_name) in enumerate(self.data_processor.leave_one_task_out_cv()):
            print("____________TRAIN LEAVE OUT ON ", i + 1, "____________")
            print(f"Train tasks {train_tasks_name}")
            print(f"Test tasks {test_tasks_name}")
            n_train_tasks = len(train_tasks)
            n_samples_per_train_tasks = tuple([len(train_tasks[task]) for task in train_tasks])
        
            for _, n_tasks in enumerate(np.arange(2, n_train_tasks + 1)):
                print(f"### TASK: {n_tasks} ###")
                params = {
                    'n_samples_per_task': n_samples_per_train_tasks[0:n_tasks],
                }

                selected_train_tasks = {k: v for k, v in list(train_tasks.items())[:n_tasks]}
                selected_test_tasks = {k: v for k, v in list(test_tasks.items())[:n_tasks]}

                X_train, y_train, columns = self.data_processor.get_xy_split(selected_train_tasks)
                X_test, y_test, _ = self.data_processor.get_xy_split(selected_test_tasks)            

                for method in self.methods:
                    print(f"Method name: {method.name}")
                    method.fit(X_train, y_train, params)
                    loss = method.evaluate(X_test, y_test)
                    self.results[method.name].append(loss)

                    result_entry = {
                        "leave_out_task": test_tasks_name[0],  # Assumes one left-out task
                        "n_tasks": n_tasks,
                        "method": method.name,
                        "loss": loss
                    }

                    try:
                        result_entry["selected_features"] = list(columns[method.selected_features])
                        print(f"Selected columns {columns[method.selected_features]}")
                    except Exception:
                        result_entry["selected_features"] = None
                        print("_____")
                    print(f'| Loss: {loss:.6f} |')
                    print("_____")

                    self.all_results.append(result_entry)

            print(f"EXPERIMENT RESULTS")
            for method in self.methods:
                mean_mse = np.mean(self.results[method.name])
                std_mse = np.std(self.results[method.name])
                print(f"  {method.name}: {mean_mse:.6f} ± {std_mse:.6f}") 

                self.results[method.name] = []

        # Save results to CSV
        self._save_results(annot)
        return self

    def _save_results(self, annot):
        import pandas as pd
        output_file = f"leave_one_out_city_china_test_on_season_{annot}.csv"
        df = pd.DataFrame(self.all_results)
        df.to_csv(output_file, index=False)
        print(f"\n✅ Results saved to {output_file}")


    def run_experiment(self, annot=None):
        self._run(annot)

        if annot is not None:
            print("#__________ RESULT FOR TASK: ", annot, "_____________#")
        print(f"Experiment result - Tasks: {self.data_processor.task_division}")

        for method in self.methods:
            mean_mse = np.mean(self.results[method.name])
            std_mse = np.std(self.results[method.name])
            print(f"  {method.name}: {mean_mse:.6f} ± {std_mse:.6f}") 