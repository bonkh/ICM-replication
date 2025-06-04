import matplotlib.pyplot as plt
import os
import numpy as np
import pickle
import seaborn as sns
import pandas as pd

save_dir = "Experiment_01_top_left"
file_name = "tl_norm__100_2.0_1.0_0.5.pkl"
# file_name = 'tl_norm__100_2.0_1.0_0.5_02.pkl'
# file_name = 'tl_norm__100_2.0_1.0_0.5_03.pkl'

# Full path to the file
file_path = os.path.join(save_dir, file_name)

# Open and load the data
with open(file_path, "rb") as f:
    loaded_data = pickle.load(f)

# Access the components
results = loaded_data["results"]
methods, color_dict, legends, markers = loaded_data["plotting"]
n_train_tasks = loaded_data["n_train_tasks"]
n_repeat = 100


# Loop through each method
for method in methods:
    plt.figure(figsize=(10, 6))

    # Build long-format DataFrame
    df_long = pd.DataFrame()
    for task_idx, n_task in enumerate(n_train_tasks):
        temp_df = pd.DataFrame(
            {
                "Run_ID": np.arange(n_repeat),
                "Loss": results[method][:, task_idx],
                "n_train_tasks": str(n_task),  # Convert to string for legend clarity
            }
        )
        df_long = pd.concat([df_long, temp_df], ignore_index=True)

    # Plot using seaborn lineplot
    sns.lineplot(
        data=df_long,
        x="Run_ID",
        y="Loss",
        hue="n_train_tasks",
        palette="tab10",
        marker="o",
    )

    plt.title(f"Loss across Runs ({method})")
    plt.xlabel("Run ID")
    plt.ylabel("Loss (MSE)")
    plt.legend(title="n_train_tasks")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
