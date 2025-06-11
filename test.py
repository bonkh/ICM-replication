import torch
import numpy as np
import gc
from plotting import *
import pickle

rc("text", usetex=False)

save_dir = "Experiment_7"
 
with open('Experiment_7/mse_2__100_1.0_1.0_0.5.pkl', "rb") as f:
    data = pickle.load(f)

# print(data["results"]['shat'])


results = data["results"]
mean_mse = results["mean"].mean(axis=0)   # shape (4,)
shat_mse = results["shat"].mean(axis=0)   # shape (4,)
cicm_mse = results["cicm"].mean(axis=0)

print(mean_mse.shape)

if mean_mse.shape[0] == 4:
    scenarios = ['Intervene_none', 'Intervene_3', 'Intervene_3,4', "Intervene_3,4,5"]
else:
    scenarios = ['Intervene_none', 'Intervene_3', 'Intervene_3,4']
x = np.arange(len(scenarios))
width = 0.25

plt.figure(figsize=(10, 6))
plt.bar(x - width, mean_mse, width, label='Mean')
plt.bar(x, shat_mse, width, label='Shat')
plt.bar(x + width, cicm_mse, width, label='CICM')

plt.ylabel("MSE")
plt.xlabel("Intervention")
plt.title("Comparison of MSE across Intervention")
plt.xticks(x, scenarios)
plt.legend()
plt.tight_layout()

plt.savefig(os.path.join(save_dir,'mse_plot_2.pdf'),
            bbox_inches='tight', format='pdf', dpi=300)
plt.close()
