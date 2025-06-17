import torch
import numpy as np
import gc
from plotting import *
import pickle

rc("text", usetex=False)

save_dir = "Experiment_7"
 
with open('Experiment_7/icm_vs_cicm.pkl', "rb") as f:
    data = pickle.load(f)
print(data)

count_subset = data['count_subset'] / float(data['n_repeat'])
count_icp = data['count_icp'] / float(data['n_repeat'])
inter = data["inter"]

width = 0.1
x_pos = np.arange(len(inter))  # 4 different intervention cases
num_features = count_subset.shape[1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5), sharey=True)

for ax, count, title in zip(
    [ax1, ax2],
    [count_subset, count_icp],
    ['Subset Search', 'ICP']
):
    for i in range(len(inter)):
        for p in range(num_features):
            x = i + (p - 3) * width * 1.5  # Horizontal shift by feature index

            # Highlight intervened features
            is_intervened = (p - 3 in inter[i])
            color = 'r' if is_intervened else 'b'

            ax.bar(x, count[i, p], width - 0.01,
                    color=color, alpha=0.6)

    ax.set_title(title, fontsize=16)
    ax.set_xticks(np.arange(len(inter)))
    ax.set_xticklabels(['No intervention', 'Intervene 3', 'Intervene 3,4', 'Intervene 3,4,5'], fontsize=14)
    ax.set_xlabel('Intervened covariates', fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

ax1.set_ylabel(r"Percentage of repetitions for which the"
                "\n"
                r"covariates are included", fontsize=14)

# Y ticks shared
ax1.set_yticks(np.arange(0, 1.1, 0.2))
ax1.set_yticklabels([r"$0$", r"$0.2$", r"$0.4$", r"$0.6$", r"$0.8$", r"$1$"], fontsize=12)

plt.tight_layout()
plt.savefig(os.path.join(save_dir,'scen_1_plot.pdf'),
            bbox_inches='tight', format='pdf', dpi=300)
plt.close()

# plot_interv('Experiment_7/tl_5_1.0_1.0_0.5.pkl')
