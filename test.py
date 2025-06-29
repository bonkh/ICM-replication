import torch
import numpy as np
import gc
from plotting import *
import pickle

rc("text", usetex=False)

def plot_mse_results(pkl_path, save_dir="Experiment_7", save_name="mse_plot.pdf", title="Độ lỗi MSE giữa các can thiệp"):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    results = data["results"]
    print(results['pool'])
    print(results['pool'].shape)

    mean_mse = results["mean"].mean(axis=0)
    pool_mse = results["pool"].mean(axis=0)
    causal_mse = results["causal"].mean(axis=0)
    shat_mse = results["icm"].mean(axis=0)
    cicm_mse = results["cicm"].mean(axis=0)


    if mean_mse.shape[0] == 4:
        scenarios = ['Không can thiệp', 'Đặc trưng 3', 'Đặc trưng 3,4', "Đặc trưng 3,4,5"]
    else:
        scenarios = ['Intervene_none', 'Intervene_3', 'Intervene_3,4']

    x = np.arange(len(scenarios))
    width = 0.1

    plt.figure(figsize=(10, 6))
    plt.bar(x - 2*width, mean_mse, width, label='Mean')
    plt.bar(x - width, pool_mse, width, label='Pooling')
    plt.bar(x, causal_mse, width, label='Causal')
    plt.bar(x + width, shat_mse, width, label='ICM')
    plt.bar(x + 2 * width, cicm_mse, width, label='cICM')

    plt.ylabel("MSE")
    plt.xlabel("Đặc trưng bị can thiệp")
    plt.title(title)
    plt.xticks(x, scenarios)
    plt.legend()
    plt.tight_layout()


    os.makedirs(save_dir, exist_ok=True)

    # Lưu file
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, bbox_inches='tight', format='pdf', dpi=300)
    plt.close()

    print(f"Saved plot to: {save_path}")

plot_mse_results(
pkl_path="Experiment_7/mse_icm_vs_cicm_scen1.pkl",
save_dir="Experiment_7",
save_name="mse_plot_scen1.pdf",
title="Tình huống 1: Tập train và test có cùng những can thiệp với nhau"
)


plot_mse_results(
pkl_path="Experiment_7/mse_icm_vs_cicm_scen2.pkl",
save_dir="Experiment_7",
save_name="mse_plot_scen2.pdf",
title="Tình huống 2: Tập train và test có can thiệp khác nhau"
)
