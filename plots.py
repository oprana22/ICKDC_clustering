import matplotlib.pyplot as plt
import numpy as np
import warnings

#my custom modules
from ICKDC import ICKDC
from data_loader import get_all_datasets
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore")

def generate_scatter_plots():
    synthetic_data, _ = get_all_datasets() #load data

    #we only want to plot R15 and Circles
    datasets_to_plot = {
        "R15": synthetic_data["R15"],
        "Circles": synthetic_data["Circles"]
    }

    #specific gammas from our previous hyperparameter dictionary
    gammas = {"R15": 2.1, "Circles": 2.0}
    eps_vals = {"R15": 0.04, "Circles": 0.10}
    min_pts = {"R15": 14, "Circles": 5}

    #setup the figure (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Clustering Performance: Convex Clusters (R15) vs. Non-Linear Manifolds (Circles)", fontsize=16,
                 y=0.95)

    for row_idx, (dataset_name, (X, y_true)) in enumerate(datasets_to_plot.items()):
        ickdc = ICKDC(gamma=gammas[dataset_name])
        dbscan = DBSCAN(eps=eps_vals[dataset_name], min_samples=min_pts[dataset_name])

        #fit and predict
        print(f"Clustering {dataset_name} for plotting...")
        y_ickdc = ickdc.fit_predict(X)
        y_dbscan = dbscan.fit_predict(X)

        #plot 1: ground truth
        ax1 = axes[row_idx, 0]
        ax1.scatter(X[:, 0], X[:, 1], c=y_true, cmap='tab20', s=15)
        ax1.set_title(f"{dataset_name}: Ground Truth")
        ax1.set_xticks([])
        ax1.set_yticks([])

        #plot 2: ICKDC
        ax2 = axes[row_idx, 1]
        ax2.scatter(X[:, 0], X[:, 1], c=y_ickdc, cmap='tab20', s=15)
        ax2.set_title(f"{dataset_name}: ICKDC")
        ax2.set_xticks([])
        ax2.set_yticks([])

        #plot 3: DBSCAN
        ax3 = axes[row_idx, 2]
        ax3.scatter(X[:, 0], X[:, 1], c=y_dbscan, cmap='tab20', s=15)
        ax3.set_title(f"{dataset_name}: DBSCAN")
        ax3.set_xticks([])
        ax3.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)  # Make room for the main title
    plt.savefig("clustering_comparison.pdf", format='pdf', bbox_inches='tight')
    print("Plot successfully saved as 'clustering_comparison.pdf'")


if __name__ == "__main__":
    generate_scatter_plots()