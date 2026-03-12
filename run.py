import time
import warnings
import numpy as np
import pandas as pd
from ICKDC import ICKDC
from data_loader import get_all_datasets
from sklearn.cluster import (
    KMeans, DBSCAN, MeanShift,
    AgglomerativeClustering, SpectralClustering
)
from sklearn.metrics import (
    adjusted_rand_score, adjusted_mutual_info_score,
    fowlkes_mallows_score, silhouette_score
)
warnings.filterwarnings("ignore")

def run_experiments(datasets):
    results = []


    # hyperparameter dictionary with all parameters for each algorithm and dataset -- from paper tables
    # K for KMeans/AGNES is extracted dynamically from the labels.
    hyperparams = {
        #synthetic datasets
        "Aggregation": {"gamma": 4.0, "eps": 0.06, "min_samples": 11, "bandwidth": 0.155},
        "R15": {"gamma": 2.1, "eps": 0.04, "min_samples": 14, "bandwidth": 0.090},
        "Toy": {"gamma": 3.5, "eps": 0.07, "min_samples": 20, "bandwidth": 0.135},
        "Spiral": {"gamma": 1.0, "eps": 0.10, "min_samples": 2, "bandwidth": 0.250},
        "Flame": {"gamma": 11.0, "eps": 0.10, "min_samples": 9, "bandwidth": 0.250},
        "Compound": {"gamma": 1.0, "eps": 0.05, "min_samples": 1, "bandwidth": 0.220},

        #real world datasets
        "Iris": {"gamma": 15.0, "eps": 0.12, "min_samples": 5, "bandwidth": 0.270},
        "Seeds": {"gamma": 2.5, "eps": 0.24, "min_samples": 16, "bandwidth": 0.460},
        "Wine": {"gamma": 11.0, "eps": 0.48, "min_samples": 18, "bandwidth": 0.665},
        "Ecoli": {"gamma": 2.0, "eps": 0.11, "min_samples": 4, "bandwidth": 0.335},
        "Wdbc": {"gamma": 1.0, "eps": 0.41, "min_samples": 22, "bandwidth": 1.200},
        "Dermatology": {"gamma": 3.5, "eps": 0.95, "min_samples": 4, "bandwidth": 1.487},
        "Ionosphere": {"gamma": 4.8, "eps": 0.78, "min_samples": 9, "bandwidth": 1.310},
        "Parkinsons": {"gamma": 2.7, "eps": 0.50, "min_samples": 17, "bandwidth": 0.620},
        "Segmentation": {"gamma": 7.4, "eps": 0.15, "min_samples": 2, "bandwidth": 0.638},

        #my own additions
        "Circles": {"gamma": 2.0, "eps": 0.10, "min_samples": 5, "bandwidth": None},
        "Digits": {"gamma": 5.0, "eps": 0.30, "min_samples": 5, "bandwidth": None}
    }

    print("\nStarting experiments... This might take a few minutes.\n")

    for dataset_name, (X, y_true) in datasets.items():
        print(f"Processing: {dataset_name}...")

        k_true = len(np.unique(y_true)) #extract K dynamically

        p = hyperparams.get(dataset_name) #get hyperparameters for this dataset

        algorithms = {
            "ICKDC (Yours)": ICKDC(gamma=p["gamma"]),
            "DBSCAN": DBSCAN(eps=p["eps"], min_samples=p["min_samples"]),
            "K-Means": KMeans(n_clusters=k_true, random_state=42),
            "MeanShift": MeanShift(bandwidth=p["bandwidth"]) if p["bandwidth"] else MeanShift(),
            "AGNES": AgglomerativeClustering(n_clusters=k_true),
            "Spectral": SpectralClustering(n_clusters=k_true, affinity='nearest_neighbors', random_state=42)
        }

        for algo_name, algorithm in algorithms.items():
            start_time = time.time()
            try:
                y_pred = algorithm.fit_predict(X) #fit and predict
                exec_time = time.time() - start_time

                #calculate metrics
                ari = adjusted_rand_score(y_true, y_pred)
                ami = adjusted_mutual_info_score(y_true, y_pred)
                fmi = fowlkes_mallows_score(y_true, y_pred)

                #Silhouette requires at least 2  clusters, and fewer than n_samples
                n_labels = len(np.unique(y_pred))
                if 1 < n_labels < len(X):
                    silhouette = silhouette_score(X, y_pred)
                else:
                    silhouette = np.nan

            except Exception as e:
                print(f"  [!] {algo_name} failed on {dataset_name}: {e}")
                exec_time, ari, ami, fmi, silhouette = (np.nan, np.nan, np.nan, np.nan, np.nan)

            results.append({
                "Dataset": dataset_name,
                "Algorithm": algo_name,
                "ARI": round(ari, 3), "AMI": round(ami, 3),
                "FMI": round(fmi, 3), "Silhouette": round(silhouette, 3),
                "Time (s)": round(exec_time, 4)
            })

    return pd.DataFrame(results)


if __name__ == "__main__":
    synthetic_data, real_data = get_all_datasets() #load the data into two separate dictionaries

    print("\n=============================================")
    print("   RUNNING SYNTHETIC DATASETS EXPERIMENTS")
    print("=============================================")
    synthetic_results_df = run_experiments(synthetic_data)

    print("\n=============================================")
    print("   RUNNING REAL-WORLD DATASETS EXPERIMENTS")
    print("=============================================")
    real_results_df = run_experiments(real_data)

    print("\n--- SYNTHETIC EXPERIMENT RESULTS ---")
    pd.set_option('display.max_rows', None)
    print(synthetic_results_df)
    synthetic_results_df.to_csv("results_synthetic.csv", index=False)

    print("\n--- REAL-WORLD EXPERIMENT RESULTS ---")
    print(real_results_df)
    real_results_df.to_csv("results_real.csv", index=False)
    print("done!")