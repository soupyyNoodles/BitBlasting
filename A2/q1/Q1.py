"""
Q1.py - Clustering in High-Dimensional Spaces
Assignment 2, COL761 2026

Usage:
    python3 Q1.py <dataset_num>          # any numeric input -> loads both API datasets
    python3 Q1.py <path_to_dataset>.npy  # loads from .npy file

Outputs:
    - plot.png  : objective-value plot(s) with optimal k marked
    - stdout    : one optimal k for .npy input, or one line per API dataset
"""

import sys
import urllib.request
import json
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


K_MIN = 1
K_MAX = 15
N_INIT = 10
MAX_ITER = 300
RANDOM_STATE = 42


def load_from_api(dataset_num: int) -> np.ndarray:
    url = f"http://hulk.cse.iitd.ac.in:3000/dataset?student_id=cs1230041&dataset_num={dataset_num}"
    with urllib.request.urlopen(url) as response:
        raw_data = response.read().decode("utf-8")
        data = json.loads(raw_data)
    return np.array(data["X"])


def load_from_npy(path: str) -> np.ndarray:
    return np.load(path)


def compute_metrics(X: np.ndarray):
    ks = list(range(K_MIN, K_MAX + 1))
    inertias = []
    silhouettes = []

    for k in ks:
        km = KMeans(n_clusters=k, n_init=N_INIT, max_iter=MAX_ITER,
                    random_state=RANDOM_STATE)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        if k == 1:
            # Silhouette requires at least 2 clusters
            silhouettes.append(np.nan)
        else:
            silhouettes.append(silhouette_score(X, labels))

    return ks, inertias, silhouettes


def find_elbow(ks, inertias):
    inertias = np.array(inertias, dtype=float)
    # normalise so scale doesn't matter
    inertias_norm = (inertias - inertias.min()) / (inertias.max() - inertias.min() + 1e-12)
    first_diff = np.diff(inertias_norm, n=1)
    second_diff = np.diff(first_diff, n=1)
    ratio = second_diff / (np.abs(inertias_norm[1:-1]) + 1e-12)
    # ratio[i] corresponds to ks[i+1]
    elbow_idx = int(np.argmax(ratio)) + 1
    return ks[elbow_idx]


def find_optimal_k(ks, inertias, silhouettes):
    elbow_k = find_elbow(ks, inertias)
    return elbow_k


def make_plot(results, output_path="plot.png"):
    """
    results: list of tuples (dataset_title, ks, inertias, optimal_k)
    """
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5), squeeze=False)
    axes = axes[0]
    fig.suptitle("K-Means Objective Value vs Number of Clusters", fontsize=13)

    for ax, (dataset_title, ks, inertias, optimal_k) in zip(axes, results):
        ax.plot(ks, inertias, "b-o", markersize=5, linewidth=1.5, label="Objective value (WCSS)")
        ax.axvline(x=optimal_k, color="red", linestyle="--", linewidth=1.5,
                   label=f"Selected k = {optimal_k}")
        ax.set_xlabel("Number of Clusters (k)", fontsize=11)
        ax.set_ylabel("K-Means Objective Value (WCSS)", fontsize=11)
        ax.set_title(f"{dataset_title}: Objective Value vs k", fontsize=12)
        ax.set_xticks(ks)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def analyze_dataset(X: np.ndarray):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ks, inertias, silhouettes = compute_metrics(X_scaled)
    optimal_k = find_optimal_k(ks, inertias, silhouettes)
    return ks, inertias, optimal_k


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 Q1.py <dataset_num|path_to_dataset.npy>", file=sys.stderr)
        sys.exit(1)

    arg = sys.argv[1]

    # Numeric-input mode: evaluate and plot both assignment API datasets in one file.
    if arg.isdigit():
        dataset_ids = [1, 2]
        results = []
        for dataset_id in dataset_ids:
            X = load_from_api(dataset_id)
            ks, inertias, optimal_k = analyze_dataset(X)
            results.append((f"API Dataset {dataset_id}", ks, inertias, optimal_k))

        make_plot(results, output_path="plot.png")

        for dataset_id, _, _, optimal_k in results:
            print(f"dataset_{dataset_id} {optimal_k}")
        return

    # Single-dataset mode (kept for compatibility)
    if arg.endswith(".npy"):
        X = load_from_npy(arg)
        ks, inertias, optimal_k = analyze_dataset(X)
        dataset_name = arg.split("/")[-1]
        make_plot([(f"Custom NPY ({dataset_name})", ks, inertias, optimal_k)], output_path="plot.png")
        print(optimal_k)
        return

    X = load_from_npy(arg)
    ks, inertias, optimal_k = analyze_dataset(X)
    dataset_name = arg.split("/")[-1]
    make_plot([(f"Custom NPY ({dataset_name})", ks, inertias, optimal_k)], output_path="plot.png")
    print(optimal_k)


if __name__ == "__main__":
    main()
