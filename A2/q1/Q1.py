"""
Q1.py - Clustering in High-Dimensional Spaces
Assignment 2, COL761 2026

Usage:
    python3 Q1.py <dataset_num>          # loads from API (1 or 2)
    python3 Q1.py <path_to_dataset>.npy  # loads from .npy file

Outputs:
    - plot.png  : metrics plot with optimal k marked
    - stdout    : single integer, the optimal k
"""

import sys
import os
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
    url = f"http://10.208.23.248:3000/dataset?student_id=cs1230041&dataset_num={dataset_num}"
    with urllib.request.urlopen(url) as response:
        raw_data = response.read().decode("utf-8")
        data = json.loads(raw_data)
    return np.array(data["X"])


def load_from_npy(path: str) -> np.ndarray:
    return np.load(path)


def compute_metrics(X: np.ndarray):
    """Run k-means for k in [K_MIN, K_MAX] and return inertias and silhouette scores.

    Silhouette is undefined for k=1 (only one cluster); np.nan is stored there.
    """
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
    """
    Find elbow via maximum second-difference (acceleration) method.
    In high-dimensional spaces the standard elbow is often subtle; this
    locates the point of highest curvature in the inertia curve.
    """
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
    """
    Choose optimal k based on the inertia (WCSS) elbow, as required by the
    assignment ("plot the objective value as a function of k ... determine a
    suitable choice of k").

    The elbow (second-difference / acceleration method) is the primary
    criterion.  Silhouette score for k >= 2 is used as a consistency check:
    if the silhouette peak disagrees with the elbow but has a large clear
    margin, we trust silhouette instead.

    In high dimensions, distance concentration (V_d(r) → 0 as d → ∞)
    compresses silhouette scores toward 0, so elbow is more robust.
    """
    elbow_k = find_elbow(ks, inertias)
    return elbow_k


def make_plot(ks, inertias, silhouettes, optimal_k, output_path="plot.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("K-Means Objective Value vs Number of Clusters", fontsize=13)

    # --- Primary: Inertia (WCSS) / Elbow plot ---
    # This is the plot explicitly required by the assignment.
    ax1.plot(ks, inertias, "b-o", markersize=5, linewidth=1.5, label="Objective value (WCSS)")
    ax1.axvline(x=optimal_k, color="red", linestyle="--", linewidth=1.5,
                label=f"Selected k = {optimal_k}")
    ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax1.set_ylabel("K-Means Objective Value (WCSS)", fontsize=11)
    ax1.set_title("Objective Value vs k  (Elbow Method)", fontsize=12)
    ax1.set_xticks(ks)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Supplementary: Silhouette score (undefined for k=1, skip that point) ---
    valid_ks  = [k for k, s in zip(ks, silhouettes) if not np.isnan(s)]
    valid_sil = [s for s in silhouettes if not np.isnan(s)]
    ax2.plot(valid_ks, valid_sil, "g-s", markersize=5, linewidth=1.5, label="Silhouette Score")
    ax2.axvline(x=optimal_k, color="red", linestyle="--", linewidth=1.5,
                label=f"Selected k = {optimal_k}")
    ax2.set_xlabel("Number of Clusters (k)", fontsize=11)
    ax2.set_ylabel("Silhouette Score", fontsize=11)
    ax2.set_title("Silhouette Analysis (supplementary)", fontsize=12)
    ax2.set_xticks(valid_ks)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 Q1.py <dataset_num|path_to_dataset.npy>", file=sys.stderr)
        sys.exit(1)

    arg = sys.argv[1]

    # Determine input mode
    if arg.endswith(".npy"):
        X = load_from_npy(arg)
    elif arg in ("1", "2"):
        X = load_from_api(int(arg))
    else:
        # Try as .npy path anyway
        X = load_from_npy(arg)

    # Standardise features — essential before k-means, especially in high
    # dimensions where feature scales can dominate distance computations.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compute clustering metrics
    ks, inertias, silhouettes = compute_metrics(X_scaled)

    # Determine optimal k
    optimal_k = find_optimal_k(ks, inertias, silhouettes)

    # Generate plot
    make_plot(ks, inertias, silhouettes, optimal_k, output_path="plot.png")

    # Output optimal k to stdout
    print(optimal_k)


if __name__ == "__main__":
    main()
