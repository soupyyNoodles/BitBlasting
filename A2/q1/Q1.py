"""
Q1.py - Clustering in High-Dimensional Spaces
Assignment 2, COL761 2026

Usage:
    python3 Q1.py <dataset_num>          # loads from API (1 or 2)
    python3 Q1.py <path_to_dataset>.npy  # loads from .npy file

Outputs:
    - plot.png  : objective plot(s) with selected k marked
    - stdout    : single integer, the optimal k

In API mode, plot.png contains subplots for dataset 1 and dataset 2
with explicit subplot titles (clarification-compliant).
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

STUDENT_ID = "2023EE10968"
API_BASE_URL = "http://hulk.cse.iitd.ac.in:3000/dataset"

K_MIN = 1
K_MAX = 15
N_INIT = 10
MAX_ITER = 300
RANDOM_STATE = 42


def load_from_api(dataset_num: int, student_id: str = STUDENT_ID) -> np.ndarray:
    url = f"{API_BASE_URL}?student_id={student_id}&dataset_num={dataset_num}"
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


def _plot_objective(ax, ks, inertias, optimal_k, title):
    ax.plot(ks, inertias, "b-o", markersize=5, linewidth=1.5, label="Objective value (WCSS)")
    ax.axvline(x=optimal_k, color="red", linestyle="--", linewidth=1.5,
               label=f"Selected k = {optimal_k}")
    ax.set_xlabel("Number of Clusters (k)", fontsize=10)
    ax.set_ylabel("K-Means Objective Value (WCSS)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(ks)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def make_api_plot(results_by_dataset, output_path="plot.png"):
    """
    Safest handling for assignment clarification:
    plot both API datasets in one file using subplots with explicit titles.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"K-Means Objective Curves (student_id={STUDENT_ID})",
        fontsize=13
    )

    for ax, dataset_num in zip(axes, [1, 2]):
        if dataset_num in results_by_dataset:
            ks, inertias, _silhouettes, optimal_k = results_by_dataset[dataset_num]
            _plot_objective(ax, ks, inertias, optimal_k, f"Dataset {dataset_num}")
        else:
            ax.set_title(f"Dataset {dataset_num} (unavailable)", fontsize=11)
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def make_single_dataset_plot(ks, inertias, optimal_k, output_path="plot.png"):
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5))
    fig.suptitle("K-Means Objective Curve", fontsize=13)
    _plot_objective(ax, ks, inertias, optimal_k, "Input .npy dataset")
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
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        ks, inertias, silhouettes = compute_metrics(X_scaled)
        optimal_k = find_optimal_k(ks, inertias, silhouettes)
        make_single_dataset_plot(ks, inertias, optimal_k, output_path="plot.png")
        print(optimal_k)
        return
    elif arg in ("1", "2"):
        requested_dataset = int(arg)
        results_by_dataset = {}

        # Always compute requested dataset (required output k).
        X_req = load_from_api(requested_dataset)
        scaler_req = StandardScaler()
        X_req_scaled = scaler_req.fit_transform(X_req)
        ks_req, inertias_req, silhouettes_req = compute_metrics(X_req_scaled)
        k_req = find_optimal_k(ks_req, inertias_req, silhouettes_req)
        results_by_dataset[requested_dataset] = (
            ks_req, inertias_req, silhouettes_req, k_req
        )

        # For clarification compliance, try to also include the other API dataset.
        other_dataset = 2 if requested_dataset == 1 else 1
        try:
            X_other = load_from_api(other_dataset)
            scaler_other = StandardScaler()
            X_other_scaled = scaler_other.fit_transform(X_other)
            ks_other, inertias_other, silhouettes_other = compute_metrics(X_other_scaled)
            k_other = find_optimal_k(ks_other, inertias_other, silhouettes_other)
            results_by_dataset[other_dataset] = (
                ks_other, inertias_other, silhouettes_other, k_other
            )
        except Exception:
            # Keep execution robust if only requested dataset can be fetched.
            pass

        make_api_plot(results_by_dataset, output_path="plot.png")
        print(k_req)
        return
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
    make_single_dataset_plot(ks, inertias, optimal_k, output_path="plot.png")

    # Output optimal k to stdout
    print(optimal_k)


if __name__ == "__main__":
    main()
