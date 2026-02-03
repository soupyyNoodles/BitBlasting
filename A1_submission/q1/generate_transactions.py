#!/usr/bin/env python3
"""
Synthetic transactional dataset generator for Apriori vs FP-Growth runtime behavior.

Target qualitative behavior:
- Apriori: high runtime at 10%, 25%, 50%; sharp drop at 90%
- FP-Growth: consistently low runtime

Core idea:
- Inject a large correlated item block with ~60% support
- Add medium-frequency combinatorial noise
- Add rare items for realism
"""

# bash q1_2.sh 26 15000

import random
import argparse


def generate_dataset(universal_itemset: int, num_transactions: int, output_file: str):
    random.seed(42)

    # -----------------------------
    # Item universe
    # -----------------------------
    items = list(range(1, universal_itemset + 1))
    random.shuffle(items)

    # -----------------------------
    # Partition universe into MULTIPLE CORRELATED CLUSTERS
    # This is key: Apriori generates exponential candidates from correlated items
    # FP-Growth handles this efficiently with tree structure
    # -----------------------------
    # Create 4-5 clusters of highly correlated items (support 35-70%)
    num_clusters = max(4, universal_itemset // 15)  # e.g., 50 items -> ~4 clusters
    cluster_size = max(10, universal_itemset // (num_clusters + 2))  # ~12-13 items each
    
    clusters = []
    idx = 0
    for c in range(num_clusters):
        if idx + cluster_size <= len(items):
            clusters.append(items[idx:idx + cluster_size])
            idx += cluster_size
    
    # Remaining items are noise
    noise_items = items[idx:]

    transactions = []

    # -----------------------------
    # Transaction generation with STRONG correlations
    # -----------------------------
    for _ in range(num_transactions):
        T = set()

        # ---- CORRELATED CLUSTERS (critical for Apriori hardness) ----
        # Each cluster has varying support: 80%, 70%, 60%, 55%
        # This keeps many itemsets frequent at 10/25/50 thresholds
        cluster_supports = [0.80, 0.70, 0.60, 0.55]
        
        for c_idx, cluster in enumerate(clusters):
            support = cluster_supports[c_idx % len(cluster_supports)]
            if random.random() < support:
                # Add ALL items from cluster (creates dense patterns)
                T.update(cluster)
            else:
                # Partial membership (keeps support from dropping too low)
                for item in cluster:
                    if random.random() < 0.5:
                        T.add(item)

        # ---- INTER-CLUSTER CORRELATIONS ----
        # Create correlations between clusters to multiply candidate generation
        if len(clusters) >= 2 and random.random() < 0.7:
            c1, c2 = random.sample(range(len(clusters)), 2)
            # Add random samples from both clusters
            T.update(random.sample(clusters[c1], min(5, len(clusters[c1]))))
            T.update(random.sample(clusters[c2], min(5, len(clusters[c2]))))

        # ---- NOISE ITEMS (low frequency, adds realism) ----
        for item in random.sample(noise_items, min(6, len(noise_items))):
            if random.random() < 0.06:
                T.add(item)

        # ---- SAFETY: non-empty ----
        if not T:
            T.add(random.choice(clusters[0]))

        transactions.append(sorted(T))

    # -----------------------------
    # Write dataset
    # -----------------------------
    with open(output_file, "w") as f:
        for t in transactions:
            f.write(" ".join(map(str, t)) + "\n")

    # -----------------------------
    # Diagnostics (useful for report)
    # -----------------------------
    item_counts = {}
    for t in transactions:
        for i in t:
            item_counts[i] = item_counts.get(i, 0) + 1

    def pct(x): return 100 * x / num_transactions

    print(f"Dataset written to {output_file}")
    print(f"Transactions: {num_transactions}")
    print(f"Items: {universal_itemset}")
    print(f"Number of clusters: {len(clusters)}")
    print(f"Average transaction length: {sum(len(t) for t in transactions)/num_transactions:.2f}")

    freq_bins = {
        "<5%": 0, "5–10%": 0, "10–25%": 0,
        "25–50%": 0, "50–90%": 0, ">90%": 0
    }

    for c in item_counts.values():
        f = pct(c)
        if f < 5: freq_bins["<5%"] += 1
        elif f < 10: freq_bins["5–10%"] += 1
        elif f < 25: freq_bins["10–25%"] += 1
        elif f < 50: freq_bins["25–50%"] += 1
        elif f < 90: freq_bins["50–90%"] += 1
        else: freq_bins[">90%"] += 1

    print("Item frequency distribution:", freq_bins)


def main():
    parser = argparse.ArgumentParser(description="Generate transactional dataset")
    parser.add_argument("universal_itemset", type=int, help="Number of distinct items")
    parser.add_argument("num_transactions", type=int, help="Number of transactions")
    parser.add_argument(
        "--output", type=str, default="transactions.dat",
        help="Output file name"
    )

    args = parser.parse_args()
    generate_dataset(args.universal_itemset, args.num_transactions, args.output)


if __name__ == "__main__":
    main()
