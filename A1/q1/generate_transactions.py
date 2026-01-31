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
    # Partition universe
    # -----------------------------
    core_size = max(12, int(0.15 * universal_itemset))      # drives Apriori explosion
    medium_size = int(0.35 * universal_itemset)
    rare_size = universal_itemset - core_size - medium_size

    core_items = items[:core_size]
    medium_items = items[core_size:core_size + medium_size]
    rare_items = items[core_size + medium_size:]

    # -----------------------------
    # Medium item groups (correlated noise)
    # -----------------------------
    random.shuffle(medium_items)
    medium_groups = [
        medium_items[i:i + 5]
        for i in range(0, len(medium_items), 5)
        if len(medium_items[i:i + 5]) == 5
    ]

    transactions = []

    # -----------------------------
    # Transaction generation
    # -----------------------------
    for _ in range(num_transactions):
        T = set()

        # ---- CORE BLOCK (critical) ----
        # Survives 10/25/50%, dies at 90%
        if random.random() < 0.7:
            T.update(core_items)
        else:
            # partial core inclusion
            for item in core_items:
                if random.random() < 0.2:
                    T.add(item)

        # ---- MEDIUM ITEMS ----
        # Adds combinatorial candidates
        if random.random() < 0.5 and medium_groups:
            group = random.choice(medium_groups)
            k = random.randint(2, 4)
            T.update(random.sample(group, k))
        else:
            for item in random.sample(medium_items, min(3, len(medium_items))):
                if random.random() < 0.3:
                    T.add(item)

        # ---- RARE ITEMS ----
        if rare_items and random.random() < 0.1:
            T.add(random.choice(rare_items))

        # ---- SAFETY: non-empty ----
        if not T:
            T.add(random.choice(core_items))

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
    print(f"Core block size: {core_size}")
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
