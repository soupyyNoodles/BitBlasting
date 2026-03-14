"""
Evaluation Script — Compute σ(R) for a Given Set of Blocked Routes
===================================================================
Assignment 2, Q2 — COL761 2026

Reads:
  1. Graph file       : edge list  (u  v  p_uv  per line)
  2. Seed file        : initially burning nodes A0  (one node ID per line)
  3. Blocked file     : output of forest_fire.py / forest_fire_greedy.py
                        (one blocked edge  u  v  per line)
  4. k               : budget (maximum number of blocked edges allowed)
  5. num_sim          : number of Monte-Carlo simulation instances

Optional:
  --hops INT          : limit fire spread to this many hops from the seed set.
                        Pass -1 (or omit) for unlimited spread (default behaviour).

Budget rules:
  - Invalid edges (not in graph) and duplicates are always rejected with [ERROR].
  - From the remaining valid unique edges (preserved in file order):
      * count > k  : [WARNING] — only the FIRST k valid edges are used.
      * count == k : [OK] — normal evaluation.
      * count < k  : [OK] now, [WARNING] printed alongside final results.

Usage:
    python evaluate.py --graph_file <graph_file> --seed_file <seed_file> \
                       --blocked_file <blocked_file> --k <k> --num_sim <num_sim> \
                       [--hops H]

Example:
    python evaluate.py --graph_file graph.txt --seed_file seeds.txt \
                       --blocked_file output.txt --k 10 --num_sim 1000
    python evaluate.py --graph_file graph.txt --seed_file seeds.txt \
                       --blocked_file output.txt --k 10 --num_sim 1000 --hops 3
    python evaluate.py --graph_file graph.txt --seed_file seeds.txt \
                       --blocked_file output.txt --k 10 --num_sim 1000 --hops -1
"""

import random
import argparse
from collections import defaultdict


# ─────────────────────────────────────────────────────────────────────────────
# 1. Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_graph(path):
    nodes    = set()
    adj      = defaultdict(list)
    edge_set = set()
    with open(path, 'r') as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 3:
                print(f"[WARN] graph line {lineno} has fewer than 3 fields — skipping: {line!r}")
                continue
            try:
                u, v, p = int(parts[0]), int(parts[1]), float(parts[2])
            except ValueError:
                print(f"[WARN] graph line {lineno} could not be parsed — skipping: {line!r}")
                continue
            if not (0.0 < p <= 1.0):
                print(f"[WARN] graph line {lineno}: p={p} outside (0,1] for edge ({u},{v})")
            nodes.add(u); nodes.add(v)
            adj[u].append((v, p))
            edge_set.add((u, v))
    return nodes, dict(adj), edge_set


def load_seeds(path):
    seeds = set()
    with open(path, 'r') as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                seeds.add(int(line))
            except ValueError:
                print(f"[WARN] seed file line {lineno} is not an integer — skipping: {line!r}")
    if not seeds:
        raise ValueError(f"Seed file '{path}' yielded no valid node IDs.")
    return frozenset(seeds)


def load_blocked(path, edge_set, k):
    """
    Parse and validate blocked edges, applying budget k.

    - Edges not in graph  → [ERROR], rejected, not counted toward budget.
    - Duplicate edges     → [ERROR], rejected, not counted toward budget.
    - If valid count > k  → [WARNING], only first k (file order) are used.
    - If valid count < k  → noted here, [WARNING] printed with final results.

    Returns: (blocked_frozenset, n_valid_total, over_budget, under_budget)
    """
    valid_ordered   = []
    seen            = set()
    invalid_count   = 0
    duplicate_count = 0

    with open(path, 'r') as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) < 2:
                print(f"[WARN] blocked line {lineno}: fewer than 2 fields — skipping: {line!r}")
                continue
            try:
                u, v = int(parts[0]), int(parts[1])
            except ValueError:
                print(f"[WARN] blocked line {lineno}: could not parse — skipping: {line!r}")
                continue

            edge = (u, v)
            if edge not in edge_set:
                invalid_count += 1
                print(f"[ERROR] Line {lineno}: edge {edge} does NOT exist in the graph — rejected.")
                continue
            if edge in seen:
                duplicate_count += 1
                print(f"[ERROR] Line {lineno}: edge {edge} is a duplicate — rejected.")
                continue
            seen.add(edge)
            valid_ordered.append(edge)

    n_valid_total = len(valid_ordered)

    if invalid_count or duplicate_count:
        print(f"\n[SUMMARY] {invalid_count} invalid edge(s) and "
              f"{duplicate_count} duplicate edge(s) rejected (not counted toward budget).")

    over_budget  = n_valid_total > k
    under_budget = n_valid_total < k

    if over_budget:
        print(f"\n[WARNING] *** OVER BUDGET ***")
        print(f"          File has {n_valid_total} valid unique edge(s), but k={k}.")
        print(f"          Only the first {k} valid edge(s) in file order will be evaluated.")
        print(f"          The remaining {n_valid_total - k} edge(s) are IGNORED.")
        used = valid_ordered[:k]
    else:
        used = valid_ordered

    if not over_budget and not under_budget:
        print(f"[OK] Exactly k={k} valid unique blocked edges loaded — full budget used.")
    elif under_budget:
        print(f"[OK] {n_valid_total}/{k} valid unique blocked edge(s) loaded (under budget).")

    return frozenset(used), n_valid_total, over_budget, under_budget


# ─────────────────────────────────────────────────────────────────────────────
# 2. Simulation
# ─────────────────────────────────────────────────────────────────────────────

def simulate_once(adj, source_set, blocked, rng, hops=None):
    """
    One Independent-Cascade realisation on the graph with `blocked` edges removed.

    Parameters
    ----------
    adj        : adjacency list
    source_set : initially burning nodes A0
    blocked    : set of (u, v) edges that are blocked
    rng        : seeded random instance
    hops       : if provided, fire spread is limited to this many hops from
                 the seed set. Nodes beyond `hops` steps away from any seed
                 cannot be ignited. None means unlimited spread.

    Returns |A_∞| — total number of burned nodes at termination.
    """
    burned = set(source_set)
    # Each frontier entry is (node, current_hop_depth) so we can enforce
    # the hop limit. Seeds start at depth 0.
    frontier = [(node, 0) for node in source_set]

    while frontier:
        next_frontier = []
        for u, depth in frontier:
            # If hops is set and we've reached the limit, this node cannot
            # ignite any further neighbours — skip without stopping the BFS,
            # since other nodes at shallower depths may still propagate.
            if hops is not None and depth >= hops:
                continue
            for (v, p) in adj.get(u, ()):
                if (u, v) not in blocked and v not in burned:
                    if rng.random() < p:
                        burned.add(v)
                        next_frontier.append((v, depth + 1))
        frontier = next_frontier

    return len(burned)


def estimate_sigma(adj, source_set, blocked, num_sim, base_seed=42, hops=None):
    """
    Monte-Carlo estimate of σ(blocked) = E[|A_∞|] over `num_sim` realisations.

    Parameters
    ----------
    hops : passed through to simulate_once; limits spread to this many hops
           from the seed set. None means unlimited spread.
    """
    rng     = random.Random(base_seed)
    results = [simulate_once(adj, source_set, blocked, rng, hops=hops)
               for _ in range(num_sim)]
    return sum(results) / num_sim


# ─────────────────────────────────────────────────────────────────────────────
# 3. Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sigma(R) and rho(R) for blocked forest fire routes.")
    parser.add_argument('--graph_file')
    parser.add_argument('--seed_file')
    parser.add_argument('--blocked_file')
    parser.add_argument('--k',       type=int)
    parser.add_argument('--num_sim', type=int)
    parser.add_argument('--base_seed', type=int, default=42)
    parser.add_argument(
        '--hops', type=int, default=-1,
        help="Limit fire spread to this many hops from the seed set. "
             "-1 (default) means unlimited spread."
    )
    args = parser.parse_args()

    # Normalise: -1 sentinel → None (unlimited)
    hops     = None if args.hops < 0 else args.hops
    hops_str = "unlimited" if hops is None else str(hops)

    print("=" * 65)
    print("  Forest Fire Spread Evaluator  —  COL761 A2 Q2")
    print("=" * 65)
    print(f"  Graph        : {args.graph_file}")
    print(f"  Seed file    : {args.seed_file}")
    print(f"  Blocked file : {args.blocked_file}")
    print(f"  k (budget)   : {args.k}")
    print(f"  Simulations  : {args.num_sim}")
    print(f"  Base seed    : {args.base_seed}")
    print(f"  Hops         : {hops_str}")
    print("=" * 65)

    nodes, adj, edge_set = load_graph(args.graph_file)
    source_set           = load_seeds(args.seed_file)
    blocked, n_valid_total, over_budget, under_budget = load_blocked(
        args.blocked_file, edge_set, args.k)

    missing_seeds = source_set - nodes
    if missing_seeds:
        print(f"[WARN] {len(missing_seeds)} seed node(s) not in graph: "
              f"{sorted(missing_seeds)[:10]}{'...' if len(missing_seeds) > 10 else ''}")

    all_edges = [(u, v) for u, nbrs in adj.items() for (v, _) in nbrs]
    print(f"\n  Graph nodes        : {len(nodes)}")
    print(f"  Graph edges        : {len(all_edges)}")
    seeds_preview = sorted(source_set)[:10]
    print(f"  Seeds |A0|         : {len(source_set)}  {seeds_preview}"
          f"{'...' if len(source_set) > 10 else ''}")
    print(f"  Valid edges in file: {n_valid_total}  (budget k={args.k})")
    print(f"  Edges used for σ(R): {len(blocked)}")

    print(f"\n  Computing σ(∅)  (baseline, no edges blocked) ...", flush=True)
    mu0 = estimate_sigma(adj, source_set, frozenset(), args.num_sim, args.base_seed,
                         hops=hops)

    print(f"  Computing σ(R)  (with {len(blocked)} blocked edges) ...", flush=True)
    muR = estimate_sigma(adj, source_set, blocked, args.num_sim, args.base_seed,
                         hops=hops)

    reduction     = mu0 - muR
    reduction_pct = 100.0 * reduction / max(mu0, 1e-9)
    rho_R         = reduction / max(mu0, 1e-9)

    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  σ(∅)           = {mu0:>10.4f}")
    print(f"  σ(R)           = {muR:>10.4f}")
    print(f"  |R| evaluated  = {len(blocked)}  (budget k={args.k})")
    print(f"  Reduction      = {reduction:>8.4f}  ({reduction_pct:.2f}%)")
    print(f"  ρ(R)           = {rho_R:.6f}")
    print(f"  Hops           = {hops_str}")
    print("=" * 65)

    if over_budget:
        print(f"\n  [WARNING] *** OVER-BUDGET SUBMISSION ***")
        print(f"            {n_valid_total} valid edge(s) found; only the first {args.k} evaluated.")
        print(f"            σ(R) above reflects only those {args.k} edges.")
        print(f"            A clean k-edge submission with better choices may score higher.")
    if under_budget:
        print(f"\n  [WARNING] Under-budget: {n_valid_total}/{args.k} edge blocks used.")
        print(f"            Blocking {args.k - n_valid_total} more edge(s) could improve ρ(R).")

    # Machine-readable summary line
    print(f"\nSUMMARY  sigma_empty={mu0:.4f}  sigma_R={muR:.4f}  rho_R={rho_R:.6f}"
          f"  edges_evaluated={len(blocked)}  k={args.k}  num_sim={args.num_sim}"
          f"  hops={hops_str}")


if __name__ == "__main__":
    main()
