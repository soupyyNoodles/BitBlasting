import argparse
from pathlib import Path

import numpy as np

import faiss


def rank_from_neighbors(neighbors: np.ndarray, num_base: int, K: int) -> np.ndarray:
    flat = neighbors.reshape(-1)
    counts = np.bincount(flat, minlength=num_base)
    first_seen = np.full(num_base, np.iinfo(np.int64).max, dtype=np.int64)
    ids, first = np.unique(flat, return_index=True)
    first_seen[ids] = first
    return np.lexsort((np.arange(num_base, dtype=np.int64), first_seen, -counts))[:K]


def make_subset(
    source_dir: Path,
    out_dir: Path,
    num_base: int,
    num_queries: int,
    k: int,
    K: int,
    seed: int,
    random_base: bool,
    random_queries: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    base = np.load(source_dir / "data.npy", mmap_mode="r")
    queries = np.load(source_dir / "visible.npy", mmap_mode="r")

    if random_base:
        base_ids = np.sort(rng.choice(base.shape[0], size=num_base, replace=False))
    else:
        base_ids = np.arange(num_base)
    if random_queries:
        query_ids = np.sort(rng.choice(queries.shape[0], size=num_queries, replace=False))
    else:
        query_ids = np.arange(num_queries)

    base_mini = np.ascontiguousarray(base[base_ids], dtype=np.float32)
    query_mini = np.ascontiguousarray(queries[query_ids], dtype=np.float32)

    index = faiss.IndexFlatL2(base_mini.shape[1])
    index.add(base_mini)
    _, neighbors = index.search(query_mini, k)
    gt = rank_from_neighbors(neighbors, num_base=base_mini.shape[0], K=K)

    np.save(out_dir / "data.npy", base_mini)
    np.save(out_dir / "visible.npy", query_mini)
    np.save(out_dir / "visible_nearest_neighbors.npy", neighbors.astype(np.int64))
    np.savetxt(out_dir / "visible_GT.txt", gt.reshape(-1, 1), fmt="%d")

    with (out_dir / "README.txt").open("w") as f:
        f.write(f"source_dir={source_dir}\n")
        f.write(f"num_base={num_base}\n")
        f.write(f"num_queries={num_queries}\n")
        f.write(f"k={k}\n")
        f.write(f"K={K}\n")
        f.write(f"seed={seed}\n")
        f.write(f"random_base={random_base}\n")
        f.write(f"random_queries={random_queries}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Q1 mini datasets with exact FlatL2 ground truth.")
    parser.add_argument("--source_dir", required=True, help="Path to D1/processed or D2/processed.")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_base", type=int, required=True)
    parser.add_argument("--num_queries", type=int, required=True)
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_base", action="store_true")
    parser.add_argument("--random_queries", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_subset(
        source_dir=Path(args.source_dir),
        out_dir=Path(args.out_dir),
        num_base=args.num_base,
        num_queries=args.num_queries,
        k=args.k,
        K=args.K,
        seed=args.seed,
        random_base=args.random_base,
        random_queries=args.random_queries,
    )


if __name__ == "__main__":
    main()
