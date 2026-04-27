import argparse
import json
import time
from pathlib import Path

import numpy as np

import faiss

from Q1.submission import (
    SearchConfig,
    _as_f32_contiguous,
    _build_index,
    _rank,
)


def ndcg_at_k(student_ranking: np.ndarray, ground_truth: np.ndarray) -> float:
    K = int(ground_truth.shape[0])
    rel_by_index = {int(ground_truth[i]): float(K - i) for i in range(K)}
    idcg = sum((K - i) / np.log2(i + 2.0) for i in range(K))

    dcg = 0.0
    seen = set()
    for j, idx in enumerate(student_ranking[:K]):
        idx = int(idx)
        if idx in seen:
            continue
        seen.add(idx)
        dcg += rel_by_index.get(idx, 0.0) / np.log2(j + 2.0)
    return float(dcg / idcg)


def recall_at_k(pred: np.ndarray, truth: np.ndarray) -> float:
    hits = 0
    total = pred.shape[0] * pred.shape[1]
    for i in range(pred.shape[0]):
        hits += len(set(map(int, pred[i])) & set(map(int, truth[i])))
    return hits / total


def recall_at_k_batch(pred: np.ndarray, truth: np.ndarray) -> tuple[int, int]:
    hits = 0
    total = pred.shape[0] * pred.shape[1]
    for i in range(pred.shape[0]):
        hits += len(set(map(int, pred[i])) & set(map(int, truth[i])))
    return hits, total


def run_config(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    visible_gt: np.ndarray,
    visible_neighbors: np.ndarray,
    config: SearchConfig,
    k: int,
    K: int,
    tie_mode: str,
    compute_recall: bool,
) -> tuple[float, float, float | None]:
    counts = np.zeros(base_vectors.shape[0], dtype=np.int64)
    first_seen = np.full(base_vectors.shape[0], np.iinfo(np.int64).max, dtype=np.int64)

    t0 = time.perf_counter()
    index = _build_index(base_vectors, config)
    build_time = time.perf_counter() - t0

    stream_offset = 0
    recall_hits = 0
    recall_total = 0
    t1 = time.perf_counter()
    for start in range(0, query_vectors.shape[0], config.batch_size):
        stop = min(query_vectors.shape[0], start + config.batch_size)
        _, neighbors = index.search(query_vectors[start:stop], k)
        stream_offset = _search_batch(counts, first_seen, neighbors, stream_offset)
        if compute_recall:
            hits, total = recall_at_k_batch(neighbors, visible_neighbors[start:stop])
            recall_hits += hits
            recall_total += total
    search_time = time.perf_counter() - t1

    pred_ranking = _rank(counts, first_seen, K, tie_mode=tie_mode)
    return (
        build_time + search_time,
        ndcg_at_k(pred_ranking, visible_gt),
        (recall_hits / recall_total) if compute_recall and recall_total else None,
    )


def _search_batch(
    counts: np.ndarray,
    first_seen: np.ndarray,
    neighbors: np.ndarray,
    stream_offset: int,
) -> int:
    flat = neighbors.reshape(-1)
    counts += np.bincount(flat, minlength=counts.shape[0])
    unique_ids, first_idx = np.unique(flat, return_index=True)
    absolute_pos = first_idx.astype(np.int64) + np.int64(stream_offset)
    current = first_seen[unique_ids]
    update_mask = absolute_pos < current
    first_seen[unique_ids[update_mask]] = absolute_pos[update_mask]
    return stream_offset + flat.size


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Q1 search configurations.")
    parser.add_argument("--dataset_dir", required=True, help="Path to D1/processed or D2/processed.")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--tie_mode", choices=["first_seen", "index"], default="first_seen")
    parser.add_argument("--compute_recall", action="store_true")
    parser.add_argument("--include_flat", action="store_true")
    parser.add_argument("--include_hnsw", action="store_true")
    parser.add_argument("--include_ivfpq", action="store_true")
    parser.add_argument("--nlist", default="4096,8192,16384")
    parser.add_argument("--nprobe", default="32,64,96,128")
    parser.add_argument("--out_json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads is not None:
        faiss.omp_set_num_threads(args.threads)
    dataset_dir = Path(args.dataset_dir)

    base_vectors = _as_f32_contiguous(np.load(dataset_dir / "data.npy"))
    query_vectors = _as_f32_contiguous(np.load(dataset_dir / "visible.npy"))
    visible_gt = np.loadtxt(dataset_dir / "visible_GT.txt", dtype=np.int64)
    visible_neighbors = np.load(dataset_dir / "visible_nearest_neighbors.npy")

    if args.max_queries is not None:
        query_vectors = query_vectors[:args.max_queries]
        visible_neighbors = visible_neighbors[:args.max_queries]

    num_base = base_vectors.shape[0]
    configs = []
    if args.include_flat:
        configs.append(SearchConfig(name="flat", batch_size=4_096))
    if num_base >= 400_000:
        nlists = [int(x) for x in args.nlist.split(",") if x.strip()]
        nprobes = [int(x) for x in args.nprobe.split(",") if x.strip()]
        for nlist in nlists:
            for nprobe in nprobes:
                configs.append(
                    SearchConfig(
                        name="ivf_flat",
                        batch_size=8_192,
                        nlist=nlist,
                        nprobe=nprobe,
                    )
                )
                if args.include_ivfpq:
                    configs.append(
                        SearchConfig(
                            name="ivfpq",
                            batch_size=8_192,
                            nlist=nlist,
                            nprobe=nprobe,
                            pq_m=16,
                            pq_bits=8,
                        )
                    )
        if args.include_hnsw:
            configs.append(
                SearchConfig(
                    name="hnsw",
                    batch_size=4_096,
                    hnsw_m=32,
                    hnsw_ef_construction=200,
                    hnsw_ef_search=128,
                )
            )
    if not configs:
        configs.append(
            SearchConfig(
                name="flat" if num_base < 400_000 else "ivf_flat",
                batch_size=4_096 if num_base < 400_000 else 8_192,
                nlist=None if num_base < 400_000 else 8_192,
                nprobe=None if num_base < 400_000 else 64,
            )
        )

    print(f"FAISS threads: {faiss.omp_get_max_threads()}")
    print(f"Base shape: {base_vectors.shape}")
    print(f"Query shape: {query_vectors.shape}")
    print()

    results = []
    for config in configs:
        elapsed, ndcg, recall = run_config(
            base_vectors=base_vectors,
            query_vectors=query_vectors,
            visible_gt=visible_gt,
            visible_neighbors=visible_neighbors,
            config=config,
            k=args.k,
            K=args.K,
            tie_mode=args.tie_mode,
            compute_recall=args.compute_recall,
        )
        row = {
            "name": config.name,
            "nlist": config.nlist,
            "nprobe": config.nprobe,
            "time_sec": elapsed,
            "ndcg": ndcg,
            f"recall_at_{args.k}": recall,
            "tie_mode": args.tie_mode,
            "max_queries": args.max_queries,
        }
        results.append(row)
        print(
            f"{config.name:<8} "
            f"nlist={str(config.nlist):>5} "
            f"nprobe={str(config.nprobe):>4} "
            f"time={elapsed:8.2f}s "
            f"ndcg={ndcg:7.5f} "
            f"recall@{args.k}={recall if recall is not None else 'skip'}"
        )
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
