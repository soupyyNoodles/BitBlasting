import argparse
import importlib.util
import time
from pathlib import Path

import numpy as np


def ndcg_at_k(pred: np.ndarray, gt: np.ndarray) -> float:
    K = gt.shape[0]
    rel = {int(gt[i]): float(K - i) for i in range(K)}
    idcg = sum((K - i) / np.log2(i + 2.0) for i in range(K))
    dcg = 0.0
    seen = set()
    for j, idx in enumerate(pred[:K]):
        idx = int(idx)
        if idx in seen:
            continue
        seen.add(idx)
        dcg += rel.get(idx, 0.0) / np.log2(j + 2.0)
    return float(dcg / idcg)


def load_solve(path: Path):
    spec = importlib.util.spec_from_file_location("q1_submission_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.solve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Q1 solve() on a mini dataset and check output invariants.")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--submission", default=str(Path(__file__).with_name("submission.py")))
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--K", type=int, default=100)
    parser.add_argument("--time_budget", type=float, default=70)
    parser.add_argument("--min_ndcg", type=float, default=0.999)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    solve = load_solve(Path(args.submission))

    base = np.load(dataset_dir / "data.npy")
    queries = np.load(dataset_dir / "visible.npy")
    gt = np.loadtxt(dataset_dir / "visible_GT.txt", dtype=np.int64)

    start = time.perf_counter()
    pred = np.asarray(solve(base, queries, args.k, args.K, args.time_budget), dtype=np.int64).reshape(-1)
    elapsed = time.perf_counter() - start
    score = ndcg_at_k(pred, gt)

    assert pred.shape == (args.K,), f"Expected shape {(args.K,)}, got {pred.shape}"
    assert len(set(map(int, pred))) == args.K, "Output contains duplicate indices"
    assert int(pred.min()) >= 0 and int(pred.max()) < base.shape[0], "Output contains invalid base index"
    assert score >= args.min_ndcg, f"nDCG {score:.6f} below threshold {args.min_ndcg:.6f}"

    print(f"OK elapsed={elapsed:.3f}s ndcg={score:.6f}")


if __name__ == "__main__":
    main()
