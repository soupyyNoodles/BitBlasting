# Assignment runner walkthrough

This document explains what each part of the similarity-search assignment harness does and how the pieces fit together.

## Files

| File | Role |
|------|------|
| `main.py` | CLI, data loading, timing, optional hard timeout, evaluation, writing outputs |
| `submission.py` | **Student file** — implement `solve(...)` here (starter raises `NotImplementedError`) |

## Student contract: `solve`

Students implement a single function:

```text
solve(base_vectors, query_vectors, k, K, time_budget) -> np.ndarray of shape (K,)
```

- `base_vectors`: `(N, d)`, `float32`
- `query_vectors`: `(Q, d)`, `float32`
- `k`: per-query neighbor count (e.g. for building candidates before aggregating)
- `K`: length of the final ranked list of **base vector indices** (order matters)
- `time_budget`: seconds (hint for internal stopping; the grader may also enforce a wall-clock limit)

The wrapper does **not** print large arrays; it only reports scalar **runtime** and **nDCG@K**.

## `main.py` — data loading

- **`load_vector_matrix`**: Loads a 2D `.npy` file. With `--mmap`, uses memory mapping so huge files are not necessarily fully read into RAM at once; dtype is converted to `float32` with a contiguous copy only when needed.
- **`--transpose`**: If your arrays are stored column-wise as `(d, N)` for the base and `(d, Q)` for queries (instead of `(N, d)` and `(Q, d)`), pass this flag so each matrix is transposed after load before calling `solve`.
- **`load_ground_truth`**: Loads the reference ranking. Supports `.npy` or plaintext (one integer index per line, first line = rank 1, etc.). Length must match `--K`.

## `main.py` — parallelism

- OpenMP thread count is influenced by **`OMP_NUM_THREADS`** (set before `faiss` is imported in `main()`).
- **`configure_parallelism`**: Calls `faiss.omp_set_num_threads` so FAISS respects `--parallel` / `--no-parallel` and optional `--threads`.
- Students should still **batch** queries (and any other heavy loops) instead of iterating one query at a time when `Q` is large.

## `main.py` — running `solve` and time limits

Two modes:

1. **Default (subprocess + `fork` on Linux)**  
   `solve` runs in a child process. The parent waits up to `--time_limit` seconds. If the child is still running, it is terminated, a warning is printed, and the result is treated as missing (then normalized to `-1` placeholders — see below).  
   This gives a **hard** wall-clock cap without relying on cooperative checks inside student code. On Linux, `fork` avoids pickling multi-gigabyte arrays to the child.

2. **`--no_subprocess`**  
   `solve` runs in-process. This avoids process overhead and is appropriate for very large resident sets, but overrun is only detected after `solve` returns (soft limit: warning if wall time > `--time_limit`).

The **`time_budget`** argument passed into `solve` defaults to `--time_limit` unless `--time_budget` is set explicitly (useful if you want students to reserve time for post-processing).

## `main.py` — output normalization

- **`normalize_student_output`**: Ensures a length-`K` `int64` vector. Too-long outputs are truncated; too-short are padded with `-1`. `None` becomes all `-1`.
- **`write_indices`**: Saves one index per line (human-readable, small logs).

## `main.py` — evaluation: nDCG@K

- Ground truth defines a **graded relevance** by rank: the item at rank `r` (1-based) in the reference list gets relevance `K - r + 1` mapped by index.
- **IDCG** is the DCG of that ideal ordering.
- **DCG** of the student list uses the same relevance function for each index; unknown or duplicate indices (after the first occurrence) contribute no extra gain at later ranks.

This yields a score in `[0, 1]`, with `1` when the student order matches the ground truth.

## CLI usage

Example:

```bash
python main.py \
  --base_vectors ./data/base.npy \
  --query_vectors ./data/query.npy \
  --ground_truth ./data/gt.npy \
  --k 10 \
  --K 100 \
  --time_limit 60 \
  --output ./my_indices.txt
```

Optional flags: `--submission_module`, `--mmap`, `--transpose`, `--parallel` / `--no-parallel`, `--threads`, `--time_budget`, `--no_subprocess`, `--output` (default `output_indices.txt`).
