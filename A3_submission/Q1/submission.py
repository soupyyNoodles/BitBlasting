import math
import os
from dataclasses import dataclass

import numpy as np
import faiss


@dataclass(frozen=True)
class SearchConfig:
    name: str
    batch_size: int
    nlist: int | None = None
    nprobe: int | None = None
    hnsw_m: int | None = None
    hnsw_ef_construction: int | None = None
    hnsw_ef_search: int | None = None
    pq_m: int | None = None
    pq_bits: int | None = None


def _as_f32_contiguous(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32)


def _choose_nlist(num_base: int, time_budget: float) -> int:
    override = os.environ.get("A3_Q1_NLIST")
    if override:
        return int(override)
    if num_base >= 1_500_000:
        return 8_192
    if num_base >= 900_000:
        return 3_072 if time_budget <= 20 else 4_096
    if num_base >= 400_000:
        return 4_096
    return max(256, int(4 * math.sqrt(num_base)))


def _choose_nprobe(time_budget: float, num_base: int) -> int:
    override = os.environ.get("A3_Q1_NPROBE")
    if override:
        return int(override)
    if time_budget <= 20:
        return 40 if num_base >= 900_000 else 48
    if time_budget <= 40:
        return 96
    return 128


def _choose_config(num_base: int, time_budget: float) -> SearchConfig:
    index_type = os.environ.get("A3_Q1_INDEX", "").strip().lower()
    batch_size = int(os.environ.get("A3_Q1_BATCH_SIZE", "0") or "0")
    if index_type:
        if index_type == "flat":
            return SearchConfig(name="flat", batch_size=batch_size or 4_096)
        if index_type == "ivf_flat":
            return SearchConfig(
                name="ivf_flat",
                batch_size=batch_size or 8_192,
                nlist=_choose_nlist(num_base, time_budget),
                nprobe=_choose_nprobe(time_budget, num_base),
            )
        if index_type == "hnsw":
            return SearchConfig(
                name="hnsw",
                batch_size=batch_size or 4_096,
                hnsw_m=int(os.environ.get("A3_Q1_HNSW_M", "32")),
                hnsw_ef_construction=int(os.environ.get("A3_Q1_HNSW_EFC", "200")),
                hnsw_ef_search=int(os.environ.get("A3_Q1_HNSW_EFS", "128")),
            )
        if index_type == "ivfpq":
            return SearchConfig(
                name="ivfpq",
                batch_size=batch_size or 8_192,
                nlist=_choose_nlist(num_base, time_budget),
                nprobe=_choose_nprobe(time_budget, num_base),
                pq_m=int(os.environ.get("A3_Q1_PQ_M", "16")),
                pq_bits=int(os.environ.get("A3_Q1_PQ_BITS", "8")),
            )
        raise ValueError(f"Unsupported A3_Q1_INDEX={index_type!r}")

    if num_base <= 650_000 and time_budget >= 45:
        return SearchConfig(name="flat", batch_size=batch_size or 4_096)
    return SearchConfig(
        name="ivf_flat",
        batch_size=batch_size or 8_192,
        nlist=_choose_nlist(num_base, time_budget),
        nprobe=_choose_nprobe(time_budget, num_base),
    )


def _sample_queries_if_needed(query_vectors: np.ndarray, num_base: int, time_budget: float) -> np.ndarray:
    num_queries = query_vectors.shape[0]
    if num_queries <= 100_000:
        return query_vectors
    if time_budget < 20 and num_base * num_queries > 2.5e11:
        keep = max(40_000, int(0.6 * num_queries))
        return query_vectors[:keep]
    return query_vectors


def _sample_train_vectors(base_vectors: np.ndarray, nlist: int) -> np.ndarray:
    train_size = min(
        base_vectors.shape[0],
        int(os.environ.get("A3_Q1_TRAIN_SIZE", "0") or "0") or max(nlist * 40, 100_000),
    )
    if train_size == base_vectors.shape[0]:
        return base_vectors
    rng = np.random.default_rng(0)
    return base_vectors[rng.choice(base_vectors.shape[0], size=train_size, replace=False)]


def _build_flat_index(base_vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatL2(base_vectors.shape[1])
    index.add(base_vectors)
    return index


def _build_ivf_flat_index(base_vectors: np.ndarray, config: SearchConfig) -> faiss.IndexIVFFlat:
    assert config.nlist is not None and config.nprobe is not None
    quantizer = faiss.IndexFlatL2(base_vectors.shape[1])
    index = faiss.IndexIVFFlat(quantizer, base_vectors.shape[1], config.nlist, faiss.METRIC_L2)
    index.train(_sample_train_vectors(base_vectors, config.nlist))
    index.add(base_vectors)
    index.nprobe = config.nprobe
    return index


def _build_hnsw_index(base_vectors: np.ndarray, config: SearchConfig) -> faiss.IndexHNSWFlat:
    assert config.hnsw_m is not None and config.hnsw_ef_construction is not None and config.hnsw_ef_search is not None
    index = faiss.IndexHNSWFlat(base_vectors.shape[1], config.hnsw_m)
    index.hnsw.efConstruction = config.hnsw_ef_construction
    index.hnsw.efSearch = config.hnsw_ef_search
    index.add(base_vectors)
    return index


def _build_ivfpq_index(base_vectors: np.ndarray, config: SearchConfig) -> faiss.IndexIVFPQ:
    assert config.nlist is not None and config.nprobe is not None
    assert config.pq_m is not None and config.pq_bits is not None
    quantizer = faiss.IndexFlatL2(base_vectors.shape[1])
    index = faiss.IndexIVFPQ(quantizer, base_vectors.shape[1], config.nlist, config.pq_m, config.pq_bits)
    index.train(_sample_train_vectors(base_vectors, config.nlist))
    index.add(base_vectors)
    index.nprobe = config.nprobe
    return index


def _build_index(base_vectors: np.ndarray, config: SearchConfig) -> faiss.Index:
    if config.name == "flat":
        return _build_flat_index(base_vectors)
    if config.name == "ivf_flat":
        return _build_ivf_flat_index(base_vectors, config)
    if config.name == "hnsw":
        return _build_hnsw_index(base_vectors, config)
    if config.name == "ivfpq":
        return _build_ivfpq_index(base_vectors, config)
    raise ValueError(f"Unsupported config: {config.name}")


def _search_all(index: faiss.Index, query_vectors: np.ndarray, k: int, batch_size: int, counts: np.ndarray) -> None:
    for start in range(0, query_vectors.shape[0], batch_size):
        _, neighbors = index.search(query_vectors[start : start + batch_size], k)
        counts += np.bincount(neighbors.reshape(-1), minlength=counts.shape[0])


def _rank(counts: np.ndarray, top_k: int) -> np.ndarray:
    indices = np.arange(counts.shape[0], dtype=np.int64)
    order = np.lexsort((indices, -counts))
    return order[:top_k].astype(np.int64)


def solve(
    base_vectors: np.ndarray,
    query_vectors: np.ndarray,
    k: int,
    K: int,
    time_budget: float,
) -> np.ndarray:
    base_vectors = _as_f32_contiguous(base_vectors)
    query_vectors = _sample_queries_if_needed(
        _as_f32_contiguous(query_vectors), num_base=base_vectors.shape[0], time_budget=time_budget
    )

    counts = np.zeros(base_vectors.shape[0], dtype=np.int64)
    config = _choose_config(base_vectors.shape[0], time_budget)
    index = _build_index(base_vectors, config)
    _search_all(index, query_vectors, k, config.batch_size, counts)

    return _rank(counts, K)
