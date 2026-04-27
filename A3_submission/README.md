# COL761 Assignment 3 — Submission Notes

## Environment

- **Python:** 3.10
- **PyTorch:** 2.7.1 (install with the CUDA build matching your environment, e.g. `torch==2.7.1+cu118`)
- **PyTorch Geometric:** 2.7.0
- **faiss-cpu:** 1.13.2 (Q1 only)
- **numpy:** see `Q2/requirements.txt` (Q1 grader environment installs `numpy==1.26.3` per the assignment spec)

We do **not** use `pyg_lib`, `torch_scatter`, `torch_sparse`, or `torch_cluster` — pure `torch_geometric` 2.7.0 with the standard PyTorch fallbacks is sufficient.

## Q1 (Search in High-Dimensional Space)

- Single file: `Q1/submission.py`.
- Implements `solve(base_vectors, query_vectors, k, K, time_budget)` per the assignment interface.
- Uses `faiss-cpu`. Builds an exact `IndexFlatL2` for small/loose-budget cases and an `IndexIVFFlat` (with adaptively chosen `nlist`/`nprobe`) for larger or tighter-budget cases.

## Q2 (Graph Prediction)

Layout follows §2.4 of the assignment:

```
Q2/
├── requirements.txt
└── src/
    ├── load_dataset.py       (official, unchanged)
    ├── predict.py            (official, unchanged)
    ├── evaluate.py           (official, unchanged)
    ├── train.py              (unified entry point — see Piazza train command)
    ├── train_A.py            (per-dataset training logic, called by train.py)
    ├── train_B.py
    ├── train_C.py
    ├── model_A.py
    ├── model_B.py
    ├── model_C.py
    └── utils.py
```

### Train / predict / evaluate commands

```
python train.py    --dataset A|B|C --task node|link --data_dir <abs> --model_dir <abs> --kerberos <id>
python predict.py  --dataset A|B|C --task node|link --data_dir <abs> --model_dir <abs> --output_dir <abs> --kerberos <id>
python evaluate.py --dataset A|B|C --task node|link --data_dir <abs> --output_dir <abs> --kerberos <id>
```

Models are saved as `<model_dir>/<kerberos>_model_<dataset>.pt` — the same name `predict.py` loads.

### Architectures

- **Dataset A (node classification, 7 classes):** GATv2 + Correct-and-Smooth, with multi-seed ensemble selection.
- **Dataset B (binary node classification):** PMLP (parameterised MLP with symmetric-normalised propagation at inference) wrapped with Correct-and-Smooth — graph propagation is used both at inference and through the C&S residual diffusion.
- **Dataset C (link prediction, Hits@50):** Pairwise scorer over the dataset's precomputed `gnn_feature` embeddings, trained with a binary cross-entropy objective against the supplied negatives.
