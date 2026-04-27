import argparse
from pathlib import Path

import torch
from torch_geometric.data import Data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a small B-style node-classification dataset for smoke tests.")
    parser.add_argument("--data_dir", required=True, help="Directory containing B/data.pt")
    parser.add_argument("--out_dir", required=True, help="Output data root; writes out_dir/B/data.pt")
    parser.add_argument("--num_nodes", type=int, default=50_000)
    parser.add_argument("--num_labeled", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mmap", action="store_true")
    return parser.parse_args()


def safe_load(path: Path, mmap: bool):
    kwargs = {"weights_only": False, "map_location": "cpu"}
    if mmap:
        try:
            return torch.load(path, mmap=True, **kwargs)
        except TypeError:
            pass
    return torch.load(path, **kwargs)


def main() -> None:
    args = parse_args()
    data = safe_load(Path(args.data_dir) / "B" / "data.pt", mmap=args.mmap)
    generator = torch.Generator().manual_seed(args.seed)

    labeled_perm = torch.randperm(data.labeled_nodes.numel(), generator=generator)
    labeled_pos = labeled_perm[: min(args.num_labeled, labeled_perm.numel())]
    must_keep = data.labeled_nodes[labeled_pos].unique()

    remaining = torch.ones(data.num_nodes, dtype=torch.bool)
    remaining[must_keep] = False
    candidates = remaining.nonzero(as_tuple=False).view(-1)
    extra_count = max(0, min(args.num_nodes, data.num_nodes) - must_keep.numel())
    extra = candidates[torch.randperm(candidates.numel(), generator=generator)[:extra_count]]
    kept_nodes = torch.cat([must_keep, extra]).unique().sort().values

    remap = torch.full((data.num_nodes,), -1, dtype=torch.long)
    remap[kept_nodes] = torch.arange(kept_nodes.numel())

    edge_mask = (remap[data.edge_index[0]] >= 0) & (remap[data.edge_index[1]] >= 0)
    edge_index = remap[data.edge_index[:, edge_mask]]

    labeled_mask = remap[data.labeled_nodes] >= 0
    labeled_nodes = remap[data.labeled_nodes[labeled_mask]]
    labels = data.y[labeled_mask]
    train_mask = data.train_mask[labeled_mask]
    val_mask = data.val_mask[labeled_mask]

    mini = Data(
        x=data.x[kept_nodes].contiguous(),
        edge_index=edge_index.contiguous(),
        y=labels.contiguous(),
        labeled_nodes=labeled_nodes.contiguous(),
        train_mask=train_mask.contiguous(),
        val_mask=val_mask.contiguous(),
        num_nodes=kept_nodes.numel(),
    )

    out_path = Path(args.out_dir) / "B"
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save(mini, out_path / "data.pt")
    print(f"Saved {out_path / 'data.pt'}")
    print(f"nodes={mini.num_nodes} edges={mini.edge_index.shape[1]} labeled={mini.y.numel()}")
    print(f"train={int(mini.train_mask.sum())} val={int(mini.val_mask.sum())}")


if __name__ == "__main__":
    main()
