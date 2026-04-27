import argparse
import copy
import os

import torch
import torch.nn.functional as F

from Q2.src.model_B import CorrectSmoothBinaryClassifier, build_model_b
from Q2.src.utils import (
    EarlyStopper,
    LogitAveragingEnsemble,
    binary_auc_from_logits,
    build_sym_norm_adj,
    resolve_device,
    save_json,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train large-graph binary classifiers for dataset B.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    parser.add_argument("--models", default="linear,mlp,pmlp")
    parser.add_argument("--hidden_channels", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=16384)
    parser.add_argument("--inference_chunk_size", type=int, default=131072)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--norm_type", choices=["batch", "layer", "none"], default="batch")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mmap", action="store_true")
    parser.add_argument("--ensemble_size", type=int, default=2)
    parser.add_argument("--sgc_steps", type=int, default=0)
    parser.add_argument("--sgc_alpha", type=float, default=0.0)
    parser.add_argument("--sign_steps", type=int, default=0)
    parser.add_argument("--sign_alpha", type=float, default=0.0)
    return parser.parse_args()


def safe_torch_load(path: str, mmap: bool = False):
    kwargs = {"weights_only": False, "map_location": "cpu"}
    if mmap:
        try:
            return torch.load(path, mmap=True, **kwargs)
        except TypeError:
            pass
    return torch.load(path, **kwargs)


def evaluate_feature_batches(model, features, node_idx, labels, batch_size, device) -> float:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, node_idx.numel(), batch_size):
            stop = min(node_idx.numel(), start + batch_size)
            batch_nodes = node_idx[start:stop]
            batch_logits = model.forward_features(features[batch_nodes].to(device))
            outputs.append(batch_logits.cpu())
    logits = torch.cat(outputs, dim=0)
    return binary_auc_from_logits(logits, labels)


def evaluate_full_model(model, features, edge_index, val_nodes, val_labels) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(features, edge_index)
    val_nodes = val_nodes.to(logits.device)
    return binary_auc_from_logits(logits[val_nodes], val_labels)


def maybe_sgc_features(features: torch.Tensor, edge_index: torch.Tensor, steps: int, alpha: float) -> torch.Tensor:
    if steps <= 0:
        return features

    adj = build_sym_norm_adj(edge_index, features.shape[0], device=torch.device("cpu"))
    out = features.float()
    residual = out.clone() if alpha > 0 else None
    for _ in range(steps):
        out = torch.sparse.mm(adj, out)
        if residual is not None:
            out = (1.0 - alpha) * out + alpha * residual
    return out


def maybe_sign_features(features: torch.Tensor, edge_index: torch.Tensor, steps: int, alpha: float) -> torch.Tensor:
    if steps <= 0:
        return features

    adj = build_sym_norm_adj(edge_index, features.shape[0], device=torch.device("cpu"))
    base = features.float()
    out = base
    parts = [base]
    for _ in range(steps):
        out = torch.sparse.mm(adj, out)
        if alpha > 0:
            out = (1.0 - alpha) * out + alpha * base
        parts.append(out)
    return torch.cat(parts, dim=1)


def train_base_model(model, features, train_nodes, train_labels, val_nodes, val_labels, args) -> tuple[torch.nn.Module, float]:
    device = resolve_device(args.device)
    model = model.to(device)

    train_nodes = train_nodes.cpu()
    train_labels = train_labels.cpu()
    val_nodes = val_nodes.cpu()
    val_labels = val_labels.cpu()

    class_counts = torch.bincount(train_labels, minlength=2).float()
    class_weights = (class_counts.sum() / class_counts.clamp(min=1.0)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    stopper = EarlyStopper(args.patience, mode="max")

    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(train_nodes.numel())
        for start in range(0, perm.numel(), args.batch_size):
            stop = min(perm.numel(), start + args.batch_size)
            batch_ids = perm[start:stop]
            batch_nodes = train_nodes[batch_ids]
            batch_x = features[batch_nodes].to(device)
            batch_y = train_labels[batch_ids].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model.forward_features(batch_x)
            loss = F.cross_entropy(logits, batch_y, weight=class_weights)
            loss.backward()
            optimizer.step()

        val_auc = evaluate_feature_batches(
            model=model,
            features=features,
            node_idx=val_nodes,
            labels=val_labels,
            batch_size=args.batch_size,
            device=device,
        )
        scheduler.step()
        stopper.update(val_auc, model)
        if stopper.should_stop():
            break

    model.load_state_dict(stopper.best_state)
    return model.cpu(), stopper.best_value


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    raw_path = os.path.join(args.data_dir, "B", "data.pt")
    data = safe_torch_load(raw_path, mmap=args.mmap)
    if args.sign_steps > 0:
        data.x = maybe_sign_features(data.x, data.edge_index, steps=args.sign_steps, alpha=args.sign_alpha)
    else:
        data.x = maybe_sgc_features(data.x, data.edge_index, steps=args.sgc_steps, alpha=args.sgc_alpha)

    train_nodes = data.labeled_nodes[data.train_mask]
    train_labels = data.y[data.train_mask].long()
    val_nodes = data.labeled_nodes[data.val_mask]
    val_labels = data.y[data.val_mask].long()

    full_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    full_train_mask[train_nodes] = True
    full_train_labels = torch.zeros(data.num_nodes, dtype=torch.long)
    full_train_labels[train_nodes] = train_labels.long()

    results = []
    for model_type in [name.strip() for name in args.models.split(",") if name.strip()]:
        model = build_model_b(
            model_type=model_type,
            in_channels=data.x.shape[1],
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            inference_chunk_size=args.inference_chunk_size,
            num_layers=args.num_layers,
            norm_type=args.norm_type,
        )
        trained_model, base_val_auc = train_base_model(
            model=model,
            features=data.x,
            train_nodes=train_nodes,
            train_labels=train_labels,
            val_nodes=val_nodes,
            val_labels=val_labels,
            args=args,
        )

        full_auc = evaluate_full_model(trained_model, data.x, data.edge_index, val_nodes, val_labels)
        results.append({"name": model_type, "score": full_auc, "model": trained_model})

        cs_model = CorrectSmoothBinaryClassifier(
            base_model=copy.deepcopy(trained_model),
            train_mask_full=full_train_mask,
            train_labels_full=full_train_labels,
        )
        cs_auc = evaluate_full_model(cs_model, data.x, data.edge_index, val_nodes, val_labels)
        results.append({"name": f"{model_type}_cs", "score": cs_auc, "model": cs_model})

    results.sort(key=lambda item: item["score"], reverse=True)
    best_entry = results[0]

    if len(results) >= args.ensemble_size and args.ensemble_size > 1:
        ensemble = LogitAveragingEnsemble([copy.deepcopy(entry["model"]) for entry in results[:args.ensemble_size]])
        ensemble_auc = evaluate_full_model(ensemble, data.x, data.edge_index, val_nodes, val_labels)
        if ensemble_auc > best_entry["score"]:
            best_entry = {"name": "ensemble", "score": ensemble_auc, "model": ensemble}

    out_path = os.path.join(args.out_dir, f"{args.kerberos}_model_B.pt")
    torch.save(best_entry["model"], out_path)

    save_json(
        os.path.join(args.out_dir, f"{args.kerberos}_metrics_B.json"),
        {
            "best_model": best_entry["name"],
            "best_val_auc": best_entry["score"],
            "all_results": {entry["name"]: entry["score"] for entry in results},
        },
    )
    print(f"Saved best B model to {out_path}")
    print(f"Validation AUC: {best_entry['score']:.4f}")


if __name__ == "__main__":
    main()
