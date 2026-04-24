import argparse
import copy
import os

import torch
import torch.nn.functional as F

from Q2.src.load_dataset import load_dataset
from Q2.src.model_A import CorrectSmoothNodeClassifier, build_model_a
from Q2.src.utils import EarlyStopper, LogitAveragingEnsemble, accuracy_from_logits, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train node-classification models for dataset A.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    parser.add_argument("--models", default="gcn,appnp,pmlp,mlp,sage,gatv2,gcnii")
    parser.add_argument("--hidden_channels", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", default=None, help="Comma-separated seeds. Defaults to 42,123,456.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--ensemble_size", type=int, default=5)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--drop_edge", type=float, default=0.2)
    parser.add_argument("--feature_norm", action="store_true")
    parser.add_argument("--no_cs", action="store_true", help="Disable CorrectAndSmooth wrappers.")
    return parser.parse_args()


def evaluate_model(model, x, edge_index, node_idx, labels) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(x, edge_index)
    return accuracy_from_logits(logits[node_idx], labels)


def drop_edge(edge_index: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0.0:
        return edge_index
    keep = torch.rand(edge_index.shape[1], device=edge_index.device) >= p
    if keep.any():
        return edge_index[:, keep]
    return edge_index


def train_candidate(model, x, edge_index, train_nodes, train_labels, val_nodes, val_labels, args) -> tuple[torch.nn.Module, float]:
    device = resolve_device(args.device)
    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    train_nodes = train_nodes.to(device)
    train_labels = train_labels.to(device)
    val_nodes = val_nodes.to(device)
    val_labels = val_labels.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    stopper = EarlyStopper(args.patience, mode="max")

    for _ in range(args.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        train_edge_index = drop_edge(edge_index, args.drop_edge)
        logits = model(x, train_edge_index)
        loss = F.cross_entropy(
            logits[train_nodes],
            train_labels,
            label_smoothing=args.label_smoothing,
        )
        loss.backward()
        optimizer.step()
        scheduler.step()

        val_acc = evaluate_model(model, x, edge_index, val_nodes, val_labels)
        stopper.update(val_acc, model)
        if stopper.should_stop():
            break

    model.load_state_dict(stopper.best_state)
    return model.cpu(), stopper.best_value


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = load_dataset("A", args.data_dir)
    data = dataset[0]
    data.x = data.x.float()
    if args.feature_norm:
        data.x = F.normalize(data.x, p=2, dim=1)

    train_nodes = data.labeled_nodes[data.train_mask]
    train_labels = data.y[data.train_mask]
    val_nodes = data.labeled_nodes[data.val_mask]
    val_labels = data.y[data.val_mask]

    full_train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    full_train_mask[train_nodes] = True
    full_train_labels = torch.zeros(data.num_nodes, dtype=torch.long)
    full_train_labels[train_nodes] = train_labels

    results = []
    model_types = [name.strip() for name in args.models.split(",") if name.strip()]
    seeds = [int(seed.strip()) for seed in (args.seeds or "42,123,456").split(",") if seed.strip()]
    for seed in seeds:
        set_seed(seed)
        for model_type in model_types:
            model = build_model_a(
                model_type=model_type,
                in_channels=data.x.shape[1],
                hidden_channels=args.hidden_channels,
                out_channels=dataset.num_classes,
                dropout=args.dropout,
            )
            trained_model, val_acc = train_candidate(
                model=model,
                x=data.x,
                edge_index=data.edge_index,
                train_nodes=train_nodes,
                train_labels=train_labels,
                val_nodes=val_nodes,
                val_labels=val_labels,
                args=args,
            )
            run_name = f"{model_type}_seed{seed}"
            results.append({"name": run_name, "score": val_acc, "model": trained_model})

            if args.no_cs:
                continue
            cs_model = CorrectSmoothNodeClassifier(
                base_model=copy.deepcopy(trained_model),
                train_mask_full=full_train_mask,
                train_labels_full=full_train_labels,
            )
            cs_score = evaluate_model(cs_model, data.x, data.edge_index, val_nodes, val_labels)
            results.append({"name": f"{run_name}_cs", "score": cs_score, "model": cs_model})

    results.sort(key=lambda item: item["score"], reverse=True)
    best_entry = results[0]

    if len(results) >= args.ensemble_size and args.ensemble_size > 1:
        ensemble_members = [copy.deepcopy(entry["model"]) for entry in results[:args.ensemble_size]]
        ensemble = LogitAveragingEnsemble(ensemble_members)
        ensemble_score = evaluate_model(ensemble, data.x, data.edge_index, val_nodes, val_labels)
        if ensemble_score > best_entry["score"]:
            best_entry = {"name": "ensemble", "score": ensemble_score, "model": ensemble}

    out_path = os.path.join(args.out_dir, f"{args.kerberos}_model_A.pt")
    torch.save(best_entry["model"], out_path)

    save_json(
        os.path.join(args.out_dir, f"{args.kerberos}_metrics_A.json"),
        {
            "best_model": best_entry["name"],
            "best_val_accuracy": best_entry["score"],
            "all_results": {entry["name"]: entry["score"] for entry in results},
        },
    )
    print(f"Saved best A model to {out_path}")
    print(f"Validation accuracy: {best_entry['score']:.4f}")


if __name__ == "__main__":
    main()
