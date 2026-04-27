import argparse
import copy
import os

import torch
import torch.nn.functional as F

from Q2.src.load_dataset import load_dataset
from Q2.src.model_C import PairwiseMLPLinkPredictor
from Q2.src.utils import EarlyStopper, hits_at_k, resolve_device, save_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train link-prediction models for dataset C.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    parser.add_argument("--models", default="mlp")
    parser.add_argument("--hidden_channels", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--pair_chunk_size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def evaluate_model(model, x, edge_index, valid_pos, valid_neg) -> float:
    model.eval()
    with torch.no_grad():
        pos_scores = model(x, edge_index, valid_pos)
        P, K, _ = valid_neg.shape
        neg_scores = model(x, edge_index, valid_neg.view(P * K, 2)).view(P, K)
    return hits_at_k(pos_scores.cpu(), neg_scores.cpu(), k=50)





def train_candidate(model, x, edge_index, train_edges, train_labels, valid_pos, valid_neg, args):
    device = resolve_device(args.device)
    model = model.to(device)
    x = x.to(device)
    edge_index = edge_index.to(device)
    train_edges = train_edges.to(device)
    train_labels = train_labels.to(device)
    valid_pos = valid_pos.to(device)
    valid_neg = valid_neg.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    stopper = EarlyStopper(args.patience, mode="max")

    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(train_edges.shape[0], device=device)
        for start in range(0, perm.numel(), args.batch_size):
            stop = min(perm.numel(), start + args.batch_size)
            batch_ids = perm[start:stop]
            batch_edges = train_edges[batch_ids]
            batch_labels = train_labels[batch_ids]

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, edge_index, batch_edges)
            loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_hits = evaluate_model(model, x, edge_index, valid_pos, valid_neg)
        stopper.update(val_hits, model)
        if stopper.should_stop():
            break

    model.load_state_dict(stopper.best_state)
    return model.cpu(), stopper.best_value


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)

    dataset = load_dataset("C", args.data_dir)

    train_edges = torch.cat([dataset.train_pos, dataset.train_neg], dim=0)
    train_labels = torch.cat(
        [
            torch.ones(dataset.train_pos.shape[0], dtype=torch.float32),
            torch.zeros(dataset.train_neg.shape[0], dtype=torch.float32),
        ],
        dim=0,
    )

    results = []
    for model_type in [name.strip() for name in args.models.split(",") if name.strip()]:
        if model_type == "mlp":
            model = PairwiseMLPLinkPredictor(
                in_channels=dataset.x.shape[1],
                hidden_channels=args.hidden_channels,
                dropout=args.dropout,
                pair_chunk_size=args.pair_chunk_size,
            )

        else:
            raise ValueError(f"Unknown C model type: {model_type}")

        trained_model, val_hits = train_candidate(
            model=model,
            x=dataset.x,
            edge_index=dataset.edge_index,
            train_edges=train_edges,
            train_labels=train_labels,
            valid_pos=dataset.valid_pos,
            valid_neg=dataset.valid_neg,
            args=args,
        )
        results.append({"name": model_type, "score": val_hits, "model": trained_model})

    results.sort(key=lambda item: item["score"], reverse=True)
    best_entry = results[0]
    out_path = os.path.join(args.out_dir, f"{args.kerberos}_model_C.pt")
    torch.save(best_entry["model"], out_path)

    save_json(
        os.path.join(args.out_dir, f"{args.kerberos}_metrics_C.json"),
        {
            "best_model": best_entry["name"],
            "best_val_hits_at_50": best_entry["score"],
            "all_results": {entry["name"]: entry["score"] for entry in results},
        },
    )
    print("\n--- Model Hits@50 Comparison ---")
    for entry in results:
        print(f"Model: {entry['name']:<20} | Val Hits@50: {entry['score']:.4f}")
    print("---------------------------\n")

    print(f"Saved best C model to {out_path}")
    print(f"Validation Hits@50: {best_entry['score']:.4f}")


if __name__ == "__main__":
    main()
