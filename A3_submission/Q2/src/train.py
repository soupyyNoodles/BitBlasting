from models import CorrectSmoothNodeClassifier, build_model_a, CorrectSmoothBinaryClassifier, build_model_b, PairwiseMLPLinkPredictor
from utils import EarlyStopper, LogitAveragingEnsemble, accuracy_from_logits, binary_auc_from_logits, build_sym_norm_adj, resolve_device, save_json, set_seed, hits_at_k
from load_dataset import load_dataset
import torch
import torch.nn.functional as F
import copy
import os
import argparse



def evaluate_model_A(model, x, edge_index, node_idx, labels) -> float:
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


def train_candidate_A(model, x, edge_index, train_nodes, train_labels, val_nodes, val_labels, args) -> tuple[torch.nn.Module, float]:
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

        val_acc = evaluate_model_A(model, x, edge_index, val_nodes, val_labels)
        stopper.update(val_acc, model)
        if stopper.should_stop():
            break

    model.load_state_dict(stopper.best_state)
    return model.cpu(), stopper.best_value


def run_A(args) -> None:
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

    set_seed(args.seed)
    model_type = args.models.split(",")[0].strip()

    model = build_model_a(
        model_type=model_type,
        in_channels=data.x.shape[1],
        hidden_channels=args.hidden_channels,
        out_channels=dataset.num_classes,
        dropout=args.dropout,
    )
    trained_model, val_acc = train_candidate_A(
        model=model,
        x=data.x,
        edge_index=data.edge_index,
        train_nodes=train_nodes,
        train_labels=train_labels,
        val_nodes=val_nodes,
        val_labels=val_labels,
        args=args,
    )
    run_name = f"{model_type}_seed{args.seed}"

    cs_model = CorrectSmoothNodeClassifier(
        base_model=trained_model,
        train_mask_full=full_train_mask,
        train_labels_full=full_train_labels,
    )

    best_entry = {"name": f"{run_name}_cs", "score": 0.0, "model": cs_model}

    train_acc = evaluate_model_A(cs_model, data.x, data.edge_index, train_nodes, train_labels)

    out_path = os.path.join(args.out_dir, f"{args.kerberos}_model_A.pt")
    torch.save(best_entry["model"], out_path)

    save_json(
        os.path.join(args.out_dir, f"{args.kerberos}_metrics_A.json"),
        {
            "best_model": best_entry["name"],
            "train_accuracy": train_acc,
            "all_results": {best_entry["name"]: train_acc},
        },
    )

    print(f"Saved A model to {out_path}")
    print(f"Train Accuracy: {train_acc:.4f}")





def safe_torch_load(path: str, mmap: bool = False):
    kwargs = {"weights_only": False, "map_location": "cpu"}
    if mmap:
        try:
            return torch.load(path, mmap=True, **kwargs)
        except TypeError:
            pass
    return torch.load(path, **kwargs)


def evaluate_feature_batches_B(model, features, node_idx, labels, batch_size, device) -> float:
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


def evaluate_full_model_B(model, features, edge_index, val_nodes, val_labels) -> float:
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


def train_base_model_B(model, features, edge_index, train_nodes, train_labels, val_nodes, val_labels, args) -> tuple[torch.nn.Module, float]:
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

    if args.models.startswith("graphsage"):
        from torch_geometric.data import Data
        from torch_geometric.loader import NeighborLoader
        full_y = torch.zeros(features.shape[0], dtype=torch.long)
        full_y[train_nodes] = train_labels
        data = Data(x=features, edge_index=edge_index, y=full_y)
        train_loader = NeighborLoader(
            data,
            num_neighbors=[10] * getattr(args, 'num_layers', 4),
            batch_size=args.batch_size,
            input_nodes=train_nodes,
            shuffle=True,
        )
        for _ in range(args.epochs):
            model.train()
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch.x, batch.edge_index)
                loss = F.cross_entropy(logits[:batch.batch_size], batch.y[:batch.batch_size], weight=class_weights)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            val_auc = evaluate_full_model_B(model, features.to(device), edge_index.to(device), val_nodes, val_labels)
            stopper.update(val_auc, model)
            if stopper.should_stop():
                break
    else:
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

            val_auc = evaluate_feature_batches_B(
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


def run_B(args) -> None:
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

    set_seed(args.seed)
    model_type = args.models.split(",")[0].strip()

    model = build_model_b(
        model_type=model_type,
        in_channels=data.x.shape[1],
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        inference_chunk_size=args.inference_chunk_size,
        num_layers=args.num_layers,
        norm_type=args.norm_type,
    )
    trained_model, base_val_auc = train_base_model_B(
        model=model,
        features=data.x,
        edge_index=data.edge_index,
        train_nodes=train_nodes,
        train_labels=train_labels,
        val_nodes=val_nodes,
        val_labels=val_labels,
        args=args,
    )

    cs_model = CorrectSmoothBinaryClassifier(
        base_model=trained_model,
        train_mask_full=full_train_mask,
        train_labels_full=full_train_labels,
    )

    train_auc = evaluate_full_model_B(cs_model, data.x, data.edge_index, train_nodes, train_labels)
    best_entry = {"name": f"{model_type}_cs", "score": train_auc, "model": cs_model}

    out_path = os.path.join(args.out_dir, f"{args.kerberos}_model_B.pt")
    torch.save(best_entry["model"], out_path)

    save_json(
        os.path.join(args.out_dir, f"{args.kerberos}_metrics_B.json"),
        {
            "best_model": best_entry["name"],
            "train_auc": train_auc,
            "all_results": {best_entry["name"]: train_auc},
        },
    )
    print(f"Saved B model to {out_path}")
    print(f"Train ROC_AUC: {train_auc:.4f}")







def evaluate_model_C(model, x, edge_index, valid_pos, valid_neg) -> float:
    model.eval()
    with torch.no_grad():
        pos_scores = model(x, edge_index, valid_pos)
        P, K, _ = valid_neg.shape
        neg_scores = model(x, edge_index, valid_neg.view(P * K, 2)).view(P, K)
    return hits_at_k(pos_scores.cpu(), neg_scores.cpu(), k=50)





def train_candidate_C(model, x, edge_index, train_edges, train_labels, valid_pos, valid_neg, args):
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

        val_hits = evaluate_model_C(model, x, edge_index, valid_pos, valid_neg)
        stopper.update(val_hits, model)
        if stopper.should_stop():
            break

    model.load_state_dict(stopper.best_state)
    return model.cpu(), stopper.best_value


def run_C(args) -> None:
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

    set_seed(args.seed)
    model_type = args.models.split(",")[0].strip()

    if model_type == "mlp":
        model = PairwiseMLPLinkPredictor(
            in_channels=dataset.x.shape[1],
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            pair_chunk_size=args.pair_chunk_size,
            num_layers=getattr(args, "num_layers", 3),
        )
    else:
        raise ValueError(f"Unknown C model type: {model_type}")

    trained_model, val_hits = train_candidate_C(
        model=model,
        x=dataset.x,
        edge_index=dataset.edge_index,
        train_edges=train_edges,
        train_labels=train_labels,
        valid_pos=dataset.valid_pos,
        valid_neg=dataset.valid_neg,
        args=args,
    )

    model.eval()
    with torch.no_grad():
        P = dataset.train_pos.shape[0]
        K = 500
        device = next(model.parameters()).device
        train_pos = dataset.train_pos.to(device)
        pos_scores = model(dataset.x, dataset.edge_index, train_pos)
        neg_edges = torch.randint(0, dataset.num_nodes, (P * K, 2)).to(device)
        neg_scores = model(dataset.x, dataset.edge_index, neg_edges).view(P, K)
    
    from evaluate import hits_at_k
    train_hits = hits_at_k(pos_scores.cpu(), neg_scores.cpu(), k=50)

    best_entry = {"name": model_type, "score": train_hits, "model": trained_model}
    out_path = os.path.join(args.out_dir, f"{args.kerberos}_model_C.pt")
    torch.save(best_entry["model"], out_path)

    save_json(
        os.path.join(args.out_dir, f"{args.kerberos}_metrics_C.json"),
        {
            "best_model": best_entry["name"],
            "train_hits_at_50": train_hits,
            "all_results": {best_entry["name"]: train_hits},
        },
    )

    print(f"Saved C model to {out_path}")
    print(f"Train Hits@50: {train_hits:.4f}")




class _Args:
    pass

def train_setup_A(data_dir, out_dir, kerberos):
    args = _Args()
    args.data_dir = data_dir
    args.out_dir = out_dir
    args.kerberos = kerberos
    args.models = "gatv2"
    args.hidden_channels = 256
    args.dropout = 0.6
    args.epochs = 400
    args.patience = 60
    args.lr = 5e-3
    args.weight_decay = 5e-4
    args.seed = 456
    args.seeds = None
    args.device = None
    args.ensemble_size = 5
    args.label_smoothing = 0.1
    args.drop_edge = 0.2
    args.feature_norm = False # Disabled to match predict.py raw tensor handling
    args.no_cs = False
    return args

def train_setup_B(data_dir, out_dir, kerberos):
    args = _Args()
    args.data_dir = data_dir
    args.out_dir = out_dir
    args.kerberos = kerberos
    args.models = "graphsage" # Originally passed forwarded += ["--models", "pmlp"]
    args.hidden_channels = 512
    args.dropout = 0.2
    args.epochs = 100
    args.patience = 12
    args.batch_size = 16384
    args.inference_chunk_size = 131072
    args.num_layers = 4
    args.norm_type = "batch"
    args.lr = 2e-3
    args.weight_decay = 1e-4
    args.seed = 42
    args.device = None
    args.mmap = False
    args.ensemble_size = 2
    args.sgc_steps = 0
    args.sgc_alpha = 0.0
    args.sign_steps = 0
    args.sign_alpha = 0.0
    return args

def train_setup_C(data_dir, out_dir, kerberos):
    args = _Args()
    args.data_dir = data_dir
    args.out_dir = out_dir
    args.kerberos = kerberos
    args.models = "mlp"
    args.num_layers = 3
    args.hidden_channels = 512
    args.dropout = 0.1
    args.epochs = 500
    args.patience = 60
    args.batch_size = 1024
    args.pair_chunk_size = 4096
    args.lr = 5e-4
    args.weight_decay = 1e-5
    args.seed = 42
    args.device = None
    return args

# Replace `args = parse_args()` with using `args` generated above.


def _validate_task_dataset(dataset: str, task: str, parser: argparse.ArgumentParser) -> None:
    valid = {"node": ("A", "B"), "link": ("C",)}
    if dataset not in valid[task]:
        parser.error(
            f"--task {task} is not valid for --dataset {dataset}. "
            f"Expected dataset in {valid[task]}."
        )

def main() -> None:
    parser = argparse.ArgumentParser(description="Unified training entry point for COL761 A3 Q2.")
    parser.add_argument("--dataset", required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", required=True, choices=["node", "link"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    args = parser.parse_args()

    _validate_task_dataset(args.dataset, args.task, parser)
    
    import sys
    sys.setrecursionlimit(2000)

    if not os.path.isabs(args.data_dir):
        parser.error("--data_dir must be an absolute path")

    os.makedirs(args.model_dir, exist_ok=True)

    if args.dataset == "A":
        run_args = train_setup_A(args.data_dir, args.model_dir, args.kerberos)
        run_A(run_args)
    elif args.dataset == "B":
        run_args = train_setup_B(args.data_dir, args.model_dir, args.kerberos)
        run_B(run_args)
    else:
        run_args = train_setup_C(args.data_dir, args.model_dir, args.kerberos)
        run_C(run_args)

    expected = os.path.join(args.model_dir, f"{args.kerberos}_model_{args.dataset}.pt")
    if not os.path.isfile(expected):
        raise FileNotFoundError(
            f"Training finished but expected model not found at {expected}. "
            f"predict.py will not be able to locate it."
        )

if __name__ == "__main__":
    main()
