import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopper:
    def __init__(self, patience: int, mode: str = "max"):
        self.patience = patience
        self.mode = mode
        self.best_value = -math.inf if mode == "max" else math.inf
        self.best_state: dict[str, Any] | None = None
        self.bad_epochs = 0

    def update(self, value: float, model: nn.Module) -> bool:
        improved = value > self.best_value if self.mode == "max" else value < self.best_value
        if improved:
            self.best_value = value
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


class LogitAveragingEnsemble(nn.Module):
    def __init__(self, models: list[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, *args, **kwargs):
        outputs = [model(*args, **kwargs) for model in self.models]
        return torch.stack(outputs, dim=0).mean(dim=0)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == labels).float().mean().item())


def binary_auc_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    if logits.ndim == 2 and logits.shape[1] == 2:
        scores = torch.softmax(logits, dim=-1)[:, 1]
    else:
        scores = torch.sigmoid(logits.reshape(-1))
    return float(roc_auc_score(labels.detach().cpu().numpy(), scores.detach().cpu().numpy()))


def hits_at_k(pos_scores: torch.Tensor, neg_scores: torch.Tensor, k: int = 50) -> float:
    higher = (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
    return float((higher < k).float().mean().item())


def build_sym_norm_adj(
    edge_index: torch.Tensor,
    num_nodes: int,
    add_self_loops: bool = True,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    edge_index = edge_index.to(device=device)
    if add_self_loops:
        loops = torch.arange(num_nodes, device=edge_index.device)
        loop_edges = torch.stack([loops, loops], dim=0)
        edge_index = torch.cat([edge_index, loop_edges], dim=1)

    row, col = edge_index
    values = torch.ones(row.shape[0], device=edge_index.device, dtype=dtype)
    deg = torch.zeros(num_nodes, device=edge_index.device, dtype=dtype)
    deg.index_add_(0, row, values)
    deg_inv_sqrt = deg.clamp(min=1.0).pow(-0.5)
    norm_values = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]
    return torch.sparse_coo_tensor(edge_index, norm_values, (num_nodes, num_nodes), device=edge_index.device).coalesce()


def chunked_module_forward(module: nn.Module, x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    if x.shape[0] <= chunk_size:
        return module(x)
    outputs = []
    for start in range(0, x.shape[0], chunk_size):
        stop = min(x.shape[0], start + chunk_size)
        outputs.append(module(x[start:stop]))
    return torch.cat(outputs, dim=0)


def to_probability(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 2 and logits.shape[1] > 1:
        return torch.softmax(logits, dim=-1)
    pos = torch.sigmoid(logits.reshape(-1))
    return torch.stack([1.0 - pos, pos], dim=-1)


def probs_to_logits(probs: torch.Tensor) -> torch.Tensor:
    return torch.log(probs.clamp(min=1e-6))


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
