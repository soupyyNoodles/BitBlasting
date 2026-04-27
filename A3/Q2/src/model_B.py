import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import CorrectAndSmooth

from Q2.src.utils import build_sym_norm_adj, probs_to_logits, to_probability


class BinaryLinearClassifier(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = 2, inference_chunk_size: int = 131072):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.inference_chunk_size = inference_chunk_size

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        del edge_index
        return _chunked_logits(self.forward_features, x, next(self.parameters()).device, self.inference_chunk_size)


class BinaryFeatureMLP(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 2,
        num_layers: int = 3,
        dropout: float = 0.2,
        inference_chunk_size: int = 131072,
        norm_type: str = "batch",
    ):
        super().__init__()
        dims = [in_channels]
        dims.extend([hidden_channels] * max(0, num_layers - 1))
        dims.append(out_channels)
        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        norms = []
        for i in range(len(dims) - 2):
            if norm_type == "batch":
                norms.append(nn.BatchNorm1d(dims[i + 1]))
            elif norm_type == "layer":
                norms.append(nn.LayerNorm(dims[i + 1]))
            elif norm_type == "none":
                norms.append(nn.Identity())
            else:
                raise ValueError(f"Unknown norm_type: {norm_type}")
        self.norms = nn.ModuleList(norms)
        self.dropout = dropout
        self.inference_chunk_size = inference_chunk_size

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        for layer, norm in zip(self.layers[:-1], self.norms):
            x = layer(x)
            x = norm(x)
            x = F.gelu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        del edge_index
        return _chunked_logits(self.forward_features, x, next(self.parameters()).device, self.inference_chunk_size)


class PMLPBinaryClassifier(BinaryFeatureMLP):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int = 2,
        num_layers: int = 3,
        dropout: float = 0.2,
        inference_chunk_size: int = 131072,
        propagation_steps: int = 20,
        alpha: float = 0.1,
        norm_type: str = "batch",
    ):
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            inference_chunk_size=inference_chunk_size,
            norm_type=norm_type,
        )
        self.propagation_steps = propagation_steps
        self.alpha = alpha
        self._cached_adj: torch.Tensor | None = None
        self._cached_num_nodes: int | None = None

    def _get_adj(self, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        if self._cached_adj is None or self._cached_num_nodes != num_nodes:
            self._cached_adj = build_sym_norm_adj(edge_index, num_nodes, device=edge_index.device)
            self._cached_num_nodes = num_nodes
        return self._cached_adj

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        logits = super().forward(x, edge_index)
        if self.training or self.propagation_steps <= 0:
            return logits

        adj = self._get_adj(edge_index.to(logits.device), logits.shape[0])
        probs = torch.softmax(logits, dim=-1)
        propagated = probs
        for _ in range(self.propagation_steps):
            propagated = (1.0 - self.alpha) * torch.sparse.mm(adj, propagated) + self.alpha * probs
        return probs_to_logits(propagated)


class CorrectSmoothBinaryClassifier(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        train_mask_full: torch.Tensor,
        train_labels_full: torch.Tensor,
        num_correction_layers: int = 50,
        correction_alpha: float = 0.5,
        num_smoothing_layers: int = 50,
        smoothing_alpha: float = 0.8,
        autoscale: bool = False,
        scale: float = 1.0,
    ):
        super().__init__()
        self.base_model = base_model
        self.register_buffer("train_mask_full", train_mask_full.bool())
        self.register_buffer("train_labels_full", train_labels_full.long())
        self.num_correction_layers = num_correction_layers
        self.correction_alpha = correction_alpha
        self.num_smoothing_layers = num_smoothing_layers
        self.smoothing_alpha = smoothing_alpha
        self.autoscale = autoscale
        self.scale = scale
        self._cas: CorrectAndSmooth | None = None

    def _get_cas(self) -> CorrectAndSmooth:
        if self._cas is None:
            self._cas = CorrectAndSmooth(
                num_correction_layers=self.num_correction_layers,
                correction_alpha=self.correction_alpha,
                num_smoothing_layers=self.num_smoothing_layers,
                smoothing_alpha=self.smoothing_alpha,
                autoscale=self.autoscale,
                scale=self.scale,
            )
        return self._cas

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        logits = self.base_model(x, edge_index)
        if self.training:
            return logits

        probs = to_probability(logits)
        train_mask = self.train_mask_full.to(probs.device)
        train_labels = self.train_labels_full.to(probs.device)
        train_labels_masked = train_labels[train_mask]
        cas = self._get_cas()
        corrected = cas.correct(probs, train_labels_masked, train_mask, edge_index)
        smoothed = cas.smooth(corrected, train_labels_masked, train_mask, edge_index)
        return probs_to_logits(smoothed)


def _chunked_logits(fn, x: torch.Tensor, device: torch.device, chunk_size: int) -> torch.Tensor:
    if x.device == device and x.shape[0] <= chunk_size:
        return fn(x)

    outputs = []
    for start in range(0, x.shape[0], chunk_size):
        stop = min(x.shape[0], start + chunk_size)
        chunk = x[start:stop].to(device)
        outputs.append(fn(chunk).cpu())
    result = torch.cat(outputs, dim=0)
    if device.type == "cpu":
        return result
    return result


def build_model_b(
    model_type: str,
    in_channels: int,
    hidden_channels: int,
    dropout: float,
    inference_chunk_size: int,
    num_layers: int = 4,
    norm_type: str = "batch",
) -> nn.Module:
    if model_type == "linear":
        return BinaryLinearClassifier(in_channels, inference_chunk_size=inference_chunk_size)
    if model_type == "mlp":
        return BinaryFeatureMLP(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            inference_chunk_size=inference_chunk_size,
            norm_type=norm_type,
        )
    if model_type == "pmlp":
        return PMLPBinaryClassifier(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            inference_chunk_size=inference_chunk_size,
            norm_type=norm_type,
        )
    raise ValueError(f"Unknown B model type: {model_type}")
