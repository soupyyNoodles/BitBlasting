import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import APPNP, GATv2Conv, GCN2Conv, GCNConv, SAGEConv
from torch_geometric.nn.models import CorrectAndSmooth

from Q2.src.utils import build_sym_norm_adj, probs_to_logits


class MLPNodeClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
    ):
        super().__init__()
        dims = [in_channels]
        dims.extend([hidden_channels] * max(0, num_layers - 1))
        dims.append(out_channels)

        self.layers = nn.ModuleList(
            nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)
        )
        self.norms = nn.ModuleList(
            nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 2)
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        del edge_index
        for layer, norm in zip(self.layers[:-1], self.norms):
            x = layer(x)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.layers[-1](x)


class GCNNodeClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(max(0, num_layers - 2)):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class APPNPNodeClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.5,
        propagation_steps: int = 10,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.propagation = APPNP(K=propagation_steps, alpha=alpha)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return self.propagation(x, edge_index)


class SAGENodeClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(max(0, num_layers - 2)):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        return self.convs[-1](x, edge_index)


class GATv2NodeClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        heads: int = 8,
        dropout: float = 0.6,
    ):
        super().__init__()
        head_channels = max(8, hidden_channels // heads)
        self.conv1 = GATv2Conv(
            in_channels,
            head_channels,
            heads=heads,
            concat=True,
            dropout=dropout,
        )
        self.conv2 = GATv2Conv(
            head_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv2(x, edge_index)


class GCNIINodeClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 16,
        dropout: float = 0.5,
        alpha: float = 0.1,
        theta: float = 0.5,
    ):
        super().__init__()
        self.lin_in = nn.Linear(in_channels, hidden_channels)
        self.convs = nn.ModuleList(
            GCN2Conv(
                hidden_channels,
                alpha=alpha,
                theta=theta,
                layer=i + 1,
                shared_weights=True,
                normalize=True,
            )
            for i in range(num_layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden_channels) for _ in range(num_layers))
        self.lin_out = nn.Linear(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin_in(x))
        x0 = x
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, x0, edge_index)
            x = norm(F.relu(x) + residual)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin_out(x)


class PMLPNodeClassifier(MLPNodeClassifier):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        propagation_steps: int = 10,
        alpha: float = 0.1,
    ):
        super().__init__(in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout)
        self.propagation_steps = propagation_steps
        self.alpha = alpha
        self._cached_adj: torch.Tensor | None = None
        self._cached_num_nodes: int | None = None
        self._cached_device: torch.device | None = None

    def _get_adj(self, edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> torch.Tensor:
        if (
            self._cached_adj is None
            or self._cached_num_nodes != num_nodes
            or self._cached_device != device
        ):
            self._cached_adj = build_sym_norm_adj(edge_index, num_nodes, device=device)
            self._cached_num_nodes = num_nodes
            self._cached_device = device
        return self._cached_adj

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        logits = super().forward(x, edge_index)
        if self.training or self.propagation_steps <= 0:
            return logits

        adj = self._get_adj(edge_index, x.shape[0], x.device)
        probs = torch.softmax(logits, dim=-1)
        propagated = probs
        for _ in range(self.propagation_steps):
            propagated = (1.0 - self.alpha) * torch.sparse.mm(adj, propagated) + self.alpha * probs
        return probs_to_logits(propagated)


class CorrectSmoothNodeClassifier(nn.Module):
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

        cas = self._get_cas()
        probs = torch.softmax(logits, dim=-1)
        train_mask = self.train_mask_full.to(probs.device)
        train_labels = self.train_labels_full.to(probs.device)
        corrected = cas.correct(probs, train_labels, train_mask, edge_index)
        smoothed = cas.smooth(corrected, train_labels, train_mask, edge_index)
        return probs_to_logits(smoothed)


def build_model_a(
    model_type: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dropout: float,
) -> nn.Module:
    if model_type == "mlp":
        return MLPNodeClassifier(in_channels, hidden_channels, out_channels, dropout=dropout)
    if model_type == "gcn":
        return GCNNodeClassifier(in_channels, hidden_channels, out_channels, dropout=dropout)
    if model_type == "appnp":
        return APPNPNodeClassifier(in_channels, hidden_channels, out_channels, dropout=dropout)
    if model_type == "pmlp":
        return PMLPNodeClassifier(in_channels, hidden_channels, out_channels, dropout=dropout)
    if model_type == "sage":
        return SAGENodeClassifier(in_channels, hidden_channels, out_channels, dropout=dropout)
    if model_type in ("gat", "gatv2"):
        return GATv2NodeClassifier(in_channels, hidden_channels, out_channels, dropout=dropout)
    if model_type == "gcnii":
        return GCNIINodeClassifier(in_channels, hidden_channels, out_channels, dropout=dropout)
    raise ValueError(f"Unknown A model type: {model_type}")
