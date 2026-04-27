import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.models import CorrectAndSmooth

from utils import build_sym_norm_adj, probs_to_logits

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
        train_labels_masked = train_labels[train_mask]
        corrected = cas.correct(probs, train_labels_masked, train_mask, edge_index)
        smoothed = cas.smooth(corrected, train_labels_masked, train_mask, edge_index)
        return probs_to_logits(smoothed)


def build_model_a(
    model_type: str,
    in_channels: int,
    hidden_channels: int,
    out_channels: int,
    dropout: float,
) -> nn.Module:
    if model_type in ("gat", "gatv2"):
        return GATv2NodeClassifier(in_channels, hidden_channels, out_channels, dropout=dropout)
    raise ValueError(f"Unknown A model type: {model_type}")

