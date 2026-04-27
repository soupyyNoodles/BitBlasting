import torch
import torch.nn as nn

class PairwiseMLPLinkPredictor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        dropout: float = 0.2,
        pair_chunk_size: int = 4096,
    ):
        super().__init__()
        self.pair_chunk_size = pair_chunk_size

        input_dim = in_channels
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def _pair_features(
        self,
        x: torch.Tensor,
        edge_pairs: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        x = x.to(device)
        edge_pairs = edge_pairs.to(device)
        u = edge_pairs[:, 0].long()
        v = edge_pairs[:, 1].long()
        xu = x.index_select(0, u)
        xv = x.index_select(0, v)
        return xu * xv

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        x = x.to(device)
        scores = []
        for start in range(0, edge_pairs.shape[0], self.pair_chunk_size):
            stop = min(edge_pairs.shape[0], start + self.pair_chunk_size)
            feats = self._pair_features(x, edge_pairs[start:stop], device)
            scores.append(self.mlp(feats))
        return torch.cat(scores, dim=0).reshape(-1)
