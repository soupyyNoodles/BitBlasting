import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseHeuristicMixin:
    def __init__(self, pair_chunk_size: int):
        self.pair_chunk_size = pair_chunk_size
        self._adj_dense: torch.Tensor | None = None
        self._inv_log_deg: torch.Tensor | None = None
        self._inv_deg: torch.Tensor | None = None
        self._cached_num_nodes: int | None = None
        self._cached_device: torch.device | None = None

    def _ensure_graph_cache(self, edge_index: torch.Tensor, num_nodes: int, device: torch.device) -> None:
        if (
            self._adj_dense is not None
            and self._cached_num_nodes == num_nodes
            and self._cached_device == device
        ):
            return

        edge_index = edge_index.to(device)
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
        adj[edge_index[0], edge_index[1]] = 1.0
        adj.fill_diagonal_(0.0)
        deg = adj.sum(dim=1)

        inv_log_deg = torch.zeros_like(deg)
        mask = deg > 1
        inv_log_deg[mask] = 1.0 / torch.log(deg[mask])

        inv_deg = torch.zeros_like(deg)
        mask = deg > 0
        inv_deg[mask] = 1.0 / deg[mask]

        self._adj_dense = adj
        self._inv_log_deg = inv_log_deg
        self._inv_deg = inv_deg
        self._cached_num_nodes = num_nodes
        self._cached_device = device

    def _heuristic_features(
        self,
        edge_pairs: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        self._ensure_graph_cache(edge_index, num_nodes, device)
        edge_pairs = edge_pairs.to(device)
        u = edge_pairs[:, 0].long()
        v = edge_pairs[:, 1].long()

        common = self._adj_dense.index_select(0, u) * self._adj_dense.index_select(0, v)
        cn = common.sum(dim=1, keepdim=True)
        aa = (common * self._inv_log_deg.unsqueeze(0)).sum(dim=1, keepdim=True)
        ra = (common * self._inv_deg.unsqueeze(0)).sum(dim=1, keepdim=True)
        return torch.cat([cn, aa, ra], dim=1)


class HeuristicLinkPredictor(nn.Module, DenseHeuristicMixin):
    def __init__(self, pair_chunk_size: int = 4096, initial_weights: tuple[float, float, float] = (1.0, 1.0, 1.0)):
        nn.Module.__init__(self)
        DenseHeuristicMixin.__init__(self, pair_chunk_size=pair_chunk_size)
        self.weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    @classmethod
    def common_neighbors(cls, pair_chunk_size: int = 4096) -> "HeuristicLinkPredictor":
        model = cls(pair_chunk_size=pair_chunk_size, initial_weights=(1.0, 0.0, 0.0))
        model.weights.requires_grad_(False)
        model.bias.requires_grad_(False)
        return model

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        device = self.weights.device
        num_nodes = x.shape[0]
        scores = []
        for start in range(0, edge_pairs.shape[0], self.pair_chunk_size):
            stop = min(edge_pairs.shape[0], start + self.pair_chunk_size)
            heur = self._heuristic_features(edge_pairs[start:stop], edge_index, num_nodes, device)
            scores.append(heur @ self.weights + self.bias)
        return torch.cat(scores, dim=0).reshape(-1)


class PairwiseMLPLinkPredictor(nn.Module, DenseHeuristicMixin):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 256,
        dropout: float = 0.2,
        pair_chunk_size: int = 4096,
        use_heuristics: bool = True,
        use_raw_concat: bool = True,
        use_sum: bool = True,
        use_cosine: bool = True,
    ):
        nn.Module.__init__(self)
        DenseHeuristicMixin.__init__(self, pair_chunk_size=pair_chunk_size)
        self.use_heuristics = use_heuristics
        self.use_raw_concat = use_raw_concat
        self.use_sum = use_sum
        self.use_cosine = use_cosine

        input_dim = 2 * in_channels
        if use_raw_concat:
            input_dim += 2 * in_channels
        if use_sum:
            input_dim += in_channels
        if use_cosine:
            input_dim += 1
        if use_heuristics:
            input_dim += 3
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
        edge_index: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        x = x.to(device)
        edge_pairs = edge_pairs.to(device)
        u = edge_pairs[:, 0].long()
        v = edge_pairs[:, 1].long()
        xu = x.index_select(0, u)
        xv = x.index_select(0, v)
        parts = []
        if self.use_raw_concat:
            parts.extend([xu, xv])
        parts.extend([xu * xv, torch.abs(xu - xv)])
        if self.use_sum:
            parts.append(xu + xv)
        if self.use_cosine:
            parts.append(F.cosine_similarity(xu, xv, dim=1).unsqueeze(1))
        if self.use_heuristics:
            parts.append(self._heuristic_features(edge_pairs, edge_index, x.shape[0], device))
        return torch.cat(parts, dim=1)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        scores = []
        for start in range(0, edge_pairs.shape[0], self.pair_chunk_size):
            stop = min(edge_pairs.shape[0], start + self.pair_chunk_size)
            feats = self._pair_features(x, edge_pairs[start:stop], edge_index, device)
            scores.append(self.mlp(feats))
        return torch.cat(scores, dim=0).reshape(-1)


class DistMultLinkPredictor(nn.Module):
    def __init__(self, in_channels: int, pair_chunk_size: int = 65536):
        super().__init__()
        self.relation = nn.Parameter(torch.ones(in_channels))
        self.bias = nn.Parameter(torch.zeros(1))
        self.pair_chunk_size = pair_chunk_size

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        del edge_index
        device = self.relation.device
        x = x.to(device)
        scores = []
        for start in range(0, edge_pairs.shape[0], self.pair_chunk_size):
            stop = min(edge_pairs.shape[0], start + self.pair_chunk_size)
            pairs = edge_pairs[start:stop].to(device)
            xu = x.index_select(0, pairs[:, 0].long())
            xv = x.index_select(0, pairs[:, 1].long())
            scores.append((xu * self.relation * xv).sum(dim=1) + self.bias)
        return torch.cat(scores, dim=0).reshape(-1)
