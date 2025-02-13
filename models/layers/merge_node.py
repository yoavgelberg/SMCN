import torch
import torch.nn as nn
from typing import List, Optional


class MergeNode(nn.Module):
    def __init__(
        self,
        layers: List[nn.Module],
        embedding_dim: int,
        aggregation: str = "concatenate",
        head_layer: Optional[nn.Module] = None,
        has_learnable_head: bool = True,
        batch_norm: bool = True,
        add_residual: bool = False,
        activation: str = "relu",
    ):
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.agg = aggregation
        self.has_learnable_head = has_learnable_head
        self.add_residual = add_residual  # warning: add residual is not supported when aggregation is concat

        if head_layer is None:
            input_dim = (
                embedding_dim * len(layers)
                if aggregation == "concatenate"
                else embedding_dim
            )
            norm = nn.BatchNorm1d(embedding_dim) if batch_norm else nn.Identity()
            activation = nn.ReLU() if activation == "relu" else nn.GELU()
            if self.has_learnable_head:
                self.head = nn.Sequential(
                    nn.Linear(input_dim, embedding_dim), norm, activation
                )
        else:
            self.head = head_layer

    def forward(self, data) -> torch.Tensor:
        if self.agg == "concatenate":
            x = torch.cat([layer(data) for layer in self.layers], dim=-1)
            h = self.head(x)
            x = x + h if self.add_residual else h

        if self.agg == "sum":
            x = torch.stack([layer(data) for layer in self.layers], dim=0)
            x = torch.sum(x, dim=0)

        elif self.agg == "mean":
            x = torch.stack([layer(data) for layer in self.layers], dim=0)
            x = torch.mean(x, dim=0)

        return x
