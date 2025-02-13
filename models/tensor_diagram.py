import copy
import torch
import torch.nn as nn
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import Data
from torch_geometric.nn import MeanAggregation
from torch_geometric.nn.aggr import SumAggregation
from models.layers.merge_node import MergeNode
from models.layers.homp import IncidenceConv, AdjacencyConv
from typing import Dict, List, Union, Optional


HOMPLayer = Dict[str, Union[IncidenceConv, AdjacencyConv, MergeNode]]
SMCNLayer = Dict[
    str,
    Union[
        IncidenceConv,
        AdjacencyConv,
        MergeNode,
    ],
]
Layer = Union[HOMPLayer, SMCNLayer]


class TensorDiagram(nn.Module):
    def __init__(
        self,
        layers: List[Layer],
        embedding_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        dropout_list: Optional[List[float]] = None,
        final_dropout: float = 0.0,
        aggregation: str = "sum",
        output_ranks: Optional[List[int]] = None,
        residuals: Optional[List[bool]] = None,
        device: str = "cpu",
        zinc_head: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleDict(layer) for layer in layers])
        self.dropouts = (
            nn.ModuleList([nn.Dropout(p=dropout) for _ in layers])
            if dropout_list is None
            else nn.ModuleList([nn.Dropout(p=dropout) for dropout in dropout_list])
        )
        self.aggregation = (
            SumAggregation() if aggregation == "sum" else MeanAggregation()
        )
        self.output_ranks = output_ranks
        self.device = device
        self.zinc_head = zinc_head
        self.residuals = (
            [False for _ in self.layers] if residuals is None else residuals
        )

        if zinc_head:
            self.head = self._get_head(
                self.output_ranks, embedding_dim, output_dim, dropout
            )
        else:
            activation = nn.ReLU() if activation == "relu" else nn.GELU()
            self.linear_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(embedding_dim, embedding_dim * 2),
                        activation,
                    )
                    for _ in self.output_ranks
                ]
            )

            self.final_dropout = nn.Dropout(p=final_dropout)
            self.final_linear = nn.Linear(embedding_dim * 2, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        data = copy.deepcopy(data)
        batch_size = getattr(data, f"num_cells_0").shape[0]

        # Apply layers sequentially
        for layer, dropout_layer, residual in zip(
            self.layers, self.dropouts, self.residuals
        ):
            data = TensorDiagram._apply_homp_layer(
                data=data, layer=layer, dropout_layer=dropout_layer, residual=residual
            )

        out = []
        # Aggregate over cell features
        for rank in self.output_ranks:
            cell_features = getattr(data, f"x_{rank}")
            batch_index = torch.squeeze(getattr(data, f"x_{rank}_batch")).type(
                torch.int64
            )

            # In case of empty cells, pad final feature per complex so each complex has
            # one vector representation per rank
            batch_max_nonempty = (
                0 if batch_index.numel() == 0 else batch_index.max().item()
            )
            if batch_max_nonempty < batch_size - 1:
                batch_index = torch.concat(
                    [
                        batch_index,
                        torch.arange(batch_max_nonempty + 1, batch_size).to(
                            self.device
                        ),
                    ]
                )
                cell_features = torch.concat(
                    [
                        cell_features,
                        torch.zeros(
                            batch_size - batch_max_nonempty - 1, cell_features.shape[-1]
                        ).to(self.device),
                    ],
                    dim=0,
                )

            out.append(self.aggregation(cell_features, batch_index))

        if self.zinc_head:
            return self.head(torch.cat(out, dim=1))
        else:
            out = [self.linear_heads[i](out[i]) for i in range(len(self.output_ranks))]
            x_out = torch.sum(torch.stack(out, dim=0), dim=0)
            x_out = self.final_dropout(x_out)
            x_out = self.final_linear(x_out)
            return x_out

    @staticmethod
    def _apply_homp_layer(
        data: Data, layer: Layer, dropout_layer, residual: bool
    ) -> Data:
        for output_rank, l in layer.items():
            h = dropout_layer(l(data))
            if residual:
                x = getattr(data, output_rank) + h
            else:
                x = h

            setattr(data, output_rank, x)
        return data

    def _get_head(
        self, output_ranks: List[int], embed_dim: int, output_dim: int, dropout
    ):
        num_ranks = len(output_ranks)
        return Sequential(
            Linear(num_ranks * embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            ReLU(),
            nn.Dropout(p=dropout),
            Linear(embed_dim, 2 * embed_dim),
            nn.BatchNorm1d(2 * embed_dim),
            ReLU(),
            Linear(2 * embed_dim, output_dim),
        )
