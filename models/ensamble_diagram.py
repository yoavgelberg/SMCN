import copy
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import MeanAggregation
from torch_geometric.nn.aggr import SumAggregation
from models.layers.merge_node import MergeNode
from models.layers.homp import IncidenceConv, AdjacencyConv
from typing import Dict, List, Union, Tuple

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


class EnsembleDiagram(nn.Module):
    def __init__(
        self,
        layers: Tuple[List[Layer]],
        dropouts: Tuple[List[float]],
        residuals: Tuple[List[bool]],
        embedding_dim: Tuple[int],
        output_ranks: Tuple[List[int]],
        output_dim: int,
        final_dropout: float = 0.0,
        aggregation: str = "mean",
        device: str = "cpu",
        activation: str = "relu",
        max_layer_index: int = 1,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList([nn.ModuleDict(layer) for layer in layer_list])
                for layer_list in layers
            ]
        )
        self.num_diagrams = len(layers)
        self.dropouts = nn.ModuleList(
            [
                nn.ModuleList([nn.Dropout(p=dropout) for dropout in dropout_list])
                for dropout_list in dropouts
            ]
        )
        self.residuals = residuals
        self.aggregation = (
            SumAggregation() if aggregation == "sum" else MeanAggregation()
        )
        self.output_ranks = output_ranks
        self.device = device
        activation = nn.ReLU() if activation == "relu" else nn.GELU()
        self.linear_heads = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Linear(embedding_dim[i], embedding_dim[i] * 2),
                            activation,
                        )
                        for _ in self.output_ranks[i]
                    ]
                )
                for i in range(self.num_diagrams)
            ]
        )
        self.final_embedding_dim = sum(embedding_dim)
        self.final_dropout = nn.Dropout(p=final_dropout)
        self.final_linear = nn.Linear(self.final_embedding_dim * 2, output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x_out = torch.cat(
            [
                self.apply_tensor_diagram(data, diagram_index)
                for diagram_index in range(self.num_diagrams)
            ],
            dim=-1,
        )
        x_out = self.final_linear(x_out)
        return x_out

    def apply_tensor_diagram(self, data: Data, diagram_index: int):
        layers = self.layers[diagram_index]
        dropouts = self.dropouts[diagram_index]
        residuals = self.residuals[diagram_index]
        output_ranks = self.output_ranks[diagram_index]
        linear_heads = self.linear_heads[diagram_index]  # ?

        data = copy.deepcopy(data)
        batch_size = getattr(data, f"num_cells_0").shape[0]

        # Apply layers sequentially
        for layer, dropout_layer, residual in zip(layers, dropouts, residuals):
            data = EnsembleDiagram._apply_homp_layer(
                data=data, layer=layer, dropout_layer=dropout_layer, residual=residual
            )
        out = []
        # Aggregate over cell features
        for rank in output_ranks:
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

        out = [linear_heads[i](out[i]) for i in range(len(output_ranks))]
        x_out = torch.sum(torch.stack(out, dim=0), dim=0)
        x_out = self.final_dropout(x_out)
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
