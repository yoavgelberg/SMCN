import torch
import torch.nn as nn
from torch.nn import Sequential, Linear
from torch_geometric.data import Data
from torch_geometric.nn.conv import GINConv, GINEConv
from data.complex_data import ComplexData
from typing import Optional, Union
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    Size,
)


class IncidenceConv(nn.Module):
    def __init__(
        self,
        input_rank: int,
        output_rank: int,
        embedding_dim: int,
        mlp: Optional[nn.Module] = None,
        conv_type: str = "gin",
        batch_norm: bool = True,
        track_running_stats: bool = True,
        number_of_mlp_layers: int = 2,
        add_residual: bool = False,
        dropout: float = 0.0,
        learned: bool = True,
        train_eps: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_rank = input_rank
        self.output_rank = output_rank
        self.add_residual = add_residual
        if mlp is None:
            mlp = (
                get_mlp(
                    embedding_dim=embedding_dim,
                    batch_norm=batch_norm,
                    number_of_layers=number_of_mlp_layers,
                    activation=activation,
                )
                if learned
                else nn.Identity()
            )

        else:
            mlp = mlp

        if "gin" == conv_type:
            self.conv = GINConv(mlp, train_eps=train_eps)
        else:
            raise NotImplementedError(
                "Only GIN convs are currently implemented for IncidenceConv..."
            )

    def forward(self, data: Data) -> torch.Tensor:
        # Get data of input and output ranks
        x_s = getattr(data, f"x_{self.input_rank}")
        x_t = getattr(data, f"x_{self.output_rank}")

        # Get relevant upper / lower incidence
        incidence = (
            getattr(data, f"edge_index_incidence_{self.input_rank}_{self.output_rank}")
            if self.input_rank < self.output_rank
            else torch.flip(
                getattr(
                    data, f"edge_index_incidence_{self.output_rank}_{self.input_rank}"
                ),
                dims=(0,),
            )
        )

        # GIN forward pass (as bipartite graph)
        h = self.conv(x=(x_s, x_t), edge_index=incidence)
        if self.add_residual:
            x_t = h + x_t
        else:
            x_t = h
        return x_t


class AdjacencyConv(nn.Module):
    def __init__(
        self,
        input_rank: int,
        bridge_rank: int,
        embedding_dim: int,
        mlp: Optional[nn.Module] = None,  # overrides mlp construction if not None
        conv_type: str = "gin",
        batch_norm: bool = True,
        number_of_mlp_layers: int = 2,
        add_residual: bool = False,
        dropout: float = 0.0,
        train_eps: bool = True,
        activation: str = "relu",
    ):

        super().__init__()

        self.input_rank = input_rank
        self.bridge_rank = bridge_rank
        self.add_residual = add_residual

        mlp = (
            get_mlp(
                embedding_dim=embedding_dim,
                batch_norm=batch_norm,
                number_of_layers=number_of_mlp_layers,
                activation=activation,
            )
            if mlp is None
            else mlp
        )

        self.conv_type = conv_type
        if "gin" == conv_type:
            self.conv = GINConv(mlp, train_eps=train_eps)
        elif "gine" == conv_type:
            self.conv = GINEConv(mlp, train_eps=train_eps)
        elif "custom_gin" == conv_type:
            self.conv = CustomGINConv(mlp, train_eps=train_eps)

        else:
            raise NotImplementedError("Only GIN convs are currently implemented...")

    def forward(self, data: ComplexData) -> torch.Tensor:
        # Get data of relevant rank
        x = getattr(data, f"x_{self.input_rank}")

        # Get relevant adjacency/co-adjacency
        adjacency = (
            getattr(data, f"edge_index_adjacency_{self.input_rank}_{self.bridge_rank}")
            if self.input_rank < self.bridge_rank
            else getattr(
                data, f"edge_index_coadjacency_{self.input_rank}_{self.bridge_rank}"
            )
        )
        if self.conv_type in ["gine", "custom_gin"]:
            edge_attributes = self.get_edge_attributes(data)
            h = self.conv(x=x, edge_index=adjacency, edge_attr=edge_attributes)
        else:
            h = self.conv(x=x, edge_index=adjacency)

        if self.add_residual:
            x = h + x
        else:
            x = h
        return x

    def get_edge_attributes(self, data: ComplexData) -> torch.Tensor:
        # computes the edge attributes based on bridge cell features
        bridge_cell_values = getattr(data, f"x_{self.bridge_rank}")
        indices = (
            getattr(
                data, f"bridge_index_adjacency_{self.input_rank}_{self.bridge_rank}"
            )
            if self.input_rank < self.bridge_rank
            else getattr(
                data, f"edge_index_coadjacency_{self.input_rank}_{self.bridge_rank}"
            )
        )
        edge_attributes = bridge_cell_values[indices, :]
        return edge_attributes


class PointwiseConv(nn.Module):
    def __init__(
        self,
        rank: int,
        embedding_dim: int,
        mlp: Optional[nn.Module] = None,
        batch_norm: bool = True,
        number_of_mlp_layers: int = 2,
        add_residual: bool = False,
        activation: str = "relu",
    ):
        super().__init__()

        self.rank = rank
        self.add_residual = add_residual
        self.mlp = (
            get_mlp(
                embedding_dim=embedding_dim,
                batch_norm=batch_norm,
                number_of_layers=number_of_mlp_layers,
                activation=activation,
            )
            if mlp is None
            else mlp
        )

    def forward(self, data: Data) -> torch.Tensor:
        # Get data of input and output ranks
        x = getattr(data, f"x_{self.rank}")

        # GIN forward pass (as bipartite graph)
        h = self.mlp(x)
        if self.add_residual:
            x = h + x
        else:
            x = h
        return x


class EmbeddingAdapter(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int) -> None:
        super().__init__()

        self.adapter = nn.Linear(in_features=in_dim, out_features=out_dim)
        self.rank = rank

    def forward(self, data: ComplexData) -> torch.Tensor:
        return self.adapter(getattr(data, f"x_{self.rank}"))


class CustomGINConv(MessagePassing):
    """
    Gin which uses edge attributes by concatenation and linear map
    """

    def __init__(
        self,
        nn: torch.nn.Module,
        eps: float = 0.0,
        train_eps: bool = False,
        edge_dim: Optional[int] = None,
        **kwargs,
    ):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.empty(1))
        else:
            self.register_buffer("eps", torch.empty(1))

        nn = self.nn[0]
        if hasattr(nn, "in_features"):
            in_channels = nn.in_features
        elif hasattr(nn, "in_channels"):
            in_channels = nn.in_channels
        else:
            raise ValueError("Could not infer input channels from `nn`.")
        self.lin = (
            Linear(2 * in_channels, in_channels)
            if edge_dim is None
            else Linear(in_channels + edge_dim, in_channels)
        )
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        self.lin.reset_parameters()

    def forward(
        self,
        x: Union[Tensor, OptPairTensor],
        edge_index: Adj,
        edge_attr: OptTensor = None,
        size: Size = None,
    ) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)

        # propagate_type: (x: OptPairTensor, edge_attr: OptTensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out = out + (1 + self.eps) * x_r

        return self.nn(out)

    def message(self, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.lin(torch.concat([x_j, edge_attr], dim=-1)).relu()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(nn={self.nn})"


def get_mlp(
    embedding_dim: int,
    activation: str,
    number_of_layers: int,
    batch_norm: bool,
):
    layers = []
    activation = nn.ReLU() if activation == "relu" else nn.GELU()

    for _ in range(number_of_layers):
        layers.append(Linear(embedding_dim, embedding_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(embedding_dim))
        layers.append(activation),
    return Sequential(*layers)
