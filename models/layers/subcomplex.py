import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, MeanAggregation
from torch_geometric.nn.conv import GINConv
from torch_geometric.nn.aggr import SumAggregation
from typing import Tuple, Optional
from data.complex_data import ComplexData
from models.layers.homp import CustomGINConv


class SubComplexLowConv(nn.Module):
    """
    updates the x_i_j feature using the bag of graphs induced by the low rank cells
    """

    def __init__(
        self,
        low_rank: int,
        high_rank: int,
        input_channels: int,
        output_channels: int,
        hidden_channels: int = 16,
        number_of_mlp_layers: int = 2,
        conv_type: str = "custom_gin",
        use_batch_norm: bool = True,
        add_residual: bool = False,
        use_second_conv: bool = False,
        second_conv_type: Optional[str] = None,
        edge_dim: Optional[int] = None,
        dropout: float = 0.0,
        train_eps: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_channels = input_channels
        self.low_rank = low_rank
        self.high_rank = high_rank
        self.conv_type = conv_type
        self.edge_dim = edge_dim
        self.conv = self._get_conv(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            use_batch_norm=use_batch_norm,
            conv_type=conv_type,
            edge_dim=edge_dim,
            train_eps=train_eps,
            activation=activation,
        )
        self.use_second_conv = use_second_conv
        if use_second_conv:
            self.second_conv = self._get_second_conv(second_conv_type)

        self.bridge_rank = 1  # currently we only support low brifge rank = 1
        self.add_residual = add_residual

    def forward(self, data: Data) -> torch.Tensor:
        # Get data of input and output ranks
        x = getattr(data, f"x_{self.low_rank}_{self.high_rank}")

        edge_index = getattr(
            data,
            f"edge_index_low_adjacency_subcomplex_{self.low_rank}_{self.high_rank}",
        )
        if self.conv_type in ["gine", "custom_gin"]:
            edge_attributes = self.get_edge_attributes(data)
            h = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attributes)
        else:
            h = self.conv(x=x, edge_index=edge_index)

        x = x + h if self.add_residual else h
        x = self.second_conv(x=x, edge_index=edge_index) if self.use_second_conv else x
        return x

    def get_edge_attributes(self, data: ComplexData) -> torch.Tensor:
        # computes the edge attributes based on bridge cell features
        bridge_cell_values = getattr(data, f"x_{self.bridge_rank}")
        indices = getattr(
            data,
            f"bridge_index_low_adjacency_subcomplex_{self.low_rank}_{self.high_rank}",
        )
        edge_attributes = bridge_cell_values[indices, :]
        return edge_attributes

    @staticmethod
    def _get_conv(
        input_channels: int,
        hidden_channels: int,
        use_batch_norm: bool,
        conv_type: str,
        edge_dim: Optional[int] = None,
        train_eps: bool = True,
        activation: str = "relu",
    ):

        mlp = get_mlp(
            input_dim=input_channels,
            hidden_dim=hidden_channels,
            batch_norm=use_batch_norm,
            activation=activation,
        )

        if "gin" == conv_type:
            return GINConv(mlp, train_eps=train_eps)
        elif "gine" == conv_type:
            return GINEConv(mlp, train_eps=train_eps, edge_dim=edge_dim)
        elif "custom_gin" == conv_type:
            return CustomGINConv(mlp, train_eps=train_eps, edge_dim=edge_dim)
        else:
            raise NotImplementedError("Only GIN convs are currently implemented...")

    def _get_second_conv(self, conv_type: str):
        if "repeat" == conv_type:
            return self.conv
        elif "id" == conv_type:
            return GINConv(nn.Identity(), train_eps=True)
        else:
            raise NotImplementedError("not implemented...")


class SubComplexIncidenceConv(nn.Module):
    """
    updates the x_i_j feature using the bag of graphs induced by the low rank cells
    """

    def __init__(
        self,
        low_rank: int,
        high_rank: int,
        input_channels: int,
        output_channels: int,
        hidden_channels: int = 16,
        conv_type: str = "gin",
        use_batch_norm: bool = True,
        add_residual: bool = False,
        use_second_conv: bool = False,
        second_conv_type: Optional[str] = None,
        number_of_mlp_layers: int = 2,
        dropout: float = 0.0,
        train_eps: bool = True,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_channels = input_channels
        self.low_rank = low_rank
        self.high_rank = high_rank
        self.conv = self._get_conv(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            use_batch_norm=use_batch_norm,
            conv_type=conv_type,
            train_eps=train_eps,
            activation=activation,
        )
        self.conv_type = conv_type
        self.add_residual = add_residual

        self.use_second_conv = use_second_conv
        if use_second_conv:
            self.second_conv = self._get_second_conv(second_conv_type)

    def forward(self, data: Data) -> torch.Tensor:
        # Get data of input and output ranks
        x = getattr(data, f"x_{self.low_rank}_{self.high_rank}")
        edge_index = getattr(
            data, f"edge_index_incidence_subcomplex_{self.low_rank}_{self.high_rank}"
        )
        h = self.conv(x=x, edge_index=edge_index)
        x = x + h if self.add_residual else h
        x = self.second_conv(x=x, edge_index=edge_index) if self.use_second_conv else x
        return x

    @staticmethod
    def _get_conv(
        input_channels: int,
        hidden_channels: int,
        use_batch_norm: bool,
        conv_type: str,
        train_eps: bool = True,
        activation: str = "relu",
    ):
        mlp = get_mlp(
            input_dim=input_channels,
            hidden_dim=hidden_channels,
            batch_norm=use_batch_norm,
            activation=activation,
        )

        if "gin" == conv_type:
            return GINConv(mlp, train_eps=train_eps)

        else:
            raise NotImplementedError("Only GIN convs are currently implemented...")

    def _get_second_conv(self, conv_type: str):
        if "repeat" == conv_type:
            return self.conv
        elif "id" == conv_type:
            return GINConv(nn.Identity(), train_eps=True)
        else:
            raise NotImplementedError("not implemented...")


class SubComplexHighConv(nn.Module):
    """
    updates the x_i_j feature using the bag of graphs induced by the low rank cells
    """

    def __init__(
        self,
        low_rank: int,
        high_rank: int,
        input_channels: int,
        output_channels: int,
        hidden_channels: int = 16,
        conv_type: str = "custom_gin",
        use_batch_norm: bool = True,
        add_residual: bool = False,
        activation: str = "relu",
    ):
        super().__init__()
        self.input_channels = input_channels
        self.low_rank = low_rank
        self.high_rank = high_rank
        self.conv = self._get_conv(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            use_batch_norm=use_batch_norm,
            conv_type=conv_type,
            activation=activation,
        )
        self.bridge_rank = 0  # currently we only support high brifge rank = 0
        self.conv_type = conv_type
        self.add_residual = add_residual

    def forward(self, data: ComplexData) -> torch.Tensor:
        # Get data of input and output ranks
        x = getattr(data, f"x_{self.low_rank}_{self.high_rank}")
        edge_index = getattr(
            data,
            f"edge_index_high_adjacency_subcomplex_{self.low_rank}_{self.high_rank}",
        )
        if self.conv_type in ["gine", "custom_gin"]:
            edge_attributes = self.get_edge_attributes(data)
            h = self.conv(x=x, edge_index=edge_index, edge_attr=edge_attributes)
        else:
            h = self.conv(x=x, edge_index=edge_index)
        x = x + h if self.add_residual else h
        return x

    def get_edge_attributes(self, data: ComplexData) -> torch.Tensor:
        # computes the edge attributes based on bridge cell features
        bridge_cell_values = getattr(data, f"x_{self.bridge_rank}")
        indices = getattr(
            data,
            f"bridge_index_high_adjacency_subcomplex_{self.low_rank}_{self.high_rank}",
        )
        edge_attributes = bridge_cell_values[indices, :]
        return edge_attributes

    @staticmethod
    def _get_conv(
        input_channels: int,
        hidden_channels: int,
        use_batch_norm: bool,
        conv_type: str,
        activation: str = "relu",
    ):
        mlp = get_mlp(
            input_dim=input_channels,
            hidden_dim=hidden_channels,
            batch_norm=use_batch_norm,
            activation=activation,
        )
        if "gin" == conv_type:
            return GINConv(mlp, train_eps=True)
        elif "gine" == conv_type:
            return GINEConv(mlp, train_eps=True)
        elif "custom_gin" == conv_type:
            return CustomGINConv(mlp, train_eps=True)
        else:
            raise NotImplementedError("Only GIN convs are currently implemented...")


class SubComplexBroadcastLow(nn.Module):
    """
    brodcast cell features x_j to subcomplex featurs x_i_j where j<i
    """

    def __init__(
        self,
        low_rank: int,
        high_rank: int,
    ):
        super().__init__()
        low_rank, high_rank = min(low_rank, high_rank), max(low_rank, high_rank)
        self.low_rank = low_rank
        self.high_rank = high_rank

    def forward(self, data: Data) -> torch.Tensor:
        # get low rank cell features
        x = getattr(data, f"x_{self.low_rank}")

        # repeat each value to get subcomplex feature
        repeats = getattr(data, f"cells_high_repeats_{self.low_rank}_{self.high_rank}")
        x_repeated = torch.repeat_interleave(x, repeats.view(-1), dim=0)
        return x_repeated


class SubComplexBroadcastHigh(nn.Module):
    """
    brodcast cell features x_j to subcomplex featurs x_i_j
    """

    def __init__(
        self,
        low_rank: int,
        high_rank: int,
    ):
        super().__init__()
        low_rank, high_rank = min(low_rank, high_rank), max(low_rank, high_rank)
        self.low_rank = low_rank
        self.high_rank = high_rank

    def forward(self, data: Data) -> torch.Tensor:
        # get high rank cell features
        x = getattr(data, f"x_{self.high_rank}")

        # repeat each value  and rearange to get subcomplex features
        repeats = getattr(data, f"cells_low_repeats_{self.low_rank}_{self.high_rank}")

        alignment_indices_for_high_cell_features = getattr(
            data,
            f"cells_high_feature_alignment_subcomplex_{self.low_rank}_{self.high_rank}",
        )

        x_repeated = torch.repeat_interleave(x, repeats.view(-1), dim=0)
        x_repeated = x_repeated[alignment_indices_for_high_cell_features]
        return x_repeated


class SubComplexCrossPool(nn.Module):
    """
    pools subcomplex featurs x_i_j to cell features x_j based on x_i values
    """

    def __init__(self, low_rank: int, high_rank: int, aggregation: str = "mean"):
        super().__init__()
        self.low_rank = low_rank
        self.high_rank = high_rank
        self.first_pool = SubComplexPool(
            low_rank=low_rank,
            high_rank=high_rank,
            return_low_rank=True,
            aggregation=aggregation,
        )
        self.conv = GINConv(nn.Identity(), train_eps=False)

    def forward(self, data: Data) -> torch.Tensor:
        # get subcomplex features:
        x_s = self.first_pool(data)
        x_t = getattr(data, f"x_{self.high_rank}")

        # Get relevant upper / lower incidence
        incidence = getattr(
            data, f"edge_index_incidence_{self.low_rank}_{self.high_rank}"
        )

        # GIN forward pass (as bipartite graph)
        return self.conv(x=(x_s, x_t), edge_index=incidence)


class SubComplexPool(nn.Module):
    """
    pools subcomplex featurs x_i_j to cell features x_j
    """

    def __init__(
        self,
        low_rank: int,
        high_rank: int,
        return_low_rank: int = True,
        aggregation: str = "sum",
    ):
        super().__init__()
        self.low_rank = low_rank
        self.high_rank = high_rank
        self.return_low_rank = return_low_rank
        self.aggregation = (
            SumAggregation() if aggregation == "sum" else MeanAggregation()
        )

    def forward(self, data: Data) -> torch.Tensor:
        # get subcomplex features:
        x_subcomplex = getattr(data, f"x_{self.low_rank}_{self.high_rank}")

        # get which subcmplex nodes come from the same cell
        pool_index = (
            getattr(
                data, f"low_cell_indices_subcomplex_{self.low_rank}_{self.high_rank}"
            )
            if self.return_low_rank
            else getattr(
                data, f"high_cell_indices_subcomplex_{self.low_rank}_{self.high_rank}"
            )
        )

        # pad the pool index in case some of the last CCs have no high order cells
        num_cells_out = (
            getattr(data, f"num_cells_{self.low_rank}").sum().item()
            if self.return_low_rank
            else getattr(data, f"num_cells_{self.high_rank}").sum().item()
        )
        x_subcomplex, pool_index = pad_for_missing_cells(
            x_subcomplex=x_subcomplex, pool_index=pool_index, num_cells=num_cells_out
        )
        out = self.aggregation(x_subcomplex, pool_index)
        return out


class SubComplexBinaryMarkingEmbed(nn.Module):
    """
    embedding for binary encoding of subcomplex feautres
    """

    def __init__(self, low_rank: int, high_rank: int, embed_dim: int = 100):
        super().__init__()
        self.low_rank = low_rank
        self.high_rank = high_rank
        self.embed = nn.Embedding(2, embed_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x = getattr(data, f"x_node_marking_subcomplex_{self.low_rank}_{self.high_rank}")
        return self.embed(x.long()).squeeze()


class SubComplexDistanceMarkingEmbed(nn.Module):
    """
    embedding for distance encoding of subcomplex feautres
    """

    def __init__(
        self, low_rank: int, high_rank: int, embed_dim: int = 100, max_dist: int = 10
    ):
        super().__init__()
        self.low_rank = low_rank
        self.high_rank = high_rank
        self.max_dist = max_dist
        self.embed = nn.Embedding(max_dist + 2, embed_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x = getattr(data, f"x_node_marking_subcomplex_{self.low_rank}_{self.high_rank}")
        x = SubComplexDistanceMarkingEmbed.custom_clamp(
            x, min_val=None, max_val=self.max_dist
        )
        return self.embed(x.long()).squeeze()

    @staticmethod
    def custom_clamp(tensor, min_val, max_val):
        # First, clamp all values between min_val and max_val
        clamped_tensor = torch.clamp(tensor, min_val, max_val)

        # Create a mask for values that are greater than 1000 in the original tensor
        mask = tensor > 1000

        # Update the elements where the condition in 'mask' is True
        clamped_tensor[mask] = max_val + 1

        return clamped_tensor


def pad_for_missing_cells(
    x_subcomplex, pool_index, num_cells
) -> Tuple[torch.Tensor, torch.Tensor]:
    max_index_nonempty = 0 if pool_index.numel() == 0 else pool_index.max().item()
    if max_index_nonempty < num_cells - 1:
        pool_index = torch.cat(
            [pool_index, torch.tensor([num_cells - 1]).to(pool_index.device)]
        )
        x_subcomplex = F.pad(x_subcomplex, (0, 0, 0, 1), "constant", 0)
    return x_subcomplex, pool_index


def get_mlp(
    input_dim: int, hidden_dim: int, batch_norm: bool, activation: str = "relu"
):
    layers = []
    input_dim, hidden_dim = input_dim, hidden_dim
    activation = nn.ReLU() if activation == "relu" else nn.GELU()

    for _ in range(2):
        layers.append(Linear(input_dim, hidden_dim))
        if batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(activation),
        input_dim = hidden_dim
    return Sequential(*layers)
