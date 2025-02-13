from typing import List
from models.layers.subcomplex import (
    SubComplexPool,
    SubComplexBroadcastLow,
    SubComplexIncidenceConv,
    SubComplexLowConv,
    SubComplexDistanceMarkingEmbed,
)
from models.tensor_diagram import TensorDiagram
from models.layers.merge_node import MergeNode
from models.embeddings import TwoCellEmbedding, ZeroEmbedding
from models.layers.homp import IncidenceConv, AdjacencyConv, PointwiseConv
import torch.nn as nn


def get_feature_embed_layer(embedding_dim: int = 94):
    return nn.ModuleDict(
        {
            "x_0": ZeroEmbedding(embedding_dim=embedding_dim, num_embeddings=1, rank=0),
            "x_1": ZeroEmbedding(embedding_dim=embedding_dim, num_embeddings=1, rank=1),
        }
    )


def get_two_cell_embed_layer(
    embedding_dim=94, number_of_mlp_layers: int = 2, learned: bool = False
):
    return nn.ModuleDict(
        {
            "x_2": TwoCellEmbedding(
                embedding_dim=embedding_dim,
                number_of_mlp_layers=number_of_mlp_layers,
                learned=learned,
            )
        }
    )


def get_subcomplex_pooling_layer(aggregation: str = "sum", high_rank: int = 2):
    return nn.ModuleDict(
        {
            "x_0": SubComplexPool(
                low_rank=0, high_rank=2, return_low_rank=True, aggregation=aggregation
            ),
            f"x_{high_rank}": SubComplexPool(
                low_rank=0,
                high_rank=high_rank,
                return_low_rank=False,
                aggregation=aggregation,
            ),
        }
    )


def get_subcomplex_embed_layer(
    embedding_dim=94,
    aggregation: str = "concatenate",
    activation: str = "relu",
    high_rank: int = 2,
):
    return nn.ModuleDict(
        {
            f"x_0_{high_rank}": MergeNode(
                [
                    SubComplexBroadcastLow(low_rank=0, high_rank=high_rank),
                    SubComplexDistanceMarkingEmbed(
                        low_rank=0, high_rank=high_rank, embed_dim=embedding_dim
                    ),
                ],
                aggregation=aggregation,
                embedding_dim=embedding_dim,
                activation=activation,
            ),
        }
    )


def get_subcomplex_layer(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    use_second_conv=False,
    second_conv_type=None,
    number_of_mlp_layers: int = 2,
    edge_dim: int = None,
    aggregation: str = "concatenate",
    activation: str = "relu",
    conv_type: str = "custom_gin",
    high_rank: int = 2,
):
    return nn.ModuleDict(
        {
            f"x_0_{high_rank}": MergeNode(
                [
                    SubComplexIncidenceConv(
                        low_rank=0,
                        high_rank=high_rank,
                        input_channels=input_dim,
                        output_channels=output_dim,
                        hidden_channels=hidden_dim,
                        use_second_conv=use_second_conv,
                        second_conv_type=second_conv_type,
                        number_of_mlp_layers=number_of_mlp_layers,
                        dropout=0.0,
                        train_eps=False,
                        activation=activation,
                    ),
                    SubComplexLowConv(
                        low_rank=0,
                        high_rank=high_rank,
                        input_channels=input_dim,
                        output_channels=output_dim,
                        hidden_channels=hidden_dim,
                        use_second_conv=use_second_conv,
                        second_conv_type=second_conv_type,
                        number_of_mlp_layers=number_of_mlp_layers,
                        edge_dim=edge_dim,
                        dropout=0.0,
                        train_eps=False,
                        activation=activation,
                        conv_type=conv_type,
                    ),
                ],
                aggregation=aggregation,
                embedding_dim=output_dim,
                activation=activation,
            ),
        }
    )


def get_cin_layer(
    embed_dim: int = 94,
    number_of_mlp_layers: int = 2,
    dropout: float = 0.0,
    aggregation: str = "concatenate",
    activation: str = "relu",
):
    return nn.ModuleDict(
        {
            "x_0": MergeNode(
                [
                    AdjacencyConv(
                        input_rank=0,
                        bridge_rank=1,
                        embedding_dim=embed_dim,
                        conv_type="custom_gin",
                        number_of_mlp_layers=number_of_mlp_layers,
                        dropout=dropout,
                        train_eps=False,
                        activation=activation,
                    ),
                    PointwiseConv(
                        rank=0,
                        embedding_dim=embed_dim,
                        number_of_mlp_layers=number_of_mlp_layers,
                        activation=activation,
                    ),
                ],
                aggregation=aggregation,
                embedding_dim=embed_dim,
                activation=activation,
            ),
            "x_1": MergeNode(
                [
                    IncidenceConv(
                        input_rank=0,
                        output_rank=1,
                        embedding_dim=embed_dim,
                        conv_type="gin",
                        dropout=dropout,
                        train_eps=False,
                        activation=activation,
                    ),
                    AdjacencyConv(
                        input_rank=1,
                        bridge_rank=2,
                        embedding_dim=embed_dim,
                        conv_type="custom_gin",
                        number_of_mlp_layers=number_of_mlp_layers,
                        dropout=dropout,
                        train_eps=False,
                        activation=activation,
                    ),
                ],
                aggregation=aggregation,
                embedding_dim=embed_dim,
                activation=activation,
            ),
            "x_2": MergeNode(
                [
                    IncidenceConv(
                        input_rank=1,
                        output_rank=2,
                        embedding_dim=embed_dim,
                        conv_type="gin",
                        number_of_mlp_layers=number_of_mlp_layers,
                        dropout=dropout,
                        train_eps=False,
                        activation=activation,
                    ),
                    PointwiseConv(
                        rank=2,
                        embedding_dim=embed_dim,
                        number_of_mlp_layers=number_of_mlp_layers,
                        activation=activation,
                    ),
                ],
                aggregation=aggregation,
                embedding_dim=embed_dim,
                activation=activation,
            ),
        }
    )


def build_homp_model(
    number_of_layers: int,
    embedding_dim: int,
    device: str,
    number_of_mlp_layers: int = 2,
    dropout: float = 0.0,
    final_dropout: float = 0.0,
    in_dropout: float = 0.0,
) -> TensorDiagram:

    # Start with the embedding layers
    layers = nn.ModuleList(
        [
            nn.ModuleDict(
                {
                    "x_0": ZeroEmbedding(
                        embedding_dim=embedding_dim, num_embeddings=1, rank=0
                    ),
                    "x_1": ZeroEmbedding(
                        embedding_dim=embedding_dim, num_embeddings=1, rank=1
                    ),
                }
            ),
            nn.ModuleDict(
                {
                    "x_2": TwoCellEmbedding(
                        embedding_dim=embedding_dim,
                        number_of_mlp_layers=number_of_mlp_layers,
                        learned=False,
                    )
                }
            ),
        ]
    )

    # Add HOMP blocks
    for _ in range(number_of_layers):
        layers.append(
            nn.ModuleDict(
                {
                    "x_0": MergeNode(
                        [
                            AdjacencyConv(
                                input_rank=0,
                                bridge_rank=1,
                                embedding_dim=embedding_dim,
                                conv_type="custom_gin",
                                number_of_mlp_layers=number_of_mlp_layers,
                                train_eps=False,
                            ),
                            PointwiseConv(
                                rank=0,
                                embedding_dim=embedding_dim,
                                number_of_mlp_layers=number_of_mlp_layers,
                            ),
                        ],
                        aggregation="concatenate",
                        embedding_dim=embedding_dim,
                    ),
                    "x_1": MergeNode(
                        [
                            IncidenceConv(
                                input_rank=0,
                                output_rank=1,
                                embedding_dim=embedding_dim,
                                conv_type="gin",
                                number_of_mlp_layers=number_of_mlp_layers,
                                train_eps=False,
                            ),
                            AdjacencyConv(
                                input_rank=1,
                                bridge_rank=2,
                                embedding_dim=embedding_dim,
                                conv_type="custom_gin",
                                number_of_mlp_layers=number_of_mlp_layers,
                                train_eps=False,
                            ),
                        ],
                        aggregation="concatenate",
                        embedding_dim=embedding_dim,
                    ),
                    "x_2": MergeNode(
                        [
                            IncidenceConv(
                                input_rank=1,
                                output_rank=2,
                                embedding_dim=embedding_dim,
                                conv_type="gin",
                                number_of_mlp_layers=number_of_mlp_layers,
                                train_eps=False,
                            ),
                            PointwiseConv(
                                rank=2,
                                embedding_dim=embedding_dim,
                                number_of_mlp_layers=number_of_mlp_layers,
                            ),
                        ],
                        aggregation="concatenate",
                        embedding_dim=embedding_dim,
                    ),
                }
            )
        )
    dropout_list = [in_dropout] + (len(layers) - 1) * [dropout]
    return TensorDiagram(
        layers=layers,
        embedding_dim=embedding_dim,
        output_dim=1,
        output_ranks=[0, 1, 2],
        device=device,
        dropout_list=dropout_list,
        aggregation="concatenate",
        zinc_head=False,
        final_dropout=final_dropout,
    )


def build_subcomplex_model(
    embedding_dim: int,
    number_subgraph_layers: int,
    device: str,
    output_ranks: List[int] = [0, 2],
    number_of_mlp_layers: int = 2,
    dropout: float = 0.0,
    in_dropout: float = 0.0,
    subgraph_embedding_aggregation: str = "concatenate",
    subgraph_layer_aggregation: str = "concatenate",
    residual: bool = False,
    activation: str = "relu",
    high_rank: int = 2,
) -> TensorDiagram:
    number_subgraph_layers += 2
    embedding_dim = round(embedding_dim * 1.5)

    layers = nn.ModuleList(
        [
            get_feature_embed_layer(embedding_dim=embedding_dim),
            get_two_cell_embed_layer(
                embedding_dim=embedding_dim, number_of_mlp_layers=number_of_mlp_layers
            ),
        ]
    )
    residual_list = [False, False]

    layers.append(
        get_subcomplex_embed_layer(
            embedding_dim=embedding_dim,
            aggregation=subgraph_embedding_aggregation,
            activation=activation,
            high_rank=high_rank,
        )
    )
    residual_list.append(False)

    for _ in range(number_subgraph_layers):
        layers.append(
            get_subcomplex_layer(
                input_dim=embedding_dim,
                hidden_dim=embedding_dim,
                output_dim=embedding_dim,
                number_of_mlp_layers=number_of_mlp_layers,
                edge_dim=embedding_dim,
                aggregation=subgraph_layer_aggregation,
                activation=activation,
                conv_type="gin",
                high_rank=high_rank,
            )
        )
        residual_list.append(residual)

    layers.append(
        get_subcomplex_pooling_layer(
            aggregation=subgraph_layer_aggregation, high_rank=high_rank
        )
    )

    residual_list.append(False)

    dropout_list = [in_dropout] * 2 + (len(layers) - 2) * [dropout]
    return TensorDiagram(
        layers=layers,
        embedding_dim=embedding_dim,
        output_dim=1,
        output_ranks=output_ranks,
        device=device,
        dropout_list=dropout_list,
        final_dropout=dropout,
        aggregation="sum",
        residuals=residual_list,
        activation=activation,
        zinc_head=False,
    )


def build_homp_no_edge_model(
    number_of_layers: int,
    embedding_dim: int,
    device: str,
    number_of_mlp_layers: int = 2,
    dropout: float = 0.0,
    final_dropout: float = 0.0,
    in_dropout: float = 0.0,
) -> TensorDiagram:
    # Start with the embedding layers
    layers = nn.ModuleList(
        [
            nn.ModuleDict({"x_0": ZeroEmbedding(embedding_dim=embedding_dim)}),
            nn.ModuleDict(
                {
                    "x_2": TwoCellEmbedding(
                        embedding_dim=embedding_dim,
                        number_of_mlp_layers=number_of_mlp_layers,
                        learned=False,
                    )
                }
            ),
        ]
    )

    # Add HOMP blocks
    for _ in range(number_of_layers):
        layers.append(
            nn.ModuleDict(
                {
                    "x_0": MergeNode(
                        [
                            AdjacencyConv(
                                input_rank=0,
                                bridge_rank=1,
                                embedding_dim=embedding_dim,
                                conv_type="gin",
                                number_of_mlp_layers=number_of_mlp_layers,
                                train_eps=False,
                            ),
                            IncidenceConv(
                                input_rank=2,
                                output_rank=0,
                                embedding_dim=embedding_dim,
                                conv_type="gin",
                                number_of_mlp_layers=number_of_mlp_layers,
                                train_eps=False,
                            ),
                            PointwiseConv(
                                rank=0,
                                embedding_dim=embedding_dim,
                                number_of_mlp_layers=number_of_mlp_layers,
                            ),
                        ],
                        aggregation="concatenate",
                        embedding_dim=embedding_dim,
                    ),
                    "x_2": MergeNode(
                        [
                            IncidenceConv(
                                input_rank=0,
                                output_rank=2,
                                embedding_dim=embedding_dim,
                                conv_type="gin",
                                number_of_mlp_layers=number_of_mlp_layers,
                                train_eps=False,
                            ),
                            PointwiseConv(
                                rank=2,
                                embedding_dim=embedding_dim,
                                number_of_mlp_layers=number_of_mlp_layers,
                            ),
                        ],
                        aggregation="concatenate",
                        embedding_dim=embedding_dim,
                    ),
                }
            )
        )
    dropout_list = 2 * [in_dropout] + (len(layers) - 2) * [dropout]
    return TensorDiagram(
        layers=layers,
        embedding_dim=embedding_dim,
        output_dim=1,
        output_ranks=[0, 2],
        device=device,
        dropout_list=dropout_list,
        aggregation="concatenate",
        zinc_head=False,
        final_dropout=final_dropout,
    )
