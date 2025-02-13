import torch.nn as nn
from models.layers.subcomplex import (
    SubComplexPool,
    SubComplexBroadcastLow,
    SubComplexBroadcastHigh,
    SubComplexBinaryMarkingEmbed,
    SubComplexIncidenceConv,
    SubComplexLowConv,
    SubComplexHighConv,
    SubComplexDistanceMarkingEmbed,
)
from models.tensor_diagram import TensorDiagram
from models.layers.merge_node import MergeNode
from models.embeddings import AtomEmbedding, BondEmbedding, TwoCellEmbedding
from models.layers.homp import IncidenceConv, AdjacencyConv, EmbeddingAdapter

from typing import Optional


def get_feature_embed_layer(embedding_dim: int = 94):
    return nn.ModuleDict(
        {
            "x_0": AtomEmbedding(embedding_dim=embedding_dim),
            "x_1": BondEmbedding(embedding_dim=embedding_dim),
        }
    )


def get_two_cell_embed_layer(embedding_dim=94):
    return nn.ModuleDict({"x_2": TwoCellEmbedding(embedding_dim=embedding_dim)})


def get_subcomplex_pooling_layer(high_rank: int = 1):
    return nn.ModuleDict(
        {
            "x_0": SubComplexPool(
                low_rank=0,
                high_rank=high_rank,
                return_low_rank=True,
            ),
            f"x_{high_rank}": SubComplexPool(
                low_rank=0, high_rank=high_rank, return_low_rank=False
            ),
        }
    )


def get_marking_embedding_layer(distance: bool, embedding_dim: int, high_rank: int = 1):
    if distance:
        return SubComplexDistanceMarkingEmbed(
            low_rank=0, high_rank=high_rank, embed_dim=embedding_dim
        )
    return (
        SubComplexBinaryMarkingEmbed(
            low_rank=0, high_rank=high_rank, embed_dim=embedding_dim
        ),
    )


def get_subcomplex_embed_layer(
    embedding_dim=94, high_rank: int = 1, distance: bool = True
):
    marking_embedding_layer = get_marking_embedding_layer(
        distance=distance, embedding_dim=embedding_dim, high_rank=high_rank
    )
    return nn.ModuleDict(
        {
            f"x_0_{high_rank}": MergeNode(
                [
                    SubComplexBroadcastLow(low_rank=0, high_rank=high_rank),
                    SubComplexBroadcastHigh(low_rank=0, high_rank=high_rank),
                    marking_embedding_layer,
                ],
                aggregation="concatenate",
                embedding_dim=embedding_dim,
            ),
        }
    )


def get_subcomplex_layer(
    input_dim: int,
    output_dim: int,
    hidden_dim: int,
    add_residual: bool = False,
    use_second_conv=False,
    second_conv_type=None,
    number_of_mlp_layers: int = 2,
    edge_dim: int = None,
    high_rank: int = 1,
    distance: bool = True,
    aggregation: str = "sum",
):

    marking_embedding_layer = get_marking_embedding_layer(
        distance=distance, embedding_dim=output_dim, high_rank=high_rank
    )

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
                        add_residual=add_residual,
                        use_second_conv=use_second_conv,
                        second_conv_type=second_conv_type,
                        number_of_mlp_layers=number_of_mlp_layers,
                    ),
                    SubComplexLowConv(
                        low_rank=0,
                        high_rank=high_rank,
                        input_channels=input_dim,
                        output_channels=output_dim,
                        hidden_channels=hidden_dim,
                        add_residual=add_residual,
                        use_second_conv=use_second_conv,
                        second_conv_type=second_conv_type,
                        number_of_mlp_layers=number_of_mlp_layers,
                        edge_dim=edge_dim,
                    ),
                    marking_embedding_layer,
                ],
                aggregation=aggregation,
                embedding_dim=output_dim,
                add_residual=add_residual,
            ),
        }
    )


def get_cin_layer(
    embed_dim: int = 94, add_residual: bool = False, number_of_mlp_layers: int = 2
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
                        add_residual=add_residual,
                        number_of_mlp_layers=number_of_mlp_layers,
                    )
                ],
                aggregation="concatenate",
                embedding_dim=embed_dim,
            ),
            "x_1": MergeNode(
                [
                    IncidenceConv(
                        input_rank=0,
                        output_rank=1,
                        embedding_dim=embed_dim,
                        conv_type="gin",
                        add_residual=add_residual,
                    ),
                    AdjacencyConv(
                        input_rank=1,
                        bridge_rank=2,
                        embedding_dim=embed_dim,
                        conv_type="custom_gin",
                        add_residual=add_residual,
                        number_of_mlp_layers=number_of_mlp_layers,
                    ),
                    # IdentityConv(input_rank=1)
                ],
                aggregation="concatenate",
                embedding_dim=embed_dim,
            ),
            "x_2": MergeNode(
                [
                    IncidenceConv(
                        input_rank=1,
                        output_rank=2,
                        embedding_dim=embed_dim,
                        conv_type="gin",
                        add_residual=add_residual,
                        number_of_mlp_layers=number_of_mlp_layers,
                    ),
                ],
                aggregation="concatenate",
                embedding_dim=embed_dim,
            ),
        }
    )


def get_reduced_cin_layer(
    embed_dim: int, max_rank, add_residual: bool = False, number_of_mlp_layers: int = 2
):
    layer = nn.ModuleDict()
    if max_rank >= 0:
        layer["x_0"] = MergeNode(
            [
                AdjacencyConv(
                    input_rank=0,
                    bridge_rank=1,
                    embedding_dim=embed_dim,
                    conv_type="custom_gin",
                    add_residual=add_residual,
                    number_of_mlp_layers=number_of_mlp_layers,
                )
            ],
            aggregation="concatenate",
            embedding_dim=embed_dim,
        )
    if max_rank >= 1:
        layer["x_1"] = MergeNode(
            [
                IncidenceConv(
                    input_rank=0,
                    output_rank=1,
                    embedding_dim=embed_dim,
                    conv_type="gin",
                    add_residual=add_residual,
                    number_of_mlp_layers=number_of_mlp_layers,
                ),
                AdjacencyConv(
                    input_rank=1,
                    bridge_rank=2,
                    embedding_dim=embed_dim,
                    conv_type="custom_gin",
                    add_residual=add_residual,
                    number_of_mlp_layers=number_of_mlp_layers,
                ),
                # IdentityConv(input_rank=1)
            ],
            aggregation="concatenate",
            embedding_dim=embed_dim,
        )
    if max_rank >= 2:
        layer["x_2"] = MergeNode(
            [
                IncidenceConv(
                    input_rank=1,
                    output_rank=2,
                    embedding_dim=embed_dim,
                    conv_type="gin",
                    add_residual=add_residual,
                    number_of_mlp_layers=number_of_mlp_layers,
                ),
            ],
            aggregation="concatenate",
            embedding_dim=embed_dim,
        )

    return layer


def build_sequential_subcomplex_model(
    cin_embedding_dim: int,
    subcomplex_embedding_dim: int,
    number_cin_layers_top: int,
    number_cin_layers_bottom: int,
    number_subgraph_layers: int,
    max_output_rank: int,
    device: str,
    add_residual: bool = False,
    use_second_conv: bool = False,
    second_conv_type: Optional[str] = None,
    number_of_mlp_layers: int = 2,
    high_rank: int = 1,
) -> TensorDiagram:
    """
    sequential tenssor diagram used for ZINC
    """

    layers = nn.ModuleList(
        [
            get_feature_embed_layer(embedding_dim=cin_embedding_dim),
            get_two_cell_embed_layer(embedding_dim=cin_embedding_dim),
        ]
    )

    for _ in range(number_cin_layers_top):
        layers.append(
            get_cin_layer(
                embed_dim=cin_embedding_dim,
                add_residual=add_residual,
                number_of_mlp_layers=number_of_mlp_layers,
            )
        )

    layers.append(
        get_subcomplex_embed_layer(embedding_dim=cin_embedding_dim, high_rank=high_rank)
    )

    layers.append(
        get_subcomplex_layer(
            input_dim=cin_embedding_dim,
            hidden_dim=subcomplex_embedding_dim,
            output_dim=subcomplex_embedding_dim,
            add_residual=add_residual,
            use_second_conv=use_second_conv,
            second_conv_type=second_conv_type,
            number_of_mlp_layers=number_of_mlp_layers,
            edge_dim=cin_embedding_dim,
            high_rank=high_rank,
            aggregation="sum",
        )
    )
    for _ in range(number_subgraph_layers - 2):
        layers.append(
            get_subcomplex_layer(
                input_dim=subcomplex_embedding_dim,
                hidden_dim=subcomplex_embedding_dim,
                output_dim=subcomplex_embedding_dim,
                add_residual=add_residual,
                use_second_conv=use_second_conv,
                second_conv_type=second_conv_type,
                number_of_mlp_layers=number_of_mlp_layers,
                edge_dim=cin_embedding_dim,
                high_rank=high_rank,
                aggregation="sum",
            )
        )
    layers.append(
        get_subcomplex_layer(
            input_dim=subcomplex_embedding_dim,
            hidden_dim=cin_embedding_dim,
            output_dim=cin_embedding_dim,
            add_residual=add_residual,
            use_second_conv=use_second_conv,
            second_conv_type=second_conv_type,
            number_of_mlp_layers=number_of_mlp_layers,
            edge_dim=cin_embedding_dim,
            high_rank=high_rank,
            aggregation="sum",
        )
    )

    layers.append(get_subcomplex_pooling_layer(high_rank=high_rank))

    # layers.append(
    #     get_adapter_layer(in_dim=subcomplex_embedding_dim, out_dim=cin_embedding_dim)
    # )

    for _ in range(number_cin_layers_bottom):
        layers.append(
            get_cin_layer(
                embed_dim=cin_embedding_dim,
                add_residual=add_residual,
                number_of_mlp_layers=number_of_mlp_layers,
            )
        )

    layers.append(
        get_reduced_cin_layer(
            embed_dim=cin_embedding_dim,
            max_rank=max_output_rank,
            add_residual=add_residual,
            number_of_mlp_layers=number_of_mlp_layers,
        )
    )

    return TensorDiagram(
        layers=layers,
        embedding_dim=cin_embedding_dim,
        output_dim=1,
        output_ranks=list(range(max_output_rank + 1)),
        device=device,
    ).to(device)


def build_homp_model(
    number_of_blocks: int,
    embedding_dim: int,
    device: str,
    number_of_mlp_layers: int = 2,
) -> TensorDiagram:
    # Start with the embedding layers
    layers = nn.ModuleList(
        [
            nn.ModuleDict(
                {
                    "x_0": AtomEmbedding(embedding_dim=embedding_dim),
                    "x_1": BondEmbedding(embedding_dim=embedding_dim),
                }
            ),
            nn.ModuleDict({"x_2": TwoCellEmbedding(embedding_dim=embedding_dim)}),
        ]
    )

    # Add HOMP blocks
    for _ in range(number_of_blocks):
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
                            )
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
                            ),
                            AdjacencyConv(
                                input_rank=1,
                                bridge_rank=2,
                                embedding_dim=embedding_dim,
                                conv_type="custom_gin",
                                number_of_mlp_layers=number_of_mlp_layers,
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
                            )
                        ],
                        aggregation="concatenate",
                        embedding_dim=embedding_dim,
                    ),
                }
            )
        )

    return TensorDiagram(
        layers=layers,
        embedding_dim=embedding_dim,
        output_dim=1,
        output_ranks=[0, 1, 2],
        device=device,
    )
