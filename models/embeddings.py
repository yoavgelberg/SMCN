import torch
import torch.nn as nn
from torch_geometric.data import Data
from data.complex_data import ComplexData
from models.layers.homp import IncidenceConv
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


class ZeroEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, rank):
        super(ZeroEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.rank = rank

    def forward(self, data):
        x = torch.zeros_like(getattr(data, f"x_{self.rank}"))
        return self.embedding(x.long()).squeeze()


class AtomEmbedding(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super().__init__()
        self.atom_encoder = AtomEncoder(embedding_dim)

    def forward(self, data: ComplexData) -> torch.Tensor:
        return self.atom_encoder(data.x_0.long())


class BondEmbedding(nn.Module):
    def __init__(self, embedding_dim) -> None:
        super().__init__()
        self.bond_encoder = BondEncoder(embedding_dim)

    def forward(self, data: ComplexData) -> torch.Tensor:
        return self.bond_encoder(data.x_1.long())


class TwoCellEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim,
        dropout: float = 0.0,
        number_of_mlp_layers: int = 2,
        learned: bool = True,
    ) -> None:
        super().__init__()
        self.two_cell_encoder = (
            IncidenceConv(
                input_rank=0,
                output_rank=2,
                embedding_dim=embedding_dim,
                dropout=dropout,
                number_of_mlp_layers=number_of_mlp_layers,
            )
            if learned
            else IncidenceConv(
                input_rank=0,
                output_rank=2,
                embedding_dim=embedding_dim,
                learned=learned,
                train_eps=False,
            )
        )

    def forward(self, data: ComplexData) -> torch.Tensor:
        return self.two_cell_encoder(data)


class DistanceEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Build encoder layer

    def forward(self, data: Data) -> Data:
        pass
