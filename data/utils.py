from typing import List, Tuple, Dict
import torch
import torch_geometric
from scipy.sparse import csr_matrix
from toponetx.classes import CombinatorialComplex as CCC
import numpy as np
from torch_geometric.utils import to_dense_adj


class Utils:
    @staticmethod
    def _get_cell_indices(cc: CCC) -> Tuple[List[frozenset], List[int]]:
        """
        Gets a list of each cell in the ccc as a forzen set of node indices and a 
        corresponding list of rank of each cell.
        """
        cells = []
        cell_ranks = []
        ranks = cc.ranks
        for rank in ranks[1:]:
            col, row, incidence = cc.incidence_matrix(0, rank, index=True)
            if rank == ranks[1]:
                for cell in col:
                    cells.append(cell)
                    cell_ranks.append(0)
            for cell in row:
                cells.append(cell)
                cell_ranks.append(rank)
        return cells, cell_ranks

    @staticmethod
    def _transform_to_adjacency(matrix: csr_matrix) -> torch.Tensor:
        if isinstance(matrix, np.ndarray):
            matrix = csr_matrix(matrix)
        row_indices, col_indices = matrix.nonzero()
        edge_index = np.vstack((row_indices, col_indices))
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        return edge_index

    @staticmethod
    def construct_subcomplex_features(
        x_low: torch.Tensor,
        x_high: torch.Tensor,
        incidence: torch.Tensor,
        low_adjacency: torch.Tensor,
        low_bridge: torch.Tensor,
        high_adjacency: torch.Tensor,
        high_bridge: torch.Tensor,
        binary_marking: bool = False,
    ) -> Dict[str, torch.Tensor]:

        subcomplex_features = {}
        num_cells_low = x_low.size(0)
        num_cells_high = x_high.size(0)

        # a tensor indicating what high rank cell corresponds to each node in the 
        # product graph,  shape: (num_cells_low * num_cells_high, 1)
        subcomplex_features["high_cell_indices_subcomplex"] = torch.cat(
            [torch.arange(num_cells_high)] * num_cells_low
        ).long()

        # a tensor indicating what low rank cell corresponds to each node in the product 
        # graph,  shape: (num_cells_low * num_cells_high, 1)
        subcomplex_features["low_cell_indices_subcomplex"] = (
            torch.cat(
                [
                    torch.full(size=(num_cells_high, 1), fill_value=i)
                    for i in range(num_cells_low)
                ]
            )
            .squeeze()
            .long()
        )

        # a tensor which equals zero if the low rank cell corresponds to a 
        # subcomplex-node is a subset in the high rank cell corresponds to a 
        # subcomplex-node,  shape: (num_cells_low * num_cells_high, 1)
        subcomplex_features["x_node_marking_subcomplex"] = (
            Utils.get_subcomplex_binary_node_marking(
                incidence=incidence, num_cells_low=num_cells_low
            )
            if binary_marking
            else Utils.get_subcomplex_distance_node_marking(
                incidence=incidence,
                num_cells_low=num_cells_low,
                num_cells_high=num_cells_high,
                edge_index_low=low_adjacency,
            )
        )

        # repeats the num_cells_low, the number of repetitions is  num_cells_high
        subcomplex_features["cells_low_repeats"] = Utils.get_num_cells_low_repeated(
            num_cells_low=num_cells_low, num_cells_high=num_cells_high
        )

        # repeats the num_cells_high, the number of repetitions is  num_cells_low
        subcomplex_features["cells_high_repeats"] = Utils.get_num_cells_high_repeated(
            num_cells_low=num_cells_low, num_cells_high=num_cells_high
        )

        # a tensor of indices which tells you how to rearange the values of 
        # torch.repeat_interleave(x_high, low_repeats.view(-1), dim=0)
        # so that it would asign the correct values in the product graph. 
        # shape: (num_cells_low * num_cells_high, 1)
        subcomplex_features["cells_high_feature_alignment_subcomplex"] = (
            Utils.get_cells_high_feature_alignment_subcomplex(
                num_cells_low=num_cells_low, num_cells_high=num_cells_high
            )
        )

        # the edge index matrix induced by low adjacency on the product graph. 
        # shape: (2, num_celss_high* len(low_adjacency.t()))
        (
            subcomplex_features["bridge_index_low_adjacency_subcomplex"],
            subcomplex_features["edge_index_low_adjacency_subcomplex"],
        ) = Utils.get_edge_index_low_adjacency_subcomplex(
            low_adjacency=low_adjacency,
            bridge_index=low_bridge,
            num_cells_high=num_cells_high,
        )

        # the edge index matrix induced by high adjacency on the product graph. 
        # shape: (2, num_celss_low* len(high_adjacency.t()))
        (
            subcomplex_features["bridge_index_high_adjacency_subcomplex"],
            subcomplex_features["edge_index_high_adjacency_subcomplex"],
        ) = Utils.get_edge_index_high_adjacency_subcomplex(
            high_adjacency=high_adjacency,
            bridge_index=high_bridge,
            num_cells_high=num_cells_high,
            num_cells_low=num_cells_low,
        )

        # the edge index matrix induced by incidence on the product graph. 
        # shape: (2, num_celss_high* len(incidence.t()))
        subcomplex_features["edge_index_incidence_subcomplex"] = (
            Utils.get_edge_index_incidence_subcomplex(incidence, num_cells_high)
        )

        return subcomplex_features

    @staticmethod
    def get_edge_index_low_adjacency_subcomplex(
        low_adjacency: torch.Tensor, bridge_index: torch.Tensor, num_cells_high: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shifted_adjacencies = [
            low_adjacency * num_cells_high + i for i in range(num_cells_high)
        ]
        shifted_bridge_index = [bridge_index for _ in range(num_cells_high)]
        if num_cells_high > 0:
            return (
                torch.cat(shifted_bridge_index).long(),
                torch.cat(shifted_adjacencies, dim=-1).long(),
            )
        else:
            return torch.empty(0).long(), torch.empty(2, 0).long()

    @staticmethod
    def get_edge_index_high_adjacency_subcomplex(
        high_adjacency: torch.Tensor,
        bridge_index: torch.Tensor,
        num_cells_low: int,
        num_cells_high: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        shifted_adjacencies = [
            high_adjacency + i * num_cells_high for i in range(num_cells_low)
        ]
        shifted_bridge_index = [bridge_index for _ in range(num_cells_low)]
        return (
            torch.cat(shifted_bridge_index, dim=-1).long(),
            torch.cat(shifted_adjacencies, dim=-1).long(),
        )

    @staticmethod
    def get_edge_index_incidence_subcomplex(
        incidence: torch.Tensor, num_cells_high: int
    ) -> torch.Tensor:

        if num_cells_high == 0:
            return torch.empty(2, 0).long()
        index_shift = torch.arange(num_cells_high).view(1, -1)
        shifted_adjacencies = []
        for low_cell_index, high_cell_index in incidence.t():
            marked_index = torch.full(
                size=(1, num_cells_high),
                fill_value=high_cell_index + (num_cells_high * low_cell_index),
            )
            target_index = (num_cells_high * low_cell_index) + index_shift

            shifted_adjacency = torch.cat([marked_index, target_index], dim=0)
            shifted_adjacencies.append(shifted_adjacency)

        return torch.cat(shifted_adjacencies, dim=-1).long()

    @staticmethod
    def get_subcomplex_binary_node_marking(
        incidence: torch.Tensor, num_cells_low: int
    ) -> torch.Tensor:
        dense_incidence = Utils.to_bipartie_dese_adj(
            num_first=num_cells_low, incidence=incidence
        )
        x_node_marking = dense_incidence.reshape(
            -1, 1
        )  # This is the same as torch.cat([dense_incidence[i,:] for i in range(num_cells_low)], dim=0).unsqueeze(1)
        return x_node_marking

    @staticmethod
    def get_subcomplex_distance_node_marking(
        incidence: torch.Tensor,
        edge_index_low: torch.Tensor,
        num_cells_low: int,
        num_cells_high: int,
    ) -> torch.Tensor:
        low_rank_spd = Utils.get_all_pairs_shortest_paths(
            edge_index_low, max_num_nodes=num_cells_low
        )
        if num_cells_high == 0:
            return torch.empty(0, 1)
        spd_encode_list = []

        for i in range(num_cells_high):
            low_rank_indices = incidence[0][incidence[1] == i]
            spd_cell, _ = torch.min(low_rank_spd[:, low_rank_indices], dim=1)

            spd_encode_list.append(spd_cell.view(-1, 1))
        return torch.cat(spd_encode_list, dim=1).reshape(-1, 1)

    @staticmethod
    def get_cells_high_feature_alignment_subcomplex(
        num_cells_low: int, num_cells_high: int
    ):
        return torch.cat(
            [
                torch.arange(num_cells_high) * num_cells_low + i
                for i in range(num_cells_low)
            ]
        ).long()

    @staticmethod
    def get_num_cells_low_repeated(
        num_cells_low: int, num_cells_high: int
    ) -> torch.Tensor:
        return torch.tensor([num_cells_low] * num_cells_high).reshape(-1, 1).long()

    @staticmethod
    def get_num_cells_high_repeated(
        num_cells_low: int, num_cells_high: int
    ) -> torch.Tensor:
        return torch.tensor([num_cells_high] * num_cells_low).reshape(-1, 1).long()

    @staticmethod
    def to_bipartie_dese_adj(
        num_first: torch.Tensor, incidence: torch.Tensor
    ) -> torch.Tensor:
        dense_incidence = to_dense_adj(
            incidence + torch.tensor([0, num_first]).view(-1, 1)
        ).squeeze()
        return dense_incidence[:num_first, num_first:]

    @staticmethod
    def get_all_pairs_shortest_paths(
        edge_index, max_num_nodes: int, imputing_val: int = 1001
    ):
        """
        input is an adjacency of the original complex. Computes spd on this as graph
        """
        adj = torch_geometric.utils.to_dense_adj(
            edge_index, max_num_nodes=max_num_nodes
        ).squeeze(0)

        spd = torch.where(
            ~torch.eye(len(adj), dtype=bool) & (adj == 0),
            torch.full_like(adj, imputing_val),
            adj,
        )
        # Floyd-Warshall

        for k in range(len(spd)):
            dist_from_source_to_k = spd[:, [k]]
            dist_from_k_to_target = spd[[k], :]
            dist_from_source_to_target_via_k = (
                dist_from_source_to_k + dist_from_k_to_target
            )
            spd = torch.minimum(spd, dist_from_source_to_target_via_k)
        return spd
