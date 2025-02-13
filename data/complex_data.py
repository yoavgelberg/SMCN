from typing import List, Optional, Tuple
from torch_geometric.data import Data
import torch
from data.utils import Utils
from toponetx.classes import CombinatorialComplex as CC
import numpy as np
from torch_geometric.utils import to_dense_adj
import torch.nn.functional as F
from scipy.sparse._csr import csr_matrix


class ComplexData(Data):
    @classmethod
    def create_from_cc(
        cls,
        cc: CC,
        cell_features: Optional[List[torch.Tensor]] = None,
        y: Optional[List[torch.Tensor]] = None,
        dim: Optional[int] = None,
    ):
        dim = cc.ranks[-1] if dim is None else dim  # maximal rank of complex
        ranks = Utils._get_cell_indices(
            cc
        )  # the rank of each cell where cells are represented as frozen dicts

        cell_features = (
            cell_features if cell_features else []
        )  # list of all current cell features

        # if we don't have cell features for some ranks set them to zero
        num_cell_features = len(cell_features)
        if num_cell_features < dim + 1:
            for rk in range(num_cell_features, dim + 1):
                num_cells = int((np.array(ranks) == rk).sum())
                x = torch.zeros(size=[num_cells]).reshape(-1, 1)
                cell_features.append(x)

        # add cell featores and num of cells of rank i
        attributes = {"num_nodes": cc.number_of_nodes(), "dim": dim, "y": y}
        for i, cell_feature in enumerate(cell_features):
            attributes[f"x_{i}"] = cell_feature
            attributes[f"num_cells_{i}"] = torch.tensor([cell_feature.size(0)])

        # add connectivity induced by neighborhoods
        for rk1 in range(dim):
            for rk2 in range(rk1 + 1, dim + 1):
                incidence_matrix = cc.incidence_matrix(rk1, rk2)

                incidence = Utils._transform_to_adjacency(incidence_matrix)

                bridge_index_adjacency, adjacency = get_adjacency_from_incidence(
                    incidence_matrix
                )
                bridge_index_coadjacency, coadjacency = get_adjacency_from_incidence(
                    incidence_matrix.T
                )  # check if .T is necessary

                attributes[f"edge_index_incidence_{rk1}_{rk2}"] = incidence
                attributes[f"edge_index_adjacency_{rk1}_{rk2}"] = adjacency
                attributes[f"bridge_index_adjacency_{rk1}_{rk2}"] = (
                    bridge_index_adjacency
                )
                attributes[f"edge_index_coadjacency_{rk2}_{rk1}"] = coadjacency
                attributes[f"bridge_index_coadjacency_{rk2}_{rk1}"] = (
                    bridge_index_coadjacency
                )

        return ComplexData(**attributes)

    @classmethod
    def create_from_data(cls, data: Data):
        return ComplexData(**data.__dict__["_store"])

    def __inc__(self, key, value, *args, **kwargs):
        for rk1 in range(self.dim):
            for rk2 in range(rk1 + 1, self.dim + 1):
                if key == f"edge_index_incidence_{rk1}_{rk2}":
                    x_row = getattr(self, f"x_{rk1}")
                    x_col = getattr(self, f"x_{rk2}")

                    return torch.tensor([[x_row.size(0)], [x_col.size(0)]])

                elif key == f"edge_index_adjacency_{rk1}_{rk2}":
                    x = getattr(self, f"x_{rk1}")
                    return x.size(0)

                elif key == f"bridge_index_adjacency_{rk1}_{rk2}":
                    x = getattr(self, f"x_{rk2}")
                    return x.size(0)

                elif key == f"edge_index_coadjacency_{rk2}_{rk1}":
                    x = getattr(self, f"x_{rk2}")
                    return x.size(0)

                elif key == f"bridge_index_coadjacency_{rk2}_{rk1}":
                    x = getattr(self, f"x_{rk1}")
                    return x.size(0)

                elif f"subcomplex_{rk1}_{rk2}" in key and "edge" in key:
                    x_row = getattr(self, f"x_{rk1}")
                    x_col = getattr(self, f"x_{rk2}")
                    return x_row.size(0) * x_col.size(0)

                elif key == f"low_cell_indices_subcomplex_{rk1}_{rk2}":
                    return getattr(self, f"x_{rk1}").size(0)

                elif key == f"high_cell_indices_subcomplex_{rk1}_{rk2}":
                    return getattr(self, f"x_{rk2}").size(0)

                elif key == f"bridge_index_low_adjacency_subcomplex_{rk1}_{rk2}":
                    return getattr(self, f"x_{1}").size(0)

                elif key == f"bridge_index_high_adjacency_subcomplex_{rk1}_{rk2}":
                    return getattr(self, f"x_{0}").size(0)

        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if "num_cells" in key:
            return None
        return super().__cat_dim__(key, value, *args, **kwargs)

    def get_permuted_copies(self, num_copies: int):
        """
        Gets permuted copies of the CC, does not take into account multi cell 
        features. For that you need to reuse compute_multi_cell_feature
        """

        # compute number of cells in each rank
        number_of_cells = {
            f"num_cells_{rk}": getattr(self, f"num_cells_{rk}")
            for rk in range(self.dim + 1)
        }

        copies = []
        for i in range(num_copies):
            # create data instance
            perm_data = ComplexData()

            # get random permutation per rank
            perm_dict = {
                f"perm_{rk}": torch.randperm(number_of_cells[f"num_cells_{rk}"])
                for rk in range(self.dim + 1)
            }

            # Set dim and num cells
            perm_data.dim = self.dim
            perm_data.num_nodes = number_of_cells["num_cells_0"]
            for key, val in number_of_cells.items():
                setattr(perm_data, key, val)

            # permute cell features
            for rk in range(self.dim + 1):
                perm = perm_dict[f"perm_{rk}"]
                x = getattr(self, f"x_{rk}")
                setattr(perm_data, f"x_{rk}", x[perm])

            # permute neighborhood functions:
            for rk1 in range(self.dim):
                for rk2 in range(rk1 + 1, self.dim + 1):
                    perm1 = perm_dict[f"perm_{rk1}"]
                    perm2 = perm_dict[f"perm_{rk2}"]

                    # permute incidence
                    incidence = getattr(self, f"edge_index_incidence_{rk1}_{rk2}")
                    incidence_perm = torch.stack(
                        [perm1[incidence[0]], perm2[incidence[1]]]
                    )
                    setattr(
                        perm_data, f"edge_index_incidence_{rk1}_{rk2}", incidence_perm
                    )
                    # permute adjacency
                    adjacency = getattr(self, f"edge_index_adjacency_{rk1}_{rk2}")
                    adjacency_perm = perm1[adjacency]
                    setattr(
                        perm_data, f"edge_index_adjacency_{rk1}_{rk2}", adjacency_perm
                    )

                    # permute bridge_index_adjacency
                    bridge_index_adjacency = getattr(
                        self, f"bridge_index_adjacency_{rk1}_{rk2}"
                    )
                    bridge_index_adjacency_perm = perm2[bridge_index_adjacency]
                    setattr(
                        perm_data,
                        f"bridge_index_adjacency_{rk1}_{rk2}",
                        bridge_index_adjacency_perm,
                    )

                    # permute coadjacency
                    coadjacency = getattr(self, f"edge_index_coadjacency_{rk2}_{rk1}")
                    coadjacency_perm = perm2[coadjacency]
                    setattr(
                        perm_data,
                        f"edge_index_coadjacency_{rk2}_{rk1}",
                        coadjacency_perm,
                    )

                    # permute bridge_index_coadjacency
                    bridge_index_coadjacency = getattr(
                        self, f"bridge_index_coadjacency_{rk2}_{rk1}"
                    )
                    bridge_index_coadjacency_perm = perm1[bridge_index_coadjacency]
                    setattr(
                        perm_data,
                        f"bridge_index_coadjacency_{rk2}_{rk1}",
                        bridge_index_coadjacency_perm,
                    )

            copies.append(perm_data)
        return copies

    def compute_subcomplex_feature(
        self, low_rk: int, high_rk: int, binary_marking: bool = False
    ):
        # the birdge ranks dictate the adjacency matrix chosen for each rank,

        low_bridge_rk = 1
        high_bridge_rk = 0

        # call all relevant componets from the data class
        x_low = getattr(self, f"x_{low_rk}")
        x_high = getattr(self, f"x_{high_rk}")

        incidence = getattr(self, f"edge_index_incidence_{low_rk}_{high_rk}")
        low_adjacency = getattr(self, f"edge_index_adjacency_{low_rk}_{low_bridge_rk}")
        low_bridge_index = getattr(
            self, f"bridge_index_adjacency_{low_rk}_{low_bridge_rk}"
        )
        high_adjacency = getattr(
            self, f"edge_index_coadjacency_{high_rk}_{high_bridge_rk}"
        )
        high_bridge_index = getattr(
            self, f"bridge_index_coadjacency_{high_rk}_{high_bridge_rk}"
        )

        # construct sub graph:
        subcomplex_features = Utils.construct_subcomplex_features(
            x_low=x_low,
            x_high=x_high,
            incidence=incidence,
            low_adjacency=low_adjacency,
            high_adjacency=high_adjacency,
            low_bridge=low_bridge_index,
            high_bridge=high_bridge_index,
            binary_marking=binary_marking,
        )

        suffix = f"_{low_rk}_{high_rk}"
        for key, val in subcomplex_features.items():
            setattr(self, key + suffix, val)


def get_dense_adj(edge_index: torch.Tensor, num_cells: int):
    # expands to_dense_adj to the case where edge_index is very disconnected or even 
    # empty makes sure the matrix size is nodes times nodes
    adj_matrix = to_dense_adj(edge_index)
    pad_dif = num_cells - adj_matrix.shape[-1]
    if pad_dif > 0:
        adj_matrix = F.pad(
            adj_matrix, (0, pad_dif, 0, pad_dif), mode="constant", value=0
        )
    return adj_matrix


def get_adjacency_from_incidence(
    incidence_matrix: csr_matrix,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pairs = []
    bridge_index = []

    if incidence_matrix.shape == (1,):
        if incidence_matrix[0] == 0:
            pairs_array = np.empty((2, 0), dtype=int)
            return (
                torch.tensor(bridge_index).long(),
                torch.from_numpy(pairs_array).long(),
            )

    for i in range(incidence_matrix.shape[1]):
        # Get the non-zero indices where T[:, i] == 1
        indices = incidence_matrix[:, i].nonzero()[0]
        # print("indices: ", len(indices))
        # Generate all pairs (k, j) using broadcasting with numpy
        if len(indices) > 1:  # Only consider if there are at least two indices

            k, j = np.meshgrid(indices, indices, indexing="ij")
            mask = k != j  # Avoid (k, k) pairs
            valid_pairs = np.vstack([k[mask], j[mask]])
            pairs.append(valid_pairs)

            bridge_index += [i] * valid_pairs.shape[-1]

    # Concatenate all the pairs into a single numpy array
    if pairs:
        pairs_array = np.hstack(pairs)
    else:
        pairs_array = np.empty((2, 0), dtype=int)  # Handle the case with no pairs
    return torch.tensor(bridge_index).long(), torch.from_numpy(pairs_array).long()
