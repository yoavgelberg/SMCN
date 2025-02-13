from typing import List, Tuple
import networkx as nx
import torch
from torch_geometric.utils import from_networkx
from torch_geometric.data import InMemoryDataset
import math
from data.complex_data import ComplexData
from data.utils import Utils
from toponetx.classes import CombinatorialComplex as CCC
from torch_geometric.data.data import Data
from torch_geometric.utils import to_networkx
import numpy as np
import itertools
from tqdm import tqdm


class TorusDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        min_size: int = 18,
        max_size: int = 40,
        num_components: int = 3,
        number_of_permuted_copies: int = 32,
        low_rk: int = 0,
        high_rk: int = 1,
        first_element_of_pair: bool = True,
    ):
        self.min_size = min_size
        self.max_size = max_size
        self.num_components = num_components
        self.number_of_permuted_copies = number_of_permuted_copies
        self.first_element_of_pair = first_element_of_pair
        self.low_rk = low_rk
        self.high_rk = high_rk
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0], data_cls=ComplexData)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return "torus_dataset.pt"

    def process(self):
        # Read data into huge Data list.
        data_list = self._create_data_list()
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        self.save(data_list, self.processed_paths[0])

    def _create_data_list(self):
        data_list = []

        # get list of radi of pairs of complexes, each of which is a disconnected tori to construct tori
        radi_list = TorusDataset._get_radi(
            min_size=self.min_size,
            max_size=self.max_size,
            num_components=self.num_components,
        )

        for rad_list1, rad_list2 in tqdm(radi_list):
            # construct torus pair
            tori = (
                TorusDataset._create_disconnected_tori(rad_list1)
                if self.first_element_of_pair
                else TorusDataset._create_disconnected_tori(rad_list2)
            )
            tori_reliability = TorusDataset._create_disconnected_tori(
                rad_list1
            )  # we compare the first element of the pair to itself for reliability, making sure we cannot seperate a cc from itself

            # Randomly permute for brec procedure
            tori_copies = tori.get_permuted_copies(
                num_copies=self.number_of_permuted_copies
            )
            tori_reliability_copies = tori_reliability.get_permuted_copies(
                num_copies=self.number_of_permuted_copies
            )

            # Add subcomplex features
            tori_copies_subcomplex = []
            tori_reliability_copies_subcomplex = []
            for tori, tori_reliability in zip(tori_copies, tori_reliability_copies):
                tori.compute_subcomplex_feature(
                    low_rk=self.low_rk, high_rk=self.high_rk
                )
                tori_reliability.compute_subcomplex_feature(
                    low_rk=self.low_rk, high_rk=self.high_rk
                )

                tori_copies_subcomplex.append(tori)
                tori_reliability_copies_subcomplex.append(tori_reliability)

            data_list += tori_copies_subcomplex
            data_list += tori_reliability_copies_subcomplex

        return data_list

    @staticmethod
    def _create_disconnected_tori(rad_list: List[Tuple[int, int]]) -> Data:
        # construct first connected component

        (
            r,
            R,
        ) = rad_list[0]
        combinatorial_complex = TorusDataset._create_torus(r=r, R=R)
        # construct the rest of the complex
        for r, R in rad_list[1:]:
            # construct the next connceted component
            new_torus = TorusDataset._create_torus(r=r, R=R)

            # Gets cells of the new torus and their corresponding ranks
            cells, ranks = Utils._get_cell_indices(new_torus)

            # Reindex nodes within the cells
            cells = [
                list(np.array(list(x)) + combinatorial_complex.number_of_nodes())
                for x in cells
            ]

            # Add cells (rank 0 cells are added automatically) combining the new component with current complext
            for cell, rank in zip(cells, ranks):
                if rank > 0:
                    combinatorial_complex.add_cell(cell, rank)

        # Convert to class compatible with pytorch geometric
        combinatorial_complex = ComplexData.create_from_cc(combinatorial_complex)

        # record information about complex (for debugging purposes)
        combinatorial_complex.num_components = len(rad_list)
        combinatorial_complex.first_radius = rad_list[0][0]
        combinatorial_complex.second_radius = rad_list[0][1]

        # Add number of components
        # combinatorial_complex.num_components = len(rad_list)
        return combinatorial_complex

    @staticmethod
    def _get_updated_edge_index(
        combined_data: Data, data1: Data, data2: Data, shift: int, attr: str
    ):
        index1 = getattr(data1, attr)
        index2 = getattr(data2, attr)
        index2 += shift
        combined_index = torch.concat([index1, index2], dim=1)
        setattr(combined_data, attr, combined_index)

    @staticmethod
    def _create_torus(r: int, R: int) -> CCC:
        # Create the product of two cycles
        C_r = nx.cycle_graph(r)
        C_R = nx.cycle_graph(R)
        torus = nx.cartesian_product(C_r, C_R)

        # This is a trick for converting nodes from tuples to integers, could be improved.
        data = from_networkx(torus)
        torus = to_networkx(data)

        # Create cells of rank > 0
        two_cells = TorusDataset._create_cells(r=r, R=R)
        three_cells = TorusDataset._create_cells(r=r, R=R)

        # Build combinatorial complex from graph
        torus = CCC(torus)
        torus.add_cells_from(cells=two_cells, ranks=[2] * len(two_cells))
        return torus

    @staticmethod
    def _create_cells(r: int, R: int) -> List[Tuple[int]]:
        cells = []
        # iterate over nodes of the tours
        for x in range(r):
            for y in range(R):
                node = np.array([x, y])
                # node (x,y) corresponds to square {(x+i, y+j)| i,j in {0,1}}
                cell = [
                    TorusDataset._modulu_radius(x=node + np.array([i, j]), r=r, R=R)
                    for i in range(2)
                    for j in range(2)
                ]

                # As the nodes are ordered by (i,j) -> i*R_j aw convert them to the correct form
                cell = [TorusDataset._convert_to_index(x, R) for x in cell]
                cells.append(cell)

        return cells

    @staticmethod
    def _convert_to_index(prod_node: np.ndarray, R: int) -> int:
        #  converts node index of the form (a,b) to the index a*R+b, which is how they're arranged in the torus
        a, b = prod_node
        return a * R + b

    @staticmethod
    def _modulu_radius(x: np.ndarray, r: int, R: int) -> np.ndarray:
        return np.array([x[0] % r, x[1] % R])

    @staticmethod
    def _get_radi(
        min_size: int, max_size: int, num_components: int
    ) -> List[Tuple[Tuple[Tuple[int, int]]]]:
        """Generate a list of tuples (divisor, quotient) for randomly selected numbers."""
        radi = []
        for size in range(min_size, max_size + 1):
            # get all possible radi for torus of size
            radi_of_fixed_size = TorusDataset._find_radi_for_overall_size(
                size=size, max_num_components=num_components
            )

            # if there is more than one possible torus of size
            if len(radi_of_fixed_size) > 1:
                # create all possible pairs from all possible choice of discinnected troi of the same overall size
                radi += TorusDataset._generate_pairs_of_nonisomorphic_radi(
                    radi_of_fixed_size
                )
        return radi

    @staticmethod
    def _find_radi_for_overall_size(
        size: int, max_num_components: int
    ) -> List[Tuple[Tuple[int]]]:
        """finds all possible radi combinations for disjoint tori of a given overall size"""

        all_radi = []
        for num_components in range(1, max_num_components + 1):
            # gets all possibilites for the size of each connected component for a given size and number of components
            all_component_sizes = TorusDataset._find_component_sizes(
                size=size, num_components=num_components
            )
            # for each configuration of connected component sizes, finds all possible radi values.
            for component_sizes in all_component_sizes:
                all_radi += TorusDataset._find_component_radi(component_sizes)
        return all_radi

    @staticmethod
    def _find_component_radi(component_sizes: Tuple[int]) -> List[Tuple[int]]:
        return itertools.product(
            *[
                TorusDataset._find_radi(component_size)
                for component_size in component_sizes
            ]
        )

    @staticmethod
    def _find_component_sizes(
        size: int, num_components: int, min_value: int = 9
    ) -> List[Tuple[int, int]]:
        """
        given an overall size and a number of components, returns a list of all possible values for
         size of each connected component (min value is the minimal size of a component).
        """

        def generate_tuples(size, num_components, start, current_tuple):
            if num_components == 1:
                if size >= start:
                    yield current_tuple + (size,)
                return
            for i in range(start, size + 1):
                yield from generate_tuples(
                    size - i, num_components - 1, i, current_tuple + (i,)
                )

        return list(generate_tuples(size, num_components, min_value, ()))

    @staticmethod
    def _find_radi(size: int) -> List[int]:
        """Find all divisors of a given number n whsoe value is between 3 and sqrt(n)."""
        divisors = [
            (i, size // i) for i in range(3, int(math.sqrt(size) + 1)) if size % i == 0
        ]
        return divisors

    @staticmethod
    def _generate_pairs_of_nonisomorphic_radi(
        radi_of_fixed_size: List[Tuple[Tuple[int, int]]]
    ):
        """
        given a list of all possible radi of disconneted tori with the same fixed size,
        creates a list of all possible such pairs.
        """
        pairs = []
        for i in range(len(radi_of_fixed_size)):
            for j in range(i + 1, len(radi_of_fixed_size)):
                pairs.append((radi_of_fixed_size[i], radi_of_fixed_size[j]))
        return pairs
