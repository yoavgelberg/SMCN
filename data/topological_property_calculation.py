from typing import Tuple
import networkx as nx
import numpy as np
from networkx.classes.graph import Graph
from itertools import combinations
from data.complex_data import ComplexData
from data.utils import Utils


### second betti number computation ####
def get_betti_numbers(G: Graph, max_cycle_size: int = 18) -> Tuple[int, int, int]:
    """
    returns betti numbers b_0, b_1, b_2 of a graph lifted to a CC with cyclic lift
    """
    num_nodes = G.number_of_nodes()
    cycle_incidence_matrix = get_cycle_incidence_matrix(G=G, k=max_cycle_size)
    num_edges, num_cycles = cycle_incidence_matrix.shape
    euler_number = num_nodes - num_edges + num_cycles
    b_0 = nx.number_connected_components(G)

    # kernel_sets = find_kernel_sets(cycle_incidence_matrix)
    b_2 = num_cycles - matrix_rank_mod_2(cycle_incidence_matrix)
    b_1 = b_0 + b_2 - euler_number
    return b_0, b_1, b_2


def compute_second_betti_number(incidence_matrix):
    if incidence_matrix.shape[-1] == 0:
        return 0
    # Ensure the matrix is binary
    incidence_matrix = (incidence_matrix != 0).astype(int)
    m, n = incidence_matrix.shape  # m cycles, n edges

    # Compute the rank of the incidence matrix (dimension of the boundary space)
    boundary_dim = np.linalg.matrix_rank(incidence_matrix % 2)

    # Compute the nullity of the incidence matrix transpose (dimension of the cycle space)
    cycle_dim = m - np.linalg.matrix_rank(incidence_matrix.T % 2)

    # The second Betti number is the dimension of cycles modulo boundaries
    second_betti_number = cycle_dim - boundary_dim

    return second_betti_number


def matrix_rank_mod_2(matrix: np.ndarray) -> int:
    """
    computes the dimension of the linear space spanned by the columns of a 0-1 matrix
    where the space is over the field of size 2.
    """

    # Convert the matrix to F₂ (i.e., take all elements modulo 2)
    matrix = (matrix % 2).astype(int)

    # Get the number of rows and columns
    rows, cols = matrix.shape

    # Initialize rank
    rank = 0

    # Perform Gaussian elimination
    for j in range(cols):
        # Find a non-zero element in this column
        for i in range(rank, rows):
            if matrix[i, j] == 1:
                # Swap rows
                matrix[rank], matrix[i] = matrix[i].copy(), matrix[rank].copy()

                # Eliminate this variable from other equations
                for k in range(rank + 1, rows):
                    if matrix[k, j] == 1:
                        matrix[k] ^= matrix[rank]  # XOR operation (addition in F₂)

                rank += 1
                break

        if rank == rows:
            break

    return rank


def find_kernel_sets(incidence_matrix: np.ndarray, max_size: int = 5) -> np.ndarray:
    """
    given an incidence matrix, computes the set of kernel chanis in matrix form.
    returns a one hot encoded numpy array of shape (m, M) where m is the number of 
    cycles and M is the number of chains in the kernel
    """

    # Convert the matrix to F₂ (i.e., take all elements modulo 2)
    matrix = (incidence_matrix % 2).astype(int)

    num_cols = matrix.shape[1]
    kernel_sets = []

    def is_kernel_set(cols):
        sum_cols = np.sum(matrix[:, cols], axis=1) % 2
        return np.all(sum_cols == 0)

    # Check all possible combinations of columns
    for k in range(1, max_size):
        for combo in combinations(range(num_cols), k):
            if is_kernel_set(combo):
                kernel_sets.append(combo)

    # Create the output tensor
    output_tensor = np.zeros((num_cols, len(kernel_sets)), dtype=int)
    for i, ks in enumerate(kernel_sets):
        output_tensor[list(ks), i] = 1

    return output_tensor


def get_cycle_incidence_matrix(G, k):
    """
    given a networkx graph computes the edge-cycle incidence matrix without counting 
    chords as incident edges. 

    returns a nupy array of shape (n,m) where n is the number of edges and m is the 
    number of cycles.
    """

    def find_cycles(graph, start, current, path, visited, depth):
        if depth > k:
            return

        visited.add(current)
        path.append(current)

        for neighbor in graph.neighbors(current):
            if neighbor == start and depth > 2:
                cycle = path[:]
                cycles.append(cycle)
            elif neighbor not in visited:
                find_cycles(graph, start, neighbor, path, visited, depth + 1)

        path.pop()
        visited.remove(current)

    cycles = []
    for node in G.nodes():
        find_cycles(G, node, node, [], set(), 1)

    # Remove duplicate cycles
    unique_cycles = []
    cycle_sets = set()
    for cycle in cycles:
        cycle_set = frozenset(cycle)
        if cycle_set not in cycle_sets and len(cycle_set) == len(cycle):
            cycle_sets.add(cycle_set)
            unique_cycles.append(cycle)

    # Create edge to index mapping
    edge_to_index = {frozenset(edge): i for i, edge in enumerate(G.edges())}

    # Create incidence matrix
    incidence_matrix = np.zeros((len(G.edges()), len(unique_cycles)), dtype=int)

    for j, cycle in enumerate(unique_cycles):
        cycle_edges = list(zip(cycle, cycle[1:] + cycle[:1]))
        for edge in cycle_edges:
            edge_set = frozenset(edge)
            if edge_set in edge_to_index:
                i = edge_to_index[edge_set]
                incidence_matrix[i, j] = 1

    return incidence_matrix


### cross diameter computation ####
def get_cross_diameter(D: ComplexData) -> int:

    num_cells_low = D.num_cells_0
    num_cells_high = D.num_cells_2
    incidence = D.edge_index_incidence_0_2
    low_adjacency = D.edge_index_adjacency_0_1
    distance_matrix = Utils.get_subcomplex_distance_node_marking(
        incidence=incidence,
        num_cells_low=num_cells_low,
        num_cells_high=num_cells_high,
        edge_index_low=low_adjacency,
    )
    distance_matrix[distance_matrix == 1001] = -1
    if distance_matrix.shape[0] == 0:
        return -1
    return distance_matrix.max().item()
