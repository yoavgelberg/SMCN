from torch_geometric.utils import to_networkx
from toponetx.classes import CombinatorialComplex as CCC
import networkx as nx
from networkx import Graph
from itertools import combinations
from typing import Dict, List, Tuple
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from torch_geometric.data import Data
import torch
from data.complex_data import ComplexData
from data.topological_property_calculation import get_betti_numbers, get_cross_diameter


def lift_data(
    data: Data,
    dim: int = 2,
    min_len: int = 3,
    max_len: int = 6,
    compute_betti_numbers: bool = False,
    compute_cross_diameter: bool = False,
):
    cell_features = get_cell_features(data)
    y = data.y
    graph = to_networkx(data, to_undirected=True)

    cell_dict = get_cycle_cells(graph=graph, min_len=min_len, max_len=max_len, dim=dim)

    CC = CCC(graph)
    for i, cells in cell_dict.items():
        CC.add_cells_from(cells=cells, ranks=[i] * len(cells))
    complex_data = ComplexData.create_from_cc(
        CC, cell_features=cell_features, dim=dim, y=y
    )

    # add topological properties
    if compute_betti_numbers:
        b_0, b_1, b_2 = get_betti_numbers(graph, max_len)
        complex_data.b_0 = b_0
        complex_data.b_1 = b_1
        complex_data.b_2 = b_2
    if compute_cross_diameter:
        complex_data.cross_diameter = get_cross_diameter(complex_data)
    return complex_data


def get_cell_features(data: Data):
    cell_features = []
    if data.x is not None:
        cell_features.append(data.x.float())
    if data.edge_attr is not None:

        # convert to undirected graph to not have doubles of each edge feature
        G = to_networkx(data, to_undirected=True, edge_attrs=["edge_attr"])

        # get attributes , need to check its consistent
        edge_attr = torch.tensor(list(nx.get_edge_attributes(G, "edge_attr").values()))
        edge_attr = edge_attr.unsqueeze(1) if edge_attr.dim() == 1 else edge_attr

        # if the graph has no edges override the edge_attr computation above
        if data.edge_attr.shape[0] == 0:
            edge_attr = data.edge_attr

        cell_features.append(edge_attr.float())
    return cell_features


def get_spectral_pooling_cells(
    graph: Graph, num_clusters: int = 2, num_eigenvectors: int = 5
) -> Dict[int, List[Tuple[int]]]:
    cells = {2: []}

    # Compute the Laplacian matrix
    L = nx.laplacian_matrix(graph).todense()

    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = eigh(L)

    # Use the first 'num_eigenvectors' eigenvectors for clustering (ignore the first 
    # eigenvector if it is all zeros)
    X = eigenvectors[:, 1 : num_eigenvectors + 1]

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    labels = kmeans.labels_
    for i in range(num_clusters):
        cell = tuple(np.where(labels == i)[0])
        cells[2].append(cell)
    return cells


def get_cycle_cells(
    graph: Graph, min_len: int = 3, max_len: int = 6, dim: int = 2
) -> Dict[int, List[Tuple[int]]]:
    cells = {i: [] for i in range(2, dim + 1)}
    for i in range(min_len, max_len + 1):
        cells[2] += find_cycles_of_length_i(graph, i)
    return cells


def get_clique_cells(graph: Graph, dim: int = 2) -> Dict[int, List[Tuple[int]]]:
    cliques = nx.find_cliques(graph)
    cells = {i: [] for i in range(2, dim + 1)}
    for clique in cliques:
        if len(clique) <= dim + 1:
            for i in range(2, dim + 1):
                for c in combinations(
                    clique, i + 1
                ):  # cells of dim 2 are cliques of size 3
                    cells[i].append(tuple(c))
    return cells


def find_cycles_of_length_i(graph: Graph, i: int):
    def dfs(v, start, depth, path):
        if depth == i:
            if start in graph[v]:
                cycles.append(path + [start])
            return
        for neighbor in graph[v]:
            if neighbor not in path:  # Ensure no repeated nodes
                dfs(neighbor, start, depth + 1, path + [neighbor])

    cycles = []
    for node in graph.nodes():
        dfs(node, node, 1, [node])

    # Remove duplicate cycles by converting to a set of frozensets and back to list of lists
    unique_cycles = [
        tuple(cycle) for cycle in set(frozenset(cycle) for cycle in cycles)
    ]
    return unique_cycles
