import os
import shutil
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm
from data.lifts import lift_data
from data.complex_data import ComplexData
from typing import Optional, List, Dict
import numpy as np
import random


class ComplexDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        data_list: List[ComplexData] = None,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.data_list = data_list
        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0], data_cls=ComplexData)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def process(self):
        self.save(self.data_list, self.processed_paths[0])


def construct_datasets(
    root,
    datasets: Dict[str, InMemoryDataset],
    dim: int = 2,
    min_len: int = 3,
    max_len: int = 6,
    use_subcomplex_features: bool = False,
    subcomplex_low_rank: Optional[int] = None,
    subcomplex_high_rank: Optional[int] = None,
    compute_cross_diameter: bool = False,
    compute_betti_numbers: bool = False,
    binary_marking: bool = False,
):
    """
    Given a dict of train, val, test graph datasets, lifts them into combinatorial complex datasets.
    :param root: Path for the root directory of the new datasets.
    :param datasets: Dict with train, validation and test datasets to lift.
    :param dim: Dimension of the combinatorial complexes to lift to.
    :param lift_method: Lifting algorithm.
    :param min_len: For cyclic/clique lifting, min size for constructing a cell
    :param max_len: For cyclic/clique lifting, max size for constructing a cell
    :param num_clusters: For spectral lifting, num cluster in k-means
    :param num_eigenvectors: For spectral lifting, num eigenvectors used in pos encoding
    :param use_subcomplex_features: bool
    :param subcomplex_low_rank: an integer indicating the  low rank to use in subcomplex_features
    :param subcomplex_high_rank: an integer indicating the high rank to use in subcomplex_features
    :return: Dict of new train, validation and test datasets.
    """
    # Lift each data object in the datasets into a combinatorial complex
    datasets = {
        name: [
            lift_data(
                data,
                dim=dim,
                min_len=min_len,
                max_len=max_len,
                compute_betti_numbers=compute_betti_numbers,
                compute_cross_diameter=compute_cross_diameter,
            )
            for data in tqdm(dataset)
        ]
        for name, dataset in datasets.items()
    }
    print("lifted")

    if compute_cross_diameter:
        datasets = filter_feature(datasets, "cross_diameter")
        datasets = normalize_feature(datasets, "cross_diameter")

    if compute_betti_numbers:
        datasets = get_data_with_most_common_features(datasets, feature_name="b_2")
        datasets = normalize_feature(datasets, "b_2")

    # Compute max number of cells for each rank
    if use_subcomplex_features:
        multi_cell_datasets = {}
        for name, dataset in datasets.items():
            print(name)
            multi_cell_datalist = [
                add_multi_cell_features(
                    data,
                    subcomplex_low_rank=subcomplex_low_rank,
                    subcomplex_high_rank=subcomplex_high_rank,
                    binary_marking=binary_marking,
                )
                for data in tqdm(dataset)
            ]
            multi_cell_datasets[name] = ComplexDataset(
                root=root + f"/{name}", data_list=multi_cell_datalist
            )
            del dataset
    else:
        multi_cell_datasets = datasets

    return multi_cell_datasets


def add_multi_cell_features(
    complex_data: ComplexData,
    subcomplex_low_rank: Optional[List[int]] = None,
    subcomplex_high_rank: Optional[List[int]] = None,
    binary_marking: bool = False,
) -> ComplexData:
    complex_data.compute_subcomplex_feature(
        low_rk=subcomplex_low_rank,
        high_rk=subcomplex_high_rank,
        binary_marking=binary_marking,
    )
    return complex_data


def delete_all_files_in_directory(directory):
    # Check if the directory exists
    if os.path.exists(directory):
        # Iterate over all files and directories within the specified directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                # Check if it's a file and delete it
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                # Check if it's a directory and remove it
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"Directory {directory} does not exist.")


def normalize_feature(datasets, feature_name):
    values = np.array([getattr(data, feature_name) for data in datasets["train"]])
    mean = values.mean()
    std = values.std()
    for dataset in datasets.values():
        for data in dataset:
            value = (getattr(data, feature_name) - mean) / std
            setattr(data, feature_name + "_normalized", value)
            setattr(data, "y_mean", mean)
            setattr(data, "y_std", std)
    return datasets


def get_data_with_most_common_features(
    datasets, feature_name, num_top_features: int = 3
):

    # shuffle all datapoints from train val test
    big_dataset = [data for dataset in list(datasets.values()) for data in dataset]
    random.shuffle(big_dataset)

    # get all target values
    feature_values = np.array([getattr(data, feature_name) for data in big_dataset])

    # find the most common values
    unique_values, counts = np.unique(feature_values, return_counts=True)
    top_k_values = unique_values[np.argsort(counts)][-num_top_features:]

    # find the number of samples to take from each value to get the same ammout of 
    # samples per category
    num_per_category = counts[np.argsort(counts)][-num_top_features]

    # create stratefied dataset
    stratified_dataset = []
    for val in top_k_values:
        num_sampled = 0
        for i, current_val in enumerate(feature_values):
            if val == current_val:
                num_sampled += 1
                stratified_dataset.append(big_dataset[i])

            if num_sampled >= num_per_category:
                break
    random.shuffle(stratified_dataset)

    # train val test split
    n = len(stratified_dataset)
    train_end = int(n * 0.6)
    val_end = train_end + int(n * 0.2)

    datasets = {
        "train": stratified_dataset[:train_end],
        "val": stratified_dataset[train_end:val_end],
        "test": stratified_dataset[val_end:],
    }
    return datasets


def filter_feature(datasets, feature_name):
    return {
        name: [data for data in dataset if getattr(data, feature_name) >= 0]
        for name, dataset in datasets.items()
    }
