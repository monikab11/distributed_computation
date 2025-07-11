import base64
import functools
import hashlib
import json
import random
from pathlib import Path

import codetiming
import networkx as nx
import numpy as np
import torch
import torch_geometric.utils as tg_utils
import yaml
from matplotlib import pyplot as plt
from torch_geometric.data import InMemoryDataset

from my_graphs_dataset import GraphDataset


class SimpleDataset(InMemoryDataset):
    # https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/datasets/gdelt.py
    # If you want to define different graphs for training and testing.
    def __init__(
        self, root, loader: GraphDataset | None = None, transform=None, pre_transform=None, pre_filter=None, **kwargs
    ):
        if loader is None:
            loader = GraphDataset()
        self.loader = loader

        print("*****************************************")
        print(f"** Creating dataset with ID {self.hash_representation} **")
        print("*****************************************")

        # Calls InMemoryDataset.__init__ -> calls Dataset.__init__  -> calls Dataset._process -> calls self.process
        super().__init__(root, transform, pre_transform, pre_filter, force_reload=kwargs.get("force_reload", False))

        self.load(self.processed_paths[0])

    @property
    def raw_dir(self):
        return str(self.loader.raw_files_dir.resolve())

    @property
    def raw_file_names(self):
        """
        Return a list of all raw files in the dataset.

        This method has two jobs. The returned list with raw files is compared
        with the files currently in raw directory. Files that are missing are
        automatically downloaded using download method. The second job is to
        return the list of raw file names that will be used in the process
        method.
        """
        with open(Path(self.root) / "file_list.yaml", "r") as file:
            raw_file_list = sorted(yaml.safe_load(file))
        return raw_file_list

    @property
    def processed_file_names(self):
        """
        Return a list of all processed files in the dataset.

        If a processed file is missing, it will be automatically created using
        the process method.

        That means that if you want to reprocess the data, you need to delete
        the processed files and reimport the dataset.
        """
        return [f"data_{self.hash_representation}.pt"]

    @property
    def hash_representation(self):
        dataset_props = json.dumps([self.loader.hashable_selection, self.feature_dims, self.loader.seed])
        sha256_hash = hashlib.sha256(dataset_props.encode("utf-8")).digest()
        hash_string = base64.urlsafe_b64encode(sha256_hash).decode("utf-8")[:10]
        return hash_string

    def download(self):
        """Automatically download raw files if missing."""
        # TODO: Should check and download only missing files.
        # zip_file = Path(self.root) / "raw_data.zip"
        # zip_file.unlink(missing_ok=True)  # Delete the exising zip file.
        # download_url(raw_download_url, self.root, filename="raw_data.zip")
        # extract_zip(str(zip_file.resolve()), self.raw_dir)
        raise NotImplementedError("Automatic download is not implemented yet.")

    def process(self):
        """Process the raw files into a graph dataset."""
        # Read data into huge `Data` list.
        data_list = []
        for graph in self.loader.graphs(batch_size=1, raw=False):
            data_list.append(self.make_data(graph))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        self.save(data_list, self.processed_paths[0])

    # *************************
    # *** Feature functions ***
    # *************************
    @staticmethod
    def A_matrix_row(G, size):
        """
        Returns the row of the adjacency matrix for each node.
        """
        A = nx.to_numpy_array(G)
        n = A.shape[0]
        A = np.hstack([A, np.zeros((n, size - n))])
        results = {node: A[i, :] for i, node in enumerate(G.nodes())}
        return results

    feature_functions = {
        # "one": lambda g: dict.fromkeys(g.nodes(), 1),
        "A_matrix_row": lambda g: SimpleDataset.A_matrix_row(g, 5),
        # "random": lambda g: nx.random_layout(g, seed=np.random), # This works because GraphDataset loader sets the seed
    }
    # *************************

    # Make the data.
    def make_data(self, G):
        """Create a PyG data object from a graph object."""
        # Compute and add features to the nodes in the graph.
        for feature in self.feature_functions:
            feature_val = self.feature_functions[feature](G)
            for node in G.nodes():
                G.nodes[node][feature] = feature_val[node]

        torch_G = tg_utils.from_networkx(G, group_node_attrs=list(self.feature_functions.keys()))
        torch_G.x = torch_G.x.to(torch.float32)

        torch_G.y = torch.tensor([v for k, v in G.degree], dtype=torch.float32)

        return torch_G

    @property
    def features(self):
        return list(self.feature_functions.keys())

    @functools.cached_property
    def feature_dims(self):
        """
        Calculate the dimensions of the features.

        Some features (like one-hot encoding and random) may have variable
        dimensions so dataset.num_features != len(dataset.features).
        """
        feature_dims = {}
        G = nx.path_graph(3)  # Dummy graph to get the feature dimensions.
        for feature in self.feature_functions:
            feature_val = self.feature_functions[feature](G)
            try:
                feature_dims[feature] = len(feature_val[0])
            except TypeError:
                feature_dims[feature] = 1

        return feature_dims


def inspect_dataset(dataset):
    if isinstance(dataset, InMemoryDataset):
        dataset_name = dataset.__repr__()
        y_values = dataset.y
    else:
        dataset_name = "N/A"
        y_values = torch.tensor([data.y for data in dataset])

    print()
    header = f"Dataset: {dataset_name}"
    print(header)
    print("=" * len(header))
    print(f"Num. of graphs: {len(dataset)}")
    print(f"Target:")
    print(f"    Min: {y_values.min().item():.3f}")
    print(f"    Max: {y_values.max().item():.3f}")
    print(f"    Mean: {y_values.mean().item():.3f}")
    print(f"    Std: {y_values.std().item():.3f}")
    print("=" * len(header))
    print()


def inspect_graphs(dataset, graphs: int | list = 1):
    """
    Inspect and display information about graphs in a dataset.

    This function prints detailed information about one or more graph objects
    from the given dataset, including their structural properties and features.

    Args:
        dataset: A dataset object containing graph data.
        graphs (int | list, optional): Specifies which graphs to inspect.
                    If an integer is provided, that many random graphs will be
                    selected from the dataset. If a list of indices is provided,
                    the graphs at those indices will be inspected. Defaults to 1.
    Example:
        >>> inspect_graphs(my_dataset, graphs=3)
        >>> inspect_graphs(my_dataset, graphs=[0, 5, 10])
    """

    y_name = "Target value"

    if isinstance(graphs, int):
        graphs = random.sample(range(len(dataset)), graphs)

    for i in graphs:
        # for i in random.sample(range(len(dataset)), num_graphs):
        data = dataset[i]  # Get a random graph object

        print()
        header = f"{i} - {data}"
        print(header)
        print("=" * len(header))

        # Gather some statistics about the graph.
        print(f"Number of nodes: {data.num_nodes}")
        print(f"Number of edges: {data.num_edges}")
        print(f"s{data.y}")
        print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
        print(f"Has isolated nodes: {data.has_isolated_nodes()}")
        print(f"Has self-loops: {data.has_self_loops()}")
        print(f"Is undirected: {data.is_undirected()}")
        print(f"Features:\n{data.x}")
        print("=" * len(header))
        print()

        G = tg_utils.to_networkx(data, to_undirected=True)
        nx.draw(G, with_labels=True)
        plt.show()


def main():
    root = Path(__file__).parents[0] / "Dataset"
    selected_graph_sizes = {
        5: -1,
    }
    loader = GraphDataset(selection=selected_graph_sizes, seed=42)

    with codetiming.Timer():
        dataset = SimpleDataset(root, loader, selected_features=["degree"], force_reload=True)

    inspect_dataset(dataset)
    inspect_graphs(dataset, graphs=[1])


if __name__ == "__main__":
    main()
