import pickle
import socket
import sys

import torch
import yaml
from torch_geometric.utils import to_networkx


def send_config_to_node(host, port, config):
    data = pickle.dumps(config)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall(data)
    s.close()


def load_model(config):
    model_config = torch.load(config["model"]["path"], weights_only=False)
    return model_config


def load_data(config):
    # Load the dataset and the specific test data index.
    dataset = torch.load(config["test_data"]["path"], weights_only=False)
    test_data = dataset[config["test_data"]["index"]]

    # Extract neighboring nodes for each node in the test data.
    list_of_nodes = list(config["nodes"].keys())
    neighbors = {k: [] for k in config["nodes"]}
    G = to_networkx(test_data, to_undirected=True)

    features = {}

    for node in G.nodes:
        # Populate the neighbors dictionary with the node addresses and ports.
        for neighbor in G.neighbors(node):
            addr_and_port = config["nodes"][list_of_nodes[neighbor]]
            if addr_and_port[0] == "same_as_central":
                addr_and_port[0] = f"{socket.gethostname()}.local"
            neighbors[list_of_nodes[node]].append(addr_and_port)

        # Add the node's initial feature vector.
        features[list_of_nodes[node]] = test_data.x[node]

    return neighbors, features


def main():
    if len(sys.argv) == 2:
        config_file = f"config/{sys.argv[1]}"
    else:
        config_file = "config/config.yaml"
    with open(config_file) as f:
        initial_config = yaml.safe_load(f)

    model_config = load_model(initial_config)
    neighbors, features = load_data(initial_config)

    for node_id, addr_port in initial_config["nodes"].items():
        print(f"Sending config to {node_id} at {addr_port[0]}:{addr_port[1]}")
        config = {"model": model_config, "neighbors": neighbors[node_id], "features": features[node_id]}
        try:
            send_config_to_node(addr_port[0], addr_port[1], config)
        except Exception as e:
            print(f"Failed to send config to {node_id}: {e}")


if __name__ == "__main__":
    main()
