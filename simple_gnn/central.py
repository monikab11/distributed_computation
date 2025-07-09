import yaml
import sys
import socket
import torch
import argparse
import pickle
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
    dataset = torch.load(config["test_data"]["path"])
    test_data = dataset[config["test_data"]["index"]]

    # Extract neighboring nodes for each node in the test data.
    list_of_nodes = list(config["nodes"].keys())
    neighbors = {k: [] for k in config["nodes"]}
    G = to_networkx(test_data, to_undirected=True)

    features = {}

    for node in G.nodes:
        # Populate the neighbors dictionary with the node addresses and ports.
        for neighbor in G.neighbors(node):
            neighbors[list_of_nodes[node]].append(config["nodes"][list_of_nodes[neighbor]])

        # Add the node's initial feature vector.
        features[list_of_nodes[node]] = test_data.x[node]

    return neighbors, features


def main():
    with open("config/config.yaml") as f:
        initial_config = yaml.safe_load(f)

    model_config = load_model(initial_config)
    neighbors, features = load_data(initial_config)

    for node_id, addr_port in initial_config['nodes'].items():
        print(f"Sending config to {node_id} at {addr_port[0]}:{addr_port[1]}")
        config = {"model": model_config,
                  "neighbors": neighbors[node_id],
                  "features": features[node_id]}
        try:
            send_config_to_node(addr_port[0], addr_port[1], config)
        except Exception as e:
            print(f"Failed to send config to {node_id}: {e}")

if __name__ == "__main__":
    main()
