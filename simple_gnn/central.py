import pickle
import socket
import sys

import torch
import yaml
from torch_geometric.utils import to_networkx
import networkx as nx


def send_config_to_node(host, port, config):
    data = pickle.dumps(config)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((host, port))
    s.sendall(data)
    s.close()


def load_model(config):
    model_config = torch.load(config["model"]["pathOld"], weights_only=False)
    return model_config
    # raw = torch.load(config["model"]["path"], map_location="cpu")

    # model_config = {
    #     "architecture": raw["architecture"],
    #     "in_channels": raw["in_channels"],
    #     "hidden_channels": raw["hidden_channels"],
    #     "num_layers": raw["num_layers"],
    #     "out_channels": raw["out_channels"],   # = hidden_channels (64)
    #     "state_dict": raw["encoder"],          # samo encoder ide node-ovima
    # }

    # # scorer spremi centralno ako ga želiš kasnije koristiti
    # model_config["scorer"] = raw["scorer"]
    # model_config["prediction_target"] = raw["prediction_target"]
    # if "edge_feat_mode" in raw:
    #     model_config["edge_feat_mode"] = raw["edge_feat_mode"]

    # return model_config

def load_model2(config):
    # model_config = torch.load(config["model"]["path"], weights_only=False)
    # return model_config
    raw = torch.load(config["model"]["path"], map_location="cpu")

    model_config = {
        "architecture": raw["architecture"],
        "in_channels": raw["in_channels"],
        "hidden_channels": raw["hidden_channels"],
        "num_layers": raw["num_layers"],
        "out_channels": raw["out_channels"],   # = hidden_channels (64)
        "state_dict": raw["encoder"],          # samo encoder ide node-ovima
    }

    # scorer spremi centralno ako ga želiš kasnije koristiti
    model_config["scorer"] = raw["scorer"]
    model_config["prediction_target"] = raw["prediction_target"]
    if "edge_feat_mode" in raw:
        model_config["edge_feat_mode"] = raw["edge_feat_mode"]

    model_config["features"] = raw["features"]
    print("CONFIG FEATURES: ")
    print(model_config["features"])

    return model_config 

def load_data(config):
    # Load the dataset and the specific test data index.
    dataset = torch.load(config["test_data"]["pathOld"], weights_only=False)
    test_data = dataset[config["test_data"]["indexOld"]] # odabir jednog grafa
    print("TEST DATA")
    print(test_data)

    # Extract neighboring nodes for each node in the test data.
    list_of_nodes = list(config["nodes"].keys())
    neighbors = {k: [] for k in config["nodes"]}
    G = to_networkx(test_data, to_undirected=True)
    print(G)

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

    print("features")
    print(features)
    print("neighbors")
    print(neighbors)
    return neighbors, features

def load_data2(config, model_conf):
    # Load the dataset and the specific test data index.
    dataset = torch.load(config["test_data"]["path"], weights_only=False)
    # test_data = dataset[config["test_data"]["index"]] # odabir jednog grafa
    test_data = dataset[config["test_data"]["index"]] # odabir jednog grafa
    print(test_data)

    # Extract neighboring nodes for each node in the test data.
    list_of_nodes = list(config["nodes"].keys())
    neighbors = {k: [] for k in config["nodes"]}
    graph6_bytes = test_data["graph6"].encode('utf-8')
    G = nx.from_graph6_bytes(graph6_bytes) # to_undirected=True

    features = {k: torch.empty(5) for k in config["nodes"]}


    i = 0
    for node in G.nodes:
        # Populate the neighbors dictionary with the node addresses and ports.
        for neighbor in G.neighbors(node):
            addr_and_port = config["nodes"][list_of_nodes[neighbor]]
            if addr_and_port[0] == "same_as_central":
                addr_and_port[0] = f"{socket.gethostname()}.local"
            neighbors[list_of_nodes[node]].append(addr_and_port)

        # Add the node's initial feature vector.
        # features[list_of_nodes[node]] = test_data.x[node]
        pomocno_polje = []
        for feature in model_conf["features"]:
            pomocno_polje.append(test_data["features"][feature][i])
        features[list_of_nodes[node]] = torch.tensor(pomocno_polje)
        i += 1

    print("features")
    print(features)
    print("neighbors")
    print(neighbors)
    return neighbors, features


def main():
    if len(sys.argv) == 2:
        config_file = f"config/{sys.argv[1]}"
    else:
        config_file = "config/config.yaml"
    with open(config_file) as f:
        initial_config = yaml.safe_load(f)

    model_config = load_model2(initial_config)
    neighbors, features = load_data2(initial_config, model_config)
    # print("------------------------------------------------")
    # model_config = load_model(initial_config)
    # neighbors, features = load_data(initial_config)

    for node_id, addr_port in initial_config["nodes"].items():
        print(f"Sending config to {node_id} at {addr_port[0]}:{addr_port[1]}")
        config = {"model": model_config, "neighbors": neighbors[node_id], "features": features[node_id]}
        try:
            send_config_to_node(addr_port[0], addr_port[1], config)
        except Exception as e:
            print(f"Failed to send config to {node_id}: {e}")


if __name__ == "__main__":
    main()
