import asyncio
import importlib
import pickle
import platform
import sys
from collections import defaultdict

from gnn_splitter import GNNSplitter
from torch import Tensor
from gnn_model import NodeScorer
import matplotlib.cm as cm


class Node:
    def __init__(self, node_id: str):
        self.node_id = f"node{node_id}"  # Just for printing.
        self.port = 5000 + int(node_id)  # Each node has a unique port starting from 5000, and we use it to ID the node.
        self.server = None  # Server used to receive messages from neighbors.

        self.config = {}  # Configuration holding the model properties, neighbors, and initial features.

        self.value = Tensor()  # The current representation of the node.
        self.output = Tensor()  # The interpretable output of the GNN after each layer.
        self.layer = 0  # The current layer of the GNN being processed.
        self.received = defaultdict(dict)  # Values received from neighbors, indexed by iteration number.
                                           # Also used for synchronization.

        # Check if this is running on Raspberry Pi.
        if platform.uname().machine == "aarch64":
            from led_matrix import LEDMatrix
        else:
            class LEDMatrix:
                def __init__(self):
                    pass

                def set_percentage(self, percent, resolution, base_color, mode='l2r', color_space='rgb'):
                    pass

                def colormap_to_rgb(self, value, cmap_name):
                    # cmap = cm.get_cmap(cmap_name)
                    # r, g, b, _ = cmap(value)
                    # return (int(r * 255), int(g * 255), int(b * 255))
                    pass

                def set_all(self, color, color_space='rgb'):
                    pass
                
                def off(self):
                    pass
        self.led = LEDMatrix()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Always clean up the LED matrix
        self.led.off()

        # If an exception occurred, re-raise it after cleanup
        if exc_type is not None:
            raise exc_val

    def initialize_GNN(self):
        # Load model configuration from the received config and set up the GNN model.
        model_config = self.config["model"]
        GNN = getattr(importlib.import_module("torch_geometric.nn"), model_config["architecture"])
        # print("INITIAL ")
        # print(model_config["in_channels"])
        self.model = GNN(
            in_channels=model_config["in_channels"],
            hidden_channels=model_config["hidden_channels"],
            num_layers=model_config["num_layers"],
            out_channels=model_config["out_channels"],
        )
        # stvori enkoder ovdje

        # print("MODEL PROPS")
        # print(self.model.in_channels)
        # print(self.model.hidden_channels)
        # print(self.model.num_layers)
        # print(self.model.out_channels)


        import torch
        import sys

        # if len(sys.argv) != 2:
        #     print("Usage: python inspect_pt.py <path_to_file.pt>")
        #     return

        # path = sys.argv[1]
        # print(f"Loading {path} ...")
        # data = torch.load(path, map_location="cpu", weights_only=False)

        # print("\n=== Content of the .pt file ===")
        # if isinstance(model_config, dict):
        #     for key, value in model_config.items():
        #         if isinstance(value, dict):
        #             print(f"{key}: dict with {len(value)} keys")
        #             for subkey, subval in value.items():
        #                 if hasattr(subval, 'shape'):
        #                     print(f"   {subkey}: Tensor of shape {tuple(subval.shape)}")
        #                 elif isinstance(subval, (int, float, str)):
        #                     print(f"   {subkey}: {subval}")
        #                 else:
        #                     print(f"   {subkey}: {type(subval)}")
        #         elif hasattr(value, 'shape'):
        #             print(f"{key}: Tensor of shape {tuple(value.shape)}")
        #         else:
        #             print(f"{key}: {value}")
        # else:
        #     print("File contains:", type(data))
        #     print(data)

        self.model.load_state_dict(model_config["state_dict"])
        self.distributed_model = GNNSplitter(self.model)

        # Load the initial feature vector for this node.
        self.value = self.config["features"].unsqueeze(0)  # Ensure it's a batch of size 1.
        
        print("BeFoRe")
        self.scorer = NodeScorer(node_dim=model_config["hidden_channels"],
                                hidden_dim=model_config["hidden_channels"])
        self.scorer.load_state_dict(model_config["scorer"])
        self.distributed_model.scorer = self.scorer
        print("AfTeR")
        
        # print()
        # print("FEATURES")
        # print(self.config["features"].unsqueeze(0))
        # print()
        print(f"Initial feature vector for node {self.node_id}: {self.value}")

    async def wait_for_config(self, port):
        # Use outside varable to store the handler data.
        handler_data = {}

        # Define a callback function to handle the incoming config.
        config_received = asyncio.Event()

        async def handle_config(reader, writer):
            data = await reader.read()
            handler_data["config"] = pickle.loads(data)
            print("HANDLER: ")
            print(handler_data["config"]["model"])
            writer.close()
            await writer.wait_closed()
            config_received.set()

        # Open a server to listen for the config from the central node.
        # We need to listen on our own port.
        server = await asyncio.start_server(handle_config, "0.0.0.0", port)
        print(f"Node {self.node_id} waiting for config on port {port}")

        # Wait for the config to be received.
        await config_received.wait()
        server.close()
        await server.wait_closed()

        return handler_data["config"]

    async def start(self):
        # Wait for config from central node.
        config = await self.wait_for_config(self.port)
        self.config = config
        # print("CONFIG: ") 
        # print(self.config)

        # Set up the GNN model and forward pass logic.
        self.initialize_GNN()

        # Start a server to receive messages from neighbors.
        # We are again listening on our own port.
        self.server = await asyncio.start_server(self.receive_value, "0.0.0.0", self.port)
        print(f"Node {self.node_id} listening on port {self.port}")
        await asyncio.sleep(1)  # Wait for all nodes to start

        # Continuously exchange values with neighbors and update representation.
        while self.layer < self.model.num_layers:
            await self.exchange_and_update()
            print(f"Layer {self.layer} output: {self.output}\n")
            print(self.config["model"]["min_value"])
            print(self.config["model"]["max_value"])
            val = self.output
            vmin = self.config["model"]["min_value"]
            vmax = self.config["model"]["max_value"]
            norm_val = max(0.0, min(1.0, (val - vmin) / (vmax - vmin)))
            # print("norm_val")
            # print(norm_val)
            rgb = self.led.colormap_to_rgb(norm_val, "jet")
            # rgb = self.led.set_percentage(norm_val, cmap_name="jet")
            print("rgb")
            print(rgb)
            self.led.set_all(rgb)
            # TODO: ovdje implementiraj drukciju logiku
            # npr spektar zelena - zuta - crvena, ali kako znati kako su druge obojene  
            # self.led.set_percentage(self.output / 4, 800, (255, 0, 0), "l2r")
            self.layer += 1
            await asyncio.sleep(2)

        print()
        # print(f"Final value: {self.value.item()}")
        print(f"Final value: {self.output}")
        await asyncio.sleep(5)

    async def exchange_and_update(self):
        # Send value to all neighbors.
        # Use their ports to send a message.
        # print('SENDING VALUES: ')
        # print(self.config["neighbors"])
        for neighbor_addr_port in self.config["neighbors"]:
            await self.send_value(neighbor_addr_port)

        # Wait for values from all neighbors.
        while len(self.received[self.layer]) < len(self.config["neighbors"]):
            await asyncio.sleep(0.1)

        # Update value (average consensus)
        # print("HEREEEEEEEEEEEEEEEEEE UPDATE_NODE")
        self.value, self.output = self.distributed_model.update_node(
            self.layer, self.value, list(self.received[self.layer].values())
        )

    async def send_value(self, neighbor_addr_port):
        # Connect to the neighbor with the given address and port.
        reader, writer = await asyncio.open_connection(neighbor_addr_port[0], neighbor_addr_port[1])

        if self.value.shape[1] <= 5:
            print(f"[{self.port}->{neighbor_addr_port[1]}] h^{self.layer}={self.value}")
        else:
            print(
                f"[{self.port}->{neighbor_addr_port[1]}] h^{self.layer}=[... >5, mean={self.value.mean().item():.3f}]"
            )

        msg = {"layer": self.layer, "sender": self.port, "value": self.value}
        writer.write(pickle.dumps(msg))
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def receive_value(self, reader, writer):
        # Read and load the incoming data.
        data = await reader.read()
        msg = pickle.loads(data)

        layer = msg["layer"]
        sender = msg["sender"]
        value = msg["value"]

        if value.shape[1] <= 5:
            print(f"[{self.port}<-{sender}] h^{layer}={value}")
        else:
            print(f"[{self.port}<-{sender}] h^{layer}=[... >5, mean={value.mean().item():.3f}]")

        self.received[int(layer)][sender] = value
        writer.close()
        await writer.wait_closed()


def main():
    if len(sys.argv) != 2:
        node_index = "1"
    else:
        node_index = sys.argv[1]

    # Initialize the node and then start it.
    with Node(node_index) as node:
        asyncio.run(node.start())



if __name__ == "__main__":
    main()
