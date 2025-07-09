import asyncio
import sys
from collections import defaultdict

import yaml


class Node:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.config = None
        self.value = 0.0
        self.neighbors = []
        self.ports = {}
        self.hosts = {}
        self.server = None
        self.received = defaultdict(dict)
        self.it = 0
        self.sigma = 0.5  # Default step size for average consensus

    async def wait_for_config(self, port):
        # Listen for config from central node
        config_received = asyncio.Event()
        config_data = {}
        async def handle_config(reader, writer):
            data = await reader.read()
            config = yaml.safe_load(data.decode())
            config_data['config'] = config
            writer.close()
            await writer.wait_closed()
            config_received.set()
        server = await asyncio.start_server(handle_config, '0.0.0.0', port)
        print(f"Node {self.node_id} waiting for config on port {port}")
        await config_received.wait()
        server.close()
        await server.wait_closed()
        return config_data['config']

    def get_port_from_args(self):
        # Helper to get port from config file for initial config reception
        # Only used to get the port, not the full config
        with open('config.yaml') as f:
            config = yaml.safe_load(f)
        return config['ports'][self.node_id]

    async def start(self):
        # Wait for config from central node
        config = await self.wait_for_config(self.get_port_from_args())
        self.config = config
        self.value = config['initial_values'][self.node_id]
        self.neighbors = config['neighbors'][self.node_id]
        self.ports = config['ports']
        self.hosts = config['hosts']
        # Start server to receive messages from neighbors
        self.server = await asyncio.start_server(self.handle_connection, '0.0.0.0', self.ports[self.node_id])
        print(f"Node {self.node_id} listening on port {self.ports[self.node_id]}")
        print(f"Initial value for node {self.node_id}: {self.value}")
        await asyncio.sleep(1)  # Wait for all nodes to start
        while True:
            await self.exchange_and_update()

    async def exchange_and_update(self):
        # Send value to all neighbors
        for neighbor in self.neighbors:
            await self.send_value(neighbor)

        # Wait for values from all neighbors
        while len(self.received[self.it]) < len(self.neighbors):
            await asyncio.sleep(0.1)

        # Update value (average consensus)
        self.value = self.value + self.sigma * sum(val - self.value for val in self.received[self.it].values())
        print(f"{self.node_id} @ {self.it}: {self.value} ({self.received[self.it]})")
        self.it += 1

    async def send_value(self, neighbor):
        reader, writer = await asyncio.open_connection(self.hosts[neighbor], self.ports[neighbor])
        msg = f"{self.it}:{self.node_id}:{self.value}\n"
        # print(f"Node {self.node_id} sending value to {neighbor}: {msg.strip()}")
        writer.write(msg.encode())
        await writer.drain()
        writer.close()
        await writer.wait_closed()

    async def handle_connection(self, reader, writer):
        data = await reader.readline()
        msg = data.decode().strip()
        it, sender, value = msg.split(":")
        # print(f"Node {self.node_id} received value from {sender}: {value} at iteration {it}")
        self.received[int(it)][sender] = float(value)
        writer.close()
        await writer.wait_closed()


def main():
    if len(sys.argv) != 2:
        print("Usage: python node.py <node_id>")
        sys.exit(1)
    node_id = sys.argv[1]
    node = Node(node_id)
    asyncio.run(node.start())

if __name__ == "__main__":
    main()
