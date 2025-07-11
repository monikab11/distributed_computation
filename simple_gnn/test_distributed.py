import time

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

seed = None
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)

# 1. Initialize a random connected graph with 5 nodes
num_nodes = 5
while True:
    G = nx.gnp_random_graph(num_nodes, 0.6, seed=seed, directed=False)
    if nx.is_connected(G):
        break
edge_index = torch.tensor(list(G.edges), dtype=torch.long).t().contiguous()
# Add reverse edges for undirected graph
edge_index = torch.cat([edge_index, edge_index[[1, 0], :]], dim=1)

# 2. Add random features to each node (3 features per node)
num_node_features = 3
x = torch.randn((num_nodes, num_node_features), dtype=torch.float)

data = Data(x=x, edge_index=edge_index)


# 3. Initialize a 1-layer GraphSAGE model
class GraphSAGE(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = SAGEConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        return self.conv(x, edge_index)


model = GraphSAGE(num_node_features, 4)
model.eval()

# 4. Calculate mean of features of node 0's neighbors
adj = nx.to_numpy_array(G)
neigh_idx = np.where(adj[0] == 1)[0]
neigh_feats = x[neigh_idx]
mean_neigh_feats = neigh_feats.mean(dim=0, keepdim=True)

# 5. Run CASE A and CASE B, collect outputs and compare, and time them
with torch.no_grad():
    # CASE A: Standard pass
    t0 = time.time()
    out_a = model(data.x, data.edge_index)[0]
    t1 = time.time()

    # CASE B: Isolated node 0, pass node 0 and mean neighbor features as a batch
    data_b = Data(x=(mean_neigh_feats, x[0:1]), edge_index=torch.tensor([[0, 0]], dtype=torch.long).t())
    t2 = time.time()
    out_b = model(data_b.x, data_b.edge_index)[0]
    t3 = time.time()

print(f"CASE A output: {out_a}")
print(f"CASE B output: {out_b}")
print(f"Difference: {torch.abs(out_a - out_b).max().item()} (should be close to 0)")
print(f"CASE A time: {t1 - t0:.6f}s, CASE B time: {t3 - t2:.6f}s, SPEEDUP: {(t1 - t0) / (t3 - t2):.2f}x")
