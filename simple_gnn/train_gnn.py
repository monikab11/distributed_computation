import random
from pathlib import Path

import numpy as np
import torch
from dataset import SimpleDataset
from torch_geometric.nn import GCN, GIN, GraphSAGE
from tqdm import tqdm

from my_graphs_dataset import GraphDataset

## Set up these values.
device = "cpu"
seed = 42
split_ratio = 0.8
selected_graph_sizes = {5: -1}

GNN = GraphSAGE
num_layers = 5
hidden_channels = 16

# Load the dataset.
root = Path(__file__).parents[0]
graph_loader = GraphDataset(selection=selected_graph_sizes, seed=seed)
dataset = SimpleDataset(root / "Dataset", graph_loader).to(device)

# Shuffle and split the dataset.
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
dataset = dataset.shuffle()

train_dataset = dataset[: round(len(dataset) * split_ratio)]
test_dataset = dataset[round(len(dataset) * split_ratio) :]

torch.save(list(test_dataset), root / "config" / "test_dataset.pth")

# Load the model.
model = GNN(
    in_channels=dataset.num_features, hidden_channels=hidden_channels, num_layers=num_layers, out_channels=1
).to(device)

# Set up the optimizer.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Set up the loss function.
loss_fn = torch.nn.L1Loss()

# Training loop.
for epoch in tqdm(range(1000)):
    model.train()
    total_loss = 0.0
    for data in train_dataset:
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = loss_fn(out.squeeze(), data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        tqdm.write(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_dataset)}")

# Evaluation loop.
model.eval()
total_loss = 0.0
with torch.no_grad():
    for data in test_dataset:
        out = model(data.x, data.edge_index)
        loss = loss_fn(out.squeeze(), data.y)
        total_loss += loss.item()
        print(f"Out: {out.squeeze()}, Target: {data.y}, Loss: {loss.item()}")

print(f"Test Loss: {total_loss / len(test_dataset)}")

# Save the model and its configuration.
torch.save(
    {
        "architecture": GNN.__name__,
        "in_channels": dataset.num_features,
        "hidden_channels": hidden_channels,
        "num_layers": num_layers,
        "out_channels": 1,
        "state_dict": model.state_dict(),
    },
    root / "config" / "model.pth",
)
