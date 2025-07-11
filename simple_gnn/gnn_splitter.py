import random
import time
from typing import List, Optional

import networkx as nx
import numpy as np
import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import GraphSAGE
from torch_geometric.typing import Adj, OptTensor


class GNNSplitter:
    def __init__(self, model, fake_distributed: bool = False):
        self.model = model
        self.xs: List[Tensor] = [None] * model.num_layers
        self.last_layer_called: Optional[int] = None
        self.fake_distributed = fake_distributed

        # TODO: What if model has explict separate decision layer?
        self.final_layer_idx = model.num_layers - 1

    def __call__(self, layer_idx: Optional[int], *args, **kwargs):
        r"""Call the forward method of the model with the specified layer index."""
        if layer_idx is None:
            return self.forward_all(*args, **kwargs)
        else:
            return self.forward(layer_idx, *args, **kwargs)

    def reset(self):
        r"""Reset the internal state of the splitter."""
        self.xs = [None] * self.model.num_layers
        self.last_layer_called = None

    def update_node(self, layer: int, xi: Tensor, xj_list: list[Tensor]) -> Tensor:
        if isinstance(self.model, GraphSAGE):
            # xi' = W1 * xi + W2 * mean(xj)
            xj = torch.cat(xj_list, dim=0)  # Stack neighbors' features
            xj = xj.mean(dim=0, keepdim=True)

            data = Data(x=(xj, xi), edge_index=torch.tensor([[0, 0]], dtype=torch.long).t())  # type: ignore
            out = self(layer, data.x, data.edge_index)
            if layer != self.final_layer_idx:
                intermediate = self(self.final_layer_idx, out, data.edge_index)
            else:
                intermediate = out
        else:
            raise NotImplementedError(f"{self.model.__class__.__name__} is not supported for layer computation.")

        return out, intermediate  # type: ignore

    def forward(
        self,
        layer_idx: int,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The input node features.
            edge_index (torch.Tensor or SparseTensor): The edge indices.
            edge_weight (torch.Tensor, optional): The edge weights (if
                supported by the underlying GNN layer). (default: :obj:`None`)
            edge_attr (torch.Tensor, optional): The edge features (if supported
                by the underlying GNN layer). (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
                each element to a specific example.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
            batch_size (int, optional): The number of examples :math:`B`.
                Automatically calculated if not given.
                Only needs to be passed in case the underlying normalization
                layers require the :obj:`batch` information.
                (default: :obj:`None`)
        """
        assert len(self.model.convs) == len(self.model.norms)
        # if not self.fake_distributed and self.last_layer_called is not None and layer_idx - 1 != self.last_layer_called:
        #     warnings.warn(f"Calling forward with layer_idx={layer_idx} "
        #                   f"after {self.last_layer_called}. You may be skipping layers.")

        conv = self.model.convs[layer_idx]
        norm = self.model.norms[layer_idx]

        if self.model.supports_edge_weight and self.model.supports_edge_attr:
            x = conv(x, edge_index, edge_weight=edge_weight, edge_attr=edge_attr)
        elif self.model.supports_edge_weight:
            x = conv(x, edge_index, edge_weight=edge_weight)
        elif self.model.supports_edge_attr:
            x = conv(x, edge_index, edge_attr=edge_attr)
        else:
            x = conv(x, edge_index)

        if layer_idx < self.model.num_layers - 1 or self.model.jk_mode is not None:
            if self.model.act is not None and self.model.act_first:
                x = self.model.act(x)
            if self.model.supports_norm_batch:
                x = norm(x, batch, batch_size)
            else:
                x = norm(x)
            if self.model.act is not None and not self.model.act_first:
                x = self.model.act(x)
            x = self.model.dropout(x)
            if hasattr(self.model, "jk"):
                self.xs[layer_idx] = x

        self.last_layer_called = layer_idx

        return x

    def forward_all(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: OptTensor = None,
        edge_attr: OptTensor = None,
        batch: OptTensor = None,
        batch_size: Optional[int] = None,
    ) -> tuple[Tensor, List[Tensor]]:
        r"""Forward pass for all layers."""
        xs = []
        for i in range(len(self.model.convs)):
            x = self.forward(i, x, edge_index, edge_weight, edge_attr, batch, batch_size)
            xs.append(x)

        x = self.model.jk(self.xs) if hasattr(self.model, "jk") else x
        x = self.model.lin(x) if hasattr(self.model, "lin") else x

        return x, xs


def main():
    seed = 3
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

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

    # 3. Initialize a GraphSAGE model
    model = GraphSAGE(in_channels=num_node_features, hidden_channels=4, num_layers=3, out_channels=4)
    model.eval()
    split_model = GNNSplitter(model, fake_distributed=True)  # Use fake_distributed to avoid warnings

    # 4. Compare outputs of the full model and the split model
    with torch.no_grad():
        # Full model output
        t1s = time.perf_counter()
        out_full = model(data.x, data.edge_index)
        t1e = time.perf_counter()

        # Split model output
        t2s = time.perf_counter()
        out_split, hist_split = split_model(None, data.x, data.edge_index)
        t2e = time.perf_counter()

    # NOTE: I can't figure out why the split model is faster.
    print(f"Full model output: {out_full}")
    print(f"Split model output: {out_split}")
    print(f"Difference: {torch.abs(out_full - out_split).max().item()} (should be close to 0)")
    print(
        f"Full model time: {t1e - t1s:.6f}s, Split model time: {t2e - t2s:.6f}s, SPEEDUP: {(t1e - t1s) / (t2e - t2s):.2f}x"
    )
    print()

    # 5. Get indices of node 0's neighbors
    adj = nx.to_numpy_array(G)

    # 6. Run CASE A and CASE B, collect outputs and compare, and time them
    split_model.reset()  # Reset the splitter state
    with torch.no_grad():
        # CASE A: Standard pass
        t0 = time.time()
        out_a = model(data.x, data.edge_index)[0]
        t1 = time.time()

        # CASE B: Loop over all nodes, isolate their and their neighbors' vectors, and repeat for each layer
        t2 = time.perf_counter()
        all_out = [Tensor()] * num_nodes
        for layer in range(model.num_layers):
            for node in range(num_nodes):
                neigh_idx = np.where(adj[node] == 1)[0]
                if layer == 0:
                    node_x = x[node, :].unsqueeze(0)
                    neigh_feats = x[neigh_idx]
                else:
                    node_x = all_out[node]
                    neigh_feats = hist_split[layer - 1][neigh_idx]

                neigh_feats = [tensor.unsqueeze(0) for tensor in neigh_feats]
                out = split_model.update_node(layer, node_x, neigh_feats)
                all_out[node] = out
        t3 = time.perf_counter()

    out_b = all_out[0]

    print(f"CASE A output: {out_a}")
    print(f"CASE B output: {out_b}")
    print(f"Difference: {torch.abs(out_a - out_b).max().item()} (should be close to 0)")
    print(f"CASE A time: {t1 - t0:.6f}s, CASE B time: {t3 - t2:.6f}s, SPEEDUP: {(t1 - t0) / (t3 - t2):.2f}x")


if __name__ == "__main__":
    main()
