import os
from typing import Dict, List

import numpy as np
import torch
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.utils import (add_remaining_self_loops, to_undirected, to_dense_adj, degree)

from .encoder import GraphLevelEncoder
from .decoder import GraphLevelDecoder
from .metrics import unbatch
from .helper_types import GraphLatentRepresentation
from .helper_types import Graph, Loss, TupleGraph


MAX_ALLOWED_NODES = 700

class ReGAEEncoder(GraphLevelEncoder):
    def __init__(self, params):
        super(ReGAEEncoder, self).__init__()
        self.params = params

        assert params["hidden_channels"] > 0
        assert params["hidden_channels"] % 2 == 0
        assert params["patch_dim"] > 0

        patch_dim = params["patch_dim"]
        patch_size = patch_dim*patch_dim
        self.feedforward = Linear(params["hidden_channels"]*2 + patch_size, params["hidden_channels"] * 3)
    
    def transformation(self, x):
        hidden_channels = self.params["hidden_channels"]
        patch_dim = self.params["patch_dim"]
        patch_size = patch_dim*patch_dim
        x0, x1, a = x.split([hidden_channels, hidden_channels, patch_size])
        x = self.feedforward(x)
        z0, z1, x = x.split(self.params["hidden_channels"])
        z0 = torch.sigmoid(z0)
        z1 = torch.sigmoid(z1)
        x = F.leaky_relu(x)
        first = z0 * x0 + (1 - z0) * x1
        return z1 * first + (1 - z1) * x
    
    def loss(self, graph):
        representation = self.encode(graph)
        return (representation, {})
    
    def encode(self, graph):
        return self(graph.x, graph.edge_index, graph.batch)
    
    def loss_keys(self) -> List[str]:
        return []
    
    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        path = os.path.join(path, "statedict.pt")
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        path = os.path.join(path, "statedict.pt")
        self.load_state_dict(torch.load(path))
    
    def get_params(self) -> Dict:
        return self.params
    
    def forward(self, x, edge_index, batch, *args, **kwargs):
        edge_indices = unbatch_edge_index(edge_index, batch)
        
        def adj_pad(adj):
            n = adj.shape[0]
            padding = n % self.params["patch_dim"]
            if padding == 0:
                return adj
            return F.pad(adj, pad=(0, padding, 0, padding), mode="constant", value=-1)

        def adj_patch(adj, first, second):
            patch_dim = self.params["patch_dim"]
            patch = adj[first:first+patch_dim, second:second+patch_dim]
            n = adj.shape[0]
            for i in range(first, first+patch_dim):
                for j in range(second, second+patch_dim):
                    if i > j:
                        continue
                    patch[i - first, j - second] = -1
            return patch

        xs = []
        for edge_index in edge_indices:
            adj = to_dense_adj(edge_index)[0]
            adj = adj_pad(adj)
            n = adj.shape[0] // self.params["patch_dim"] # number of patches
            x_ = torch.empty((n, n, self.params["hidden_channels"]), dtype=x.dtype, device=x.device)
            null = torch.zeros((self.params["hidden_channels"],), dtype=x.dtype, device=x.device)
            # based on zero-indexed matrix, so all indices minus 1
            for k in range(1, n):
                patch = adj_patch(adj, k, k-1)
                cat = torch.cat([null, null, patch.reshape(-1)])
                x_[k, k-1] = self.transformation(cat)
            for i in range(1, n):
                for k in range(1, n-i):
                    patch = adj_patch(adj, i+k, k-1)
                    cat = torch.cat([x_[i+k-1, k-1], x_[i+k, k], patch.reshape(-1)])
                    x_[i+k, k-1] = self.transformation(cat)
            xs.append(x_[n-1, 0, :])
            del x_

        xs = torch.stack(xs)
        return xs


class ReGAEDecoder(GraphLevelDecoder):
    def __init__(self, params):
        super(ReGAEDecoder, self).__init__()

        assert params["hidden_channels"] > 0
        assert params["hidden_channels"] % 2 == 0
        assert params["patch_dim"] > 0
        half_embedding_size = params["hidden_channels"] // 2
        patch_dim = params["patch_dim"]
        patch_size = patch_dim*patch_dim

        self.params = params
        self.fd = Linear(2 * half_embedding_size, 4 * half_embedding_size + 2 * patch_size)
        self.fd1 = Linear(half_embedding_size, 2*half_embedding_size)
        self.fd2 = Linear(half_embedding_size, 2*half_embedding_size)

    def transform_d(self, y):
        half_embedding_size = self.params["hidden_channels"] // 2
        patch_dim = self.params["patch_dim"]
        patch_size = patch_dim*patch_dim

        y1, y2 = y.split(half_embedding_size)
        z1, z2, y1_, y2_, b, c = self.fd(y).split(
            [half_embedding_size, half_embedding_size, half_embedding_size, half_embedding_size, patch_size, patch_size]
        )

        z1 = torch.sigmoid(z1)
        z2 = torch.sigmoid(z2)
        y1_ = z1 * y1 + (1 - z1) * F.leaky_relu(y1_)
        y2_ = z2 * y2 + (1 - z2) * F.leaky_relu(y2_)

        c = c.reshape((patch_dim, patch_dim))
        b = b.reshape((patch_dim, patch_dim))

        return (y1_, y2_, b, c)
    
    def transform_d1(self, y):
        z, y_ = self.fd1(y).split(self.params["hidden_channels"] // 2)
        z = torch.sigmoid(z)
        return z * y + (1 - z) * F.leaky_relu(y_)

    def transform_d2(self, y):
        z, y_ = self.fd2(y).split(self.params["hidden_channels"] // 2)
        z = torch.sigmoid(z)
        return z * y + (1 - z) * F.leaky_relu(y_)

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        path = os.path.join(path, "statedict.pt")
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = os.path.join(path, "statedict.pt")
        self.load_state_dict(torch.load(path))
    
    def internal_decode(self, representation):
        xs = representation.x
        patch_dim = self.params["patch_dim"]

        adjs = []
        cs = []

        for (ix, x) in enumerate(xs):
            # per graph
            y1 = {}
            y2 = {}
            C = {}
            B = {}

            real_number_nodes = torch.sum(representation.batch == ix)

            y1[(0, 0)], y2[(0, 0)] = x.split(self.params["hidden_channels"] // 2)
            i = 0
            n = 0
            while True:
                for k in range(0, i+1):
                    y_ = torch.cat([y1[(i-k, k)], y2[(i-k, k)]])
                    y1_, y2_, B_, C_ = self.transform_d(y_)
                    y1[(i+1-k, k)] = y1_
                    y2[(i-k, k+1)] = y2_
                    for xi in range(0, patch_dim):
                        for yi in range(0, patch_dim):
                            loc_x = (i - k) * patch_dim + xi
                            loc_y = k * patch_dim + yi
                            B[(loc_x, loc_y)] = B_[xi, yi]
                            C[(loc_x, loc_y)] = C_[xi, yi]
                y1[(0, i+1)] = self.transform_d1(y1[(0, i)])
                y2[(i+1, 0)] = self.transform_d2(y2[(i, 0)])
                if (i * patch_dim) > MAX_ALLOWED_NODES:
                    n = (i * patch_dim) + 2
                    break # failsafe
                
                if not self.training:
                    # check exit condition
                    found = False
                    start = max(0, (i-1) * patch_dim + 1)
                    for j in range(start, i*patch_dim+1):
                        summed = 0
                        for k in range(0, j+1):
                            summed += torch.sigmoid(C[(j+1-k, k)])
                        if summed / j < 0.5:
                            n = j + 2
                            found = True
                            break
                    if found:
                        break
                else:
                    # teacher forcing
                    if (i * patch_dim) >= real_number_nodes:
                        n = real_number_nodes
                        break
                i += 1
            adj = torch.zeros((n, n), dtype=xs.dtype, device=xs.device)
            for i in range(0, n):
                for j in range(0, n - i):
                    adj[n - i - 1, j] = B[(i, j)]
                    adj[j, n - i - 1] = B[(i, j)]
            adj.fill_diagonal_(6)
            C_ = torch.zeros((n, n), dtype=xs.dtype, device=xs.device)
            for i in range(0, n):
                for j in range(0, n):
                    C_[i, j] = C.get((i, j), 0)
            cs.append(C_)
            adjs.append(adj)

        return adjs, cs
    
    def combine(self, adjs):
        device = adjs[0].device
        dtype = adjs[0].dtype
        sizes = [adj.shape[0] for adj in adjs]
        n = sum(sizes)
        indices = [0] + list(np.cumsum(sizes))
        batch = torch.zeros((n,), dtype=torch.long, device=device)
        full_adj = torch.zeros((n, n), dtype=dtype, device=device)
        for i, adj in enumerate(adjs):
            full_adj[indices[i]:indices[i+1], indices[i]:indices[i+1]] = adj
            batch[indices[i]:indices[i+1]] = i

        return TupleGraph(adj=full_adj, batch=batch)

    def decode(self, representation):
        adjs, _ = self.internal_decode(representation)

        return self.combine(adjs)

    def loss(self, representation):
        adjs, cs = self.internal_decode(representation)

        edge_index = add_remaining_self_loops(representation.edge_index)[0]
        edge_index = to_undirected(edge_index)
        real_adj = to_dense_adj(edge_index)[0] # removes batch wrapper
        real_adjs = unbatch(real_adj, representation.batch)

        assert len(real_adjs) == len(cs)
        assert len(adjs) == len(cs)

        adj_ces = []
        c_ces = []
        for real_adj, pred_adj, c in zip(real_adjs, adjs, cs):
            real_n = real_adj.shape[0]
            pred_n = pred_adj.shape[0]
            if real_n > pred_n:
                padding = real_n - pred_n
                pred_adj = F.pad(pred_adj, pad=(0, padding, 0, padding), mode="constant", value=0)
            if pred_n > real_n:
                padding = pred_n - real_n
                real_adj = F.pad(real_adj, pad=(0, padding, 0, padding), mode="constant", value=0.5)
            
            real_c = torch.zeros_like(c)
            for i in range(0, pred_n):
                for j in range(0, pred_n):
                    if i+j > real_n - 2:
                        continue
                    real_c[i, j] = 1
            
            alpha = torch.sum(real_adj < 0.5) / torch.sum(real_adj >= 0.5)
            adj_ce = F.binary_cross_entropy_with_logits(pred_adj, real_adj.float(), pos_weight=alpha, reduction="none")
            adj_ce = torch.mean(adj_ce)
            adj_ces.append(adj_ce)

            c_ce = F.binary_cross_entropy_with_logits(c, real_c, reduction="none")
            c_ce = torch.mean(c_ce)
            c_ces.append(c_ce)

        adj_ce = torch.mean(torch.stack(adj_ces))
        c_ce = torch.mean(torch.stack(c_ces))

        return (self.combine(adjs), {"AdjCE": adj_ce, "CCE": c_ce})

    def get_params(self):
        return self.params

    def loss_keys(self):
        return ["AdjCE", "CCE"]


def unbatch_edge_index(edge_index: torch.Tensor, batch: torch.Tensor) -> List[torch.Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.

    :rtype: :class:`List[Tensor]`
    """
    deg = degree(batch, dtype=torch.int64)
    ptr = torch.cat([deg.new_zeros(1), deg.cumsum(dim=0)[:-1]], dim=0)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, dtype=torch.int64).cpu().tolist()
    return edge_index.split(sizes, dim=1)