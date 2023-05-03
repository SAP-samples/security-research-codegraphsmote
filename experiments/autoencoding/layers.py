from typing import Optional, Tuple

import scipy.sparse as sp

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_sparse import SparseTensor, fill_diag, matmul, mul
from torch_sparse import sum as sparsesum

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import add_remaining_self_loops, add_self_loops, coalesce
from torch_geometric.utils import from_scipy_sparse_matrix, to_scipy_sparse_matrix
from torch_geometric.utils.num_nodes import maybe_num_nodes


def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))
        return adj_t

    else:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


def gdn_norm(edge_index, edge_weight=None, num_nodes=None, dtype=None):
    device = edge_index.device
    if isinstance(edge_index, SparseTensor):
        raise NotImplementedError("Sparse adjacency is currently not supported. Please use edge_index")
    else:
        edge_index, edge_normalized = gcn_norm(edge_index, edge_weight, num_nodes=num_nodes, improved=False, add_self_loops=True, dtype=dtype)
        edge_index, edge_normalized = coalesce(edge_index, edge_normalized, num_nodes)

        edge_index, edge_l = add_self_loops(edge_index, -edge_normalized, fill_value=1., num_nodes=num_nodes)
        edge_index, edge_l = coalesce(edge_index, edge_l, num_nodes)
        assert edge_l is not None
        adj_normalized = to_scipy_sparse_matrix(edge_index, edge_normalized, num_nodes)
        sym_l = to_scipy_sparse_matrix(edge_index, edge_l, num_nodes)
        sym_l2 = sym_l.dot(sym_l)
        sym_l3 = sym_l2.dot(sym_l)
        sym_l4 = sym_l3.dot(sym_l)
        true_a = adj_normalized + 0.5 * sym_l2 - 1.0/6 * sym_l3 + 1.0/24 * sym_l4

        edge_index, edge_weight = from_scipy_sparse_matrix(true_a)

        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device, dtype=dtype)

        return edge_index, edge_weight


def gdn_inv_norm(edge_index, edge_weight=None, num_nodes=None, dtype=None):
    device = edge_index.device
    if isinstance(edge_index, SparseTensor):
        raise NotImplementedError("Sparse adjacency is currently not supported. Please use edge_index")
    else:
        edge_index, edge_normalized = gcn_norm(edge_index, edge_weight, num_nodes=num_nodes, improved=False, add_self_loops=True, dtype=dtype)
        edge_index, edge_normalized = coalesce(edge_index, edge_normalized, num_nodes)

        edge_index, edge_l = add_self_loops(edge_index, -edge_normalized, fill_value=1., num_nodes=num_nodes)
        edge_index, edge_l = coalesce(edge_index, edge_l, num_nodes)
        assert edge_l is not None
        adj_normalized = to_scipy_sparse_matrix(edge_index, edge_normalized, num_nodes)
        sym_l = to_scipy_sparse_matrix(edge_index, edge_l, num_nodes)
        sym_l2 = sym_l.dot(sym_l)
        sym_l3 = sym_l2.dot(sym_l)
        sym_l4 = sym_l3.dot(sym_l)
        true_a = sp.eye(adj_normalized.shape[0]) + sym_l + 0.5 * sym_l2 + 1.0/6 * sym_l3 + 1.0/24 * sym_l4

        edge_index, edge_weight = from_scipy_sparse_matrix(true_a)

        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device, dtype=dtype)

        return edge_index, edge_weight


class GDNConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: Optional[int] = None,
                 bias: bool = False, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        if edge_dim is not None:
            self.edge_lin = Linear(edge_dim, out_channels, bias=False,
            weight_initializer='glorot')
        
        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, edge_attr: OptTensor = None) -> Tensor:
        before_edge_index = edge_index
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gdn_norm(
                edge_index, edge_weight, x.size(self.node_dim), torch.float32)
        elif isinstance(edge_index, SparseTensor):
            raise NotImplementedError("Sparse adjacency is currently unsupported. Please use edge_index")
        x = self.lin(x)

        if edge_attr is not None:
            edge_attr = self.edge_lin(edge_attr)
            edge_attr_ = torch.zeros(
                (edge_index.shape[1], self.out_channels),
                device=edge_index.device,
                dtype=edge_attr.dtype
            )
            for i, edge in enumerate(edge_index.T):
                matches = (edge == before_edge_index.T).all(dim=1)
                if matches.any():
                    edge_attr_[i, ...] = torch.sum(edge_attr[matches], dim=0)
            edge_attr = edge_attr_

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            size=None
        )

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor, edge_attr: OptTensor) -> Tensor:
        if edge_attr is not None:
            x_j += edge_attr
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class GDNInvConv(MessagePassing):
    def __init__(self, in_channels: int, out_channels: int, edge_dim: Optional[int] = None,
                 bias: bool = False, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin = Linear(in_channels, out_channels, bias=False,
                          weight_initializer='glorot')

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        if edge_dim is not None:
            self.edge_lin = Linear(edge_dim, out_channels, bias=False,
            weight_initializer='glorot')

        self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None, edge_attr: OptTensor = None) -> Tensor:
        before_edge_index = edge_index
        if isinstance(edge_index, Tensor):
            edge_index, edge_weight = gdn_inv_norm(
                edge_index, edge_weight, x.size(self.node_dim), torch.float32)
        elif isinstance(edge_index, SparseTensor):
            raise NotImplementedError("Sparse adjacency is currently unsupported. Please use edge_index")
        
        x = self.lin(x)
        
        if edge_attr is not None:
            edge_attr = self.edge_lin(edge_attr)
            edge_attr_ = torch.zeros(
                (edge_index.shape[1], self.out_channels),
                device=edge_index.device,
                dtype=edge_attr.dtype
            )
            for i, edge in enumerate(edge_index.T):
                matches = (edge == before_edge_index.T).all(dim=1)
                if matches.any():
                    edge_attr_[i, ...] = torch.sum(edge_attr[matches], dim=0)
            edge_attr = edge_attr_

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(
            edge_index,
            x=x,
            edge_weight=edge_weight,
            edge_attr=edge_attr,
            size=None
        )

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor, edge_attr: OptTensor) -> Tensor:
        if edge_attr is not None:
            x_j += edge_attr
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j