import os
from functools import lru_cache

import torch
import numpy as np
import scipy.stats
from scipy import special, optimize, integrate
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv
from torch_geometric.nn import BatchNorm, GraphNorm
from torch_geometric.nn import global_add_pool
from .metrics import unbatch
from torch_geometric.utils import to_dense_adj, degree

from .decoder import NodeLevelDecoder
from .helper_types import TupleGraph
from .layers import GDNConv, GDNInvConv


class NodeVAEDecoder(NodeLevelDecoder):
    def __init__(self, decoder):
        super(NodeVAEDecoder, self).__init__()
        self.decoder = decoder

    def save(self, path):
        self.decoder.save(path)

    def load(self, path):
        self.decoder.load(path)

    def _sample(self, representation):
        mu = representation.x["mu"]
        logstd = representation.x["logstd"]

        sampled = mu
        # if self.training:
        if True:
            sampled = mu + torch.randn_like(logstd) * torch.exp(logstd)

        return TupleGraph(x=sampled, edge_index=representation.edge_index,
            adj=representation.adj, num_nodes=representation.num_nodes,
            y=representation.y, batch=representation.batch)

    def decode(self, representation):
        sampled = self._sample(representation)
        return self.decoder.decode(sampled)

    def loss(self, representation):
        sampled = self._sample(representation)
        return self.decoder.loss(sampled)

    def get_params(self):
        return self.decoder.get_params()

    def loss_keys(self):
        return self.decoder.loss_keys()


class InnerProductDecoder(NodeLevelDecoder):
    def __init__(self):
        super(InnerProductDecoder, self).__init__()

    def save(self, path):
        pass

    def load(self, path):
        pass

    def decode(self, representation):
        adj = torch.matmul(representation.x, representation.x.t())
        return TupleGraph(adj=adj)

    def loss(self, representation):
        decoded = self.decode(representation)
        return (decoded, {})

    def get_params(self):
        return {}

    def loss_keys(self):
        return []

class DirectedInnerProductDecoder(NodeLevelDecoder):
    def __init__(self, params):
        super(DirectedInnerProductDecoder, self).__init__()

        self.params = params
        self.layers = torch.nn.ModuleList()
        for a in range(params["n_layers"]):
            lin_layer = Linear(params["hidden_channels"], params["hidden_channels"])
            self.layers.append(lin_layer)

        self.std_k = np.sqrt(2 * params["hidden_channels"])
        self.scale_k = np.sqrt(params["hidden_channels"])
        self.hidden_channels = params["hidden_channels"]
        self.targeted_degree = 5

        self.predicted_variance = torch.nn.Parameter(torch.ones((1,), dtype=torch.float32,
            requires_grad=True))

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        path = os.path.join(path, "statedict.pt")
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = os.path.join(path, "statedict.pt")
        self.load_state_dict(torch.load(path))

    def _calc_p(k, n):
        return (k + 0.5) / n

    def _pdf(x, alpha, lambd):
        '''
            pdf of variance gamma distribution
            with mu = beta = 0
            and according to https://en.wikipedia.org/wiki/Variance-gamma_distribution
        '''
        return ((alpha*alpha)**lambd * np.abs(x)**(lambd-0.5) * special.kv(lambd - 0.5, alpha*np.abs(x))) / \
            (np.sqrt(np.pi) * special.gamma(lambd) * (2*alpha)**(lambd - 0.5))

    def _cdf(q, alpha, lambd):
        return integrate.quad(lambda x: DirectedInnerProductDecoder._pdf(x, alpha, lambd), -np.inf, q)[0]

    @lru_cache(maxsize=None)
    def _icdf(q, alpha, lambd):
        # src: https://github.com/scipy/scipy/blob/main/scipy/stats/_distn_infrastructure.py#L1981
        # adapted to our pdf
        factor = 10.
        left, right = -np.inf, np.inf

        if np.isinf(left):
            left = min(-factor, right)
            while DirectedInnerProductDecoder._cdf(left, alpha, lambd)  - q > 0.:
                left, right = left * factor, left
            # left is now such that cdf(left) <= q
            # if right has changed, then cdf(right) > q

        if np.isinf(right):
            right = max(factor, left)
            while DirectedInnerProductDecoder._cdf(right, alpha, lambd)  - q < 0.:
                left, right = right, right * factor
            # right is now such that cdf(right) >= q

        return optimize.brentq(lambda x: DirectedInnerProductDecoder._cdf(x, alpha, lambd) - q, left, right)

    def decode(self, representation):
        Z = representation.x
        Z_hat = Z
        for lin_layer in self.layers:
            Z_hat = lin_layer(Z_hat)

        split = int(Z.shape[1] / 2)
        Z_1, Z_2 = Z[:,:split], Z[:,split:]
        Z_t1, Z_t2 = Z_hat[:,:split].t(), Z_hat[:,split:].t()
        A_pred_up = torch.triu(torch.matmul(Z_1, Z_t1), diagonal=1)
        A_pred_lo = torch.tril(torch.matmul(Z_2, Z_t2), diagonal=-1)

        A_pred = A_pred_up.add(A_pred_lo)

        sizes = degree(representation.batch, dtype=torch.long).tolist()
        indices = [0] + list(np.cumsum(sizes))
        for i in range(len(sizes)):
            if self.targeted_degree >= sizes[i]:
                continue
            targeted_p = 1 - DirectedInnerProductDecoder._calc_p(self.targeted_degree*sizes[i], sizes[i]*sizes[i])
            correction_factor = DirectedInnerProductDecoder._icdf(targeted_p, alpha=0.5, lambd=self.hidden_channels/2)
            A_pred[indices[i]:indices[i+1], indices[i]:indices[i+1]] -= self.predicted_variance * correction_factor/2

        # add self_loops
        # sigmoid(6) ~=~ 1
        A_pred = A_pred.fill_diagonal_(6)

        return TupleGraph(adj=A_pred)

    def loss(self, representation):
        decoded = self.decode(representation)
        return (decoded, {})

    def get_params(self):
        return self.params

    def loss_keys(self):
        return []


class MLPAdjDecoder(NodeLevelDecoder):
    def __init__(self, params):
        super(MLPAdjDecoder, self).__init__()

        self.params = params
        self.layers = torch.nn.ModuleList()
        self.hidden_channels = 2 * params["hidden_channels"]
        for i in range(params["n_layers"]):
            if i == params["n_layers"] - 1:
                lin_layer = Linear(self.hidden_channels, 1)
            else:
                lin_layer = Linear(self.hidden_channels, self.hidden_channels)
            self.layers.append(lin_layer)

        self.std_k = np.sqrt(2 * params["hidden_channels"])
        self.scale_k = np.sqrt(params["hidden_channels"])
        self.hidden_channels = params["hidden_channels"]
        self.targeted_degree = 5

        self.predicted_variance = torch.nn.Parameter(torch.ones((1,), dtype=torch.float32,
            requires_grad=True))

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        path = os.path.join(path, "statedict.pt")
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = os.path.join(path, "statedict.pt")
        self.load_state_dict(torch.load(path))

    # original source: https://github.com/pyg-team/pytorch_geometric/blob/5a4f8687f628fb4deda1d8708a7e84831c611bc2/torch_geometric/utils/unbatch.py#L9
    def _unbatch_x(src, batch):
        r"""Splits :obj:`src` according to a :obj:`batch`
        Args:
            src (Tensor): The source tensor.
            batch (LongTensor): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
                entry in :obj:`src` to a specific example. Must be ordered.
            dim (int, optional): The dimension along which to split the :obj:`src`
                tensor. (default: :obj:`0`)
        :rtype: :class:`List[Tensor]`
        """
        sizes = degree(batch, dtype=torch.long).tolist()
        indices = [0] + list(np.cumsum(sizes))
        return [src[indices[i]:indices[i+1], :] for i in range(len(sizes))]

    def decode(self, representation):
        xs = MLPAdjDecoder._unbatch_x(representation.x, representation.batch)

        index = 0
        adj = torch.zeros(
            (representation.num_nodes, representation.num_nodes),
            device=representation.x.device,
            dtype=representation.x.dtype
            )
        for x in xs:
            # per graph
            # for every column and row
            num_nodes = x.shape[0]
            for row in range(num_nodes):
                row_xs = x[row].expand((num_nodes, -1))
                col_xs = x
                full_x = torch.cat((row_xs, col_xs), dim=1)
                for i, layer in enumerate(self.layers):
                    full_x = layer(full_x)
                    if i < self.params["n_layers"] - 1:
                        full_x = F.relu(full_x)
                adj[row+index, index:index+num_nodes] = full_x.view(-1)
            index += num_nodes

        sizes = degree(representation.batch, dtype=torch.long).tolist()
        indices = [0] + list(np.cumsum(sizes))
        for i in range(len(sizes)):
            if self.targeted_degree >= sizes[i]:
                continue
            targeted_p = 1 - DirectedInnerProductDecoder._calc_p(self.targeted_degree*sizes[i], sizes[i]*sizes[i])
            correction_factor = DirectedInnerProductDecoder._icdf(targeted_p, alpha=0.5, lambd=self.hidden_channels/2)
            adj[indices[i]:indices[i+1], indices[i]:indices[i+1]] -= self.predicted_variance * correction_factor/2

        return TupleGraph(adj=adj)

    def loss(self, representation):
        decoded = self.decode(representation)
        return (decoded, {})

    def get_params(self):
        return self.params

    def loss_keys(self):
        return []


class AbstractGNNDecoder(NodeLevelDecoder):
    def __init__(self, params):
        super(AbstractGNNDecoder, self).__init__()
        self.params = params

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        path = os.path.join(path, "statedict.pt")
        torch.save(self.state_dict(), path)

    def load(self, path):
        path = os.path.join(path, "statedict.pt")
        self.load_state_dict(torch.load(path))

    def loss(self, representation):
        graph = self.decode(representation)
        return (graph, {})

    def decode(self, representation):
        return self(representation)

    def loss_keys(self):
        return []

    def get_params(self):
        return self.params


class GATDecoder(AbstractGNNDecoder):
    def __init__(self, params):
        super(GATDecoder, self).__init__(params)

        if params.get("dropout") is None:
            params["dropout"] = 0

        assert params["num_layers"] > 0
        assert params["layer_type"] == "GAT"

        self.layers = torch.nn.ModuleList([])
        self.norms = torch.nn.ModuleList([])

        for i in range(self.params["num_layers"]):
            if i == 0:
                self.layers.append(GATv2Conv(params["hidden_channels"], params["features"], dropout=params["dropout"]))
            else:
                self.layers.append(GATv2Conv(params["features"], params["features"], dropout=params["dropout"]))
            if self.params.get("norm_type") == "BatchNorm":
                self.norms.append(BatchNorm(params["features"]))
            else:
                self.norms.append(GraphNorm(params["features"]))

    def forward(self, representation, *args, **kwargs):
        initial_x = representation.x
        x = representation.x
        edge_index = representation.edge_index
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            x_old = x
            x = conv(x, edge_index)
            x = norm(x)
            if i == len(self.layers) - 1:
                x = x
            if i == 0:
                x = F.leaky_relu(x)
            else:
                x = F.leaky_relu(x + x_old)

        return TupleGraph(x=x)


class GDNDecoder(AbstractGNNDecoder):
    def __init__(self, params):
        super(GDNDecoder, self).__init__(params)

        if params.get("dropout") is None:
            params["dropout"] = 0

        assert params["layer_type"] == "GDN"

        self.first = GDNInvConv(params["hidden_channels"], params["hidden_channels"])
        self.second = GDNConv(params["hidden_channels"], params["features"])

    def forward(self, representation, *args, **kwargs):
        initial_x = representation.x
        x = representation.x
        edge_index = representation.edge_index

        x = self.first(x, edge_index)
        x = F.relu(x)
        x = self.second(x, edge_index)

        return TupleGraph(x=x)