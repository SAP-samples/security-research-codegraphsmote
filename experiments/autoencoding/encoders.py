import os.path

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATv2Conv, GINConv, GatedGraphConv
from torch_geometric.nn import BatchNorm, GraphNorm
from torch_geometric.nn import global_mean_pool

from .encoder import NodeLevelEncoder
from .helper_types import TupleGraph
from .layers import GDNConv


MAX_LOGSTD = 10

class NodeVAEEncoder(NodeLevelEncoder):
    def __init__(self, common_encoder: NodeLevelEncoder, mu_encoder: NodeLevelEncoder, logstd_encoder: NodeLevelEncoder):
        super(NodeVAEEncoder, self).__init__()
        self.common_encoder = common_encoder
        self.logstd_encoder = logstd_encoder
        self.mu_encoder = mu_encoder
    
    def save(self, path):
        self.common_encoder.save(os.path.join(path, "common_encoder"))
        self.logstd_encoder.save(os.path.join(path, "logstd_encoder"))
        self.mu_encoder.save(os.path.join(path, "mu_encoder"))
    
    def load(self, path):
        self.common_encoder.load(os.path.join(path, "common_encoder"))
        self.logstd_encoder.load(os.path.join(path, "logstd_encoder"))
        self.mu_encoder.load(os.path.join(path, "mu_encoder"))
    
    def forward(self, x, edge_index, batch):
        common_representation = self.common_encoder(x, edge_index, batch)
        
        logstd = self.logstd_encoder(common_representation, edge_index, batch)
        mu = self.mu_encoder(common_representation, edge_index, batch)

        return {"mu": mu, "logstd": logstd}

    def encode(self, graph):
        return self(graph.x, graph.edge_index, graph.batch)
    
    def loss(self, graph):
        common_representation, common_loss = self.common_encoder.loss(graph)

        common_graph = TupleGraph(x=common_representation, edge_index=graph.edge_index, batch=graph.batch)
        mu, mu_loss = self.mu_encoder.loss(common_graph)
        logstd, logstd_loss = self.logstd_encoder.loss(common_graph)

        logstd.clamp(max=MAX_LOGSTD)

        kl_loss = -0.5 * torch.mean(
            global_mean_pool(torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1).clamp(min=-MAX_LOGSTD, max=MAX_LOGSTD).unsqueeze(dim=1), graph.batch)
        )
        kl_loss = kl_loss / graph.num_nodes

        losses = {**common_loss, **mu_loss, **logstd_loss, "KLLoss": kl_loss}
        return ({"mu": mu, "logstd": logstd}, losses)

    def loss_keys(self):
        return ["KLLoss"] + \
            self.mu_encoder.loss_keys() + \
            self.logstd_encoder.loss_keys() + \
            self.common_encoder.loss_keys()
    
    def get_params(self):
        return {
            "common": self.common_encoder.get_params(),
            "mu": self.mu_encoder.get_params(),
            "logstd": self.logstd_encoder.get_params(),
            "hidden_channels": self.mu_encoder.get_params()["hidden_channels"]
            }


'''
    subclasses need to implement forward(node_features, edge_index, batch)
'''
class AbstractGNNEncoder(NodeLevelEncoder):
    def __init__(self, **params):
        super(AbstractGNNEncoder, self).__init__()
        self.params = params

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        path = os.path.join(path, "statedict.pt")
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        path = os.path.join(path, "statedict.pt")
        self.load_state_dict(torch.load(path))
    
    def loss(self, graph):
        representation = self.encode(graph)
        return (representation, {})
    
    def encode(self, graph):
        return self(graph.x, graph.edge_index, graph.batch, edge_attr=graph.edge_attr)
    
    def loss_keys(self):
        return []
    
    def get_params(self):
        return self.params


class GCNEncoder(AbstractGNNEncoder):
    def __init__(self, **params):
        if params.get("dropout") is None:
            params["dropout"] = 0.0
        super(GCNEncoder, self).__init__(**params)
        
        assert params["num_layers"] > 0
        assert params["layer_type"] == "GCN"

        self.layers = torch.nn.ModuleList([])
        self.norms = torch.nn.ModuleList([])

        for i in range(self.params["num_layers"]):
            if i == 0:
                self.layers.append(GCNConv(params["features"], params["hidden_channels"]))
            else:
                self.layers.append(GCNConv(params["hidden_channels"], params["hidden_channels"]))
            if params["norm_type"] == "None":
                self.norms.append(None)
            elif params["norm_type"] == "BatchNorm":
                self.norms.append(BatchNorm(params["hidden_channels"]))
            elif params["norm_type"] == "GraphNorm":
                self.norms.append(GraphNorm(params["hidden_channels"]))
            else:
                raise ValueError(f"Unknown norm_type {params['norm_type']}")
    
    def forward(self, x, edge_index, batch, *args, **kwargs):
        initial_x = x
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            x_old = x
            x = conv(x, edge_index)
            if norm is not None:
                x = norm(x)
            # x = F.dropout(x, p=self.params["dropout"], training=self.training)
            if i == len(self.layers) - 1:
                x = x
            elif i == 0:
                x = F.relu(x)
            else:
                x = F.relu(x + x_old)
        
        return x


class GGNNEncoder(AbstractGNNEncoder):
    def __init__(self, **params):
        if params.get("dropout") is None:
            params["dropout"] = 0.0
        super(GGNNEncoder, self).__init__(**params)
        
        assert params["num_layers"] > 0
        assert params["layer_type"] == "GGNN"

        self.layers = torch.nn.ModuleList([])
        self.norms = torch.nn.ModuleList([])

        self.ggnn = GatedGraphConv(max(params["features"], params["hidden_channels"]), params["num_layers"])
        self.out = nn.Linear(max(params["features"], params["hidden_channels"]), params["hidden_channels"])
    
    def forward(self, x, edge_index, batch, *args, **kwargs):
        x = self.ggnn(x, edge_index)
        if self.params["features"] > self.params["hidden_channels"]:
            x = self.out(x)
        
        return x


class GINEncoder(AbstractGNNEncoder):
    def __init__(self, **params):
        if params.get("dropout") is None:
            params["dropout"] = 0.0
        if params.get("train_eps") is None:
            params["train_eps"] = False
        super(GINEncoder, self).__init__(**params)
        
        assert params["num_layers"] > 0
        assert params["layer_type"] == "GIN"

        self.layers = torch.nn.ModuleList([])
        self.norms = torch.nn.ModuleList([])

        for i in range(self.params["num_layers"]):
            if i == 0:
                self.layers.append(GINConv(
                    nn.Sequential(
                        nn.Linear(params["features"], params["hidden_channels"]),
                        nn.ReLU(),
                        nn.Linear(params["hidden_channels"], params["hidden_channels"]),
                        nn.ReLU(),
                        nn.BatchNorm1d(params["hidden_channels"]),
                ), train_eps=params["train_eps"]))
            else:
                self.layers.append(GINConv(
                    nn.Sequential(
                        nn.Linear(params["hidden_channels"], params["hidden_channels"]),
                        nn.ReLU(),
                        nn.Linear(params["hidden_channels"], params["hidden_channels"]),
                        nn.ReLU(),
                        nn.BatchNorm1d(params["hidden_channels"]),
                ), train_eps=params["train_eps"]))
            if params["norm_type"] == "None":
                self.norms.append(None)
            elif params["norm_type"] == "BatchNorm":
                self.norms.append(BatchNorm(params["hidden_channels"]))
            elif params["norm_type"] == "GraphNorm":
                self.norms.append(GraphNorm(params["hidden_channels"]))
            else:
                raise ValueError(f"Unknown norm_type {params['norm_type']}")
    
    def forward(self, x, edge_index, batch, *args, **kwargs):
        initial_x = x
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            x_old = x
            x = conv(x, edge_index)
            if norm is not None:
                x = norm(x)
            # x = F.dropout(x, p=self.params["dropout"], training=self.training)
            if i == len(self.layers) - 1:
                x = x
            elif i == 0:
                x = F.relu(x)
            else:
                x = F.relu(x + x_old)
        
        return x


class GDNEncoder(AbstractGNNEncoder):
    def __init__(self, **params):
        if params.get("random_embedding_size") is None:
            params["random_embedding_size"] = 0
        super(GDNEncoder, self).__init__(**params)
        
        assert params["num_layers"] > 0
        assert params["layer_type"] == "GDN"
        assert params["random_embedding_size"] >= 0

        self.layers = torch.nn.ModuleList([])
        self.norms = torch.nn.ModuleList([])

        for i in range(self.params["num_layers"]):
            if i == 0:
                self.layers.append(GDNConv(params["features"]+params["random_embedding_size"], params["hidden_channels"], edge_dim=params["edge_dim"]))
            else:
                self.layers.append(GDNConv(params["hidden_channels"], params["hidden_channels"], edge_dim=params["edge_dim"]))
            if params["norm_type"] == "None":
                self.norms.append(None)
            elif params["norm_type"] == "BatchNorm":
                self.norms.append(BatchNorm(params["hidden_channels"]))
            elif params["norm_type"] == "GraphNorm":
                self.norms.append(GraphNorm(params["hidden_channels"]))
            else:
                raise ValueError(f"Unknown norm_type {params['norm_type']}")
    
    def forward(self, x, edge_index, batch, edge_attr=None, *args, **kwargs):
        if self.params["random_embedding_size"] > 0:
            random_embedding = torch.randn((x.shape[0], self.params["random_embedding_size"]), device=x.device, dtype=x.dtype)
            x = torch.cat((x, random_embedding), dim=1)
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            x_old = x
            x = conv(x, edge_index, edge_attr=edge_attr)
            if norm is not None:
                x = norm(x)
            if i == len(self.layers) - 1:
                # TODO: original paper has selu in last layer
                x = x
            else:
                x = F.relu(x)
        
        return x


class GATEncoder(AbstractGNNEncoder):
    def __init__(self, **params):
        super(GATEncoder, self).__init__(**params)
        
        if params.get("dropout") is None:
            params["dropout"] = 0

        assert params["num_layers"] > 0
        assert params["layer_type"] == "GAT"

        self.layers = torch.nn.ModuleList([])
        self.norms = torch.nn.ModuleList([])

        for i in range(self.params["num_layers"]):
            if i == 0:
                self.layers.append(GATv2Conv(params["features"], params["hidden_channels"], dropout=params["dropout"]))
            else:
                self.layers.append(GATv2Conv(params["hidden_channels"], params["hidden_channels"], dropout=params["dropout"]))
            if params["norm_type"] == "None":
                self.norms.append(None)
            elif params["norm_type"] == "BatchNorm":
                self.norms.append(BatchNorm(params["hidden_channels"]))
            elif params["norm_type"] == "GraphNorm":
                self.norms.append(GraphNorm(params["hidden_channels"]))
            else:
                raise ValueError(f"Unknown norm_type {params['norm_type']}")
    
    def forward(self, x, edge_index, batch, *args, **kwargs):
        initial_x = x
        for i, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            x_old = x
            x = conv(x, edge_index)
            if norm is not None:
                x = norm(x)
            if i == len(self.layers) - 1:
                x = x
            if i == 0:
                x = F.leaky_relu(x)
            else:
                x = F.leaky_relu(x + x_old)
        
        return x