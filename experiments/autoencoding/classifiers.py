import os

import torch
from torch.nn import Linear
import torch.nn.functional as F

from .classifier import GraphRepresentationClassifier


class MLPClassifier(GraphRepresentationClassifier):
    def __init__(self, params):
        super(MLPClassifier, self).__init__()
        if params.get("dropout") is None:
            params["dropout"] = 0.0
        self.params = params

        self.layers = torch.nn.ModuleList([])

        hidden_sizes = params.get("hidden_sizes")
        if hidden_sizes is None:
            hidden_sizes = [params["features"]/2**(i+1) for i in range(params["num_layers"]-2)]
        assert len(hidden_sizes) == params["num_layers"]-2
        layer_units = [params["features"]] + hidden_sizes + [params["classes"]]
        for i in range(params["num_layers"]-1):
            self.layers.append(Linear(layer_units[i], layer_units[i+1]))
    
    def forward(self, x):
        for i, lin in enumerate(self.layers):
            x = lin(x)
            if i == len(self.layers) - 1:
                x = x
            else:
                x = F.relu(x)
                x = F.dropout(x, p=self.params["dropout"], training=self.training)
        if self.params["classes"] == 1:
            x = x.reshape(-1)
        return x
    
    def loss_keys(self):
        return []
    
    def loss(self, representation):
        return (self.classify(representation), {})
    
    def get_params(self):
        return self.params
    
    def classify(self, representation):
        return self(representation)

    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        path = os.path.join(path, "statedict.pt")
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        path = os.path.join(path, "statedict.pt")
        self.load_state_dict(torch.load(path))


class IdentityClassifier(GraphRepresentationClassifier):
    def __init__(self, params):
        super(IdentityClassifier, self).__init__()
        self.params = params
    
    def forward(self, x):
        return x
    
    def loss_keys(self):
        return []
    
    def loss(self, representation):
        return (self.classify(representation), {})
    
    def get_params(self):
        return self.params
    
    def classify(self, representation):
        return self(representation)

    def save(self, path):
        pass
    
    def load(self, path):
        pass