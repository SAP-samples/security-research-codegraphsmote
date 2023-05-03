import os

import torch

from torch.nn import Conv1d, Linear, MaxPool1d
from torch_geometric.utils import to_dense_batch

from .encoders import GGNNEncoder

class DevignClassifier(torch.nn.Module):
    def __init__(self, params):
        super(DevignClassifier, self).__init__()

        self.params = params

        assert params["classes"] == 1, "DevignClassifier only supports one class output"
        self.encoder = GGNNEncoder(**{
            "num_layers": 6,
            "hidden_channels": 200,
            "layer_type": "GGNN",
            "features": params["features"]
        })
        input_dim = max(params["features"], 200)
        self.conv_l1 = Conv1d(input_dim, input_dim, 3)
        self.maxpool1 = MaxPool1d(3, stride=2)
        self.conv_l2 = Conv1d(input_dim, input_dim, 1)
        self.maxpool2 = MaxPool1d(2, stride=2)

        self.concat_dim = params["features"] + input_dim
        self.conv_l1_for_concat = Conv1d(self.concat_dim, self.concat_dim, 3)
        self.maxpool1_for_concat = MaxPool1d(3, stride=2)
        self.conv_l2_for_concat = Conv1d(self.concat_dim, self.concat_dim, 1)
        self.maxpool2_for_concat = MaxPool1d(2, stride=2)

        self.mlp_z = Linear(in_features=self.concat_dim, out_features=1)
        self.mlp_y = Linear(in_features=input_dim, out_features=1)

        # self.mlp_z = Linear(in_features=self.concat_dim, out_features=1)
        # self.mlp_y = Linear(in_features=input_dim, out_features=1)
        # self.mlp = Linear(in_features=1, out_features=1)
    
    def get_params(self):
        return self.params
    
    def loss_keys(self):
        return []
    
    def loss(self, graph):
        pred = self.classify(graph)
        return (pred, {})
    
    def classify(self, graph):
        representation = self.encoder.encode(graph)
        
        x_i, _ = to_dense_batch(graph.x, graph.batch)
        h_i, _ = to_dense_batch(representation, graph.batch)


        c_i = torch.cat((h_i, x_i), dim=-1)
        batch_size, num_node, _ = c_i.size()
        Y_1 = self.maxpool1(self.conv_l1(h_i.transpose(1, 2)).relu())
        Y_2 = self.maxpool2(self.conv_l2(Y_1).relu()).transpose(1, 2)


        Z_1 = self.maxpool1_for_concat(self.conv_l1_for_concat(c_i.transpose(1, 2)).relu())
        Z_2 = self.maxpool2_for_concat(self.conv_l2_for_concat(Z_1).relu()).transpose(1, 2)
        Y_2 = self.mlp_y(Y_2)
        Z_2 = self.mlp_z(Z_2)

        final = torch.mul(Y_2, Z_2)
        avg = final.squeeze(-1).mean(-1)

        # final = self.mlp(torch.mul(Y_2, Z_2))
        # avg = final.sum(dim=1)

        
        # result = torch.sigmoid(avg).squeeze(dim=-1)
        result = torch.sigmoid(avg)
        
        return result
    
    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        self.encoder.save(os.path.join(path, "encoder"))
        path = os.path.join(path, "statedict.pt")
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        self.encoder.load(os.path.join(path, "encoder"))
        path = os.path.join(path, "statedict.pt")
        self.load_state_dict(torch.load(path))