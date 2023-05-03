import os
from abc import ABC, abstractmethod, ABCMeta
from typing import Tuple, Dict, List

import torch
from torch_geometric.utils import dense_to_sparse

from .helper_types import LatentRepresentation, GraphLatentRepresentation, NodesLatentRepresentation
from .helper_types import Graph, Loss, TupleGraph


class Decoder(torch.nn.Module, ABC):
    @abstractmethod
    def decode(self, representation: LatentRepresentation) -> Graph:
        pass

    @abstractmethod
    def loss(self, representation: LatentRepresentation, ground_truth: Graph) -> Tuple[Graph, Dict[str, Loss]]:
        pass
    
    @abstractmethod
    def loss_keys(self) -> List[str]:
        pass
    
    @abstractmethod
    def save(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        pass


class GraphLevelDecoder(Decoder, metaclass=ABCMeta):
    @abstractmethod
    def decode(self, representation: GraphLatentRepresentation) -> Graph:
        pass

    @abstractmethod
    def loss(self, representation: GraphLatentRepresentation, ground_truth: Graph) -> Tuple[Graph, Dict[str, Loss]]:
        pass


class NodeLevelDecoder(Decoder, metaclass=ABCMeta):
    @abstractmethod
    def decode(self, representation: Graph) -> Graph:
        pass

    @abstractmethod
    def loss(self, representation: Graph, ground_truth: Graph) -> Tuple[Graph, Dict[str, Loss]]:
        pass


class CompositeDecoder(NodeLevelDecoder):
    def __init__(self, adj_decoder: NodeLevelDecoder, feature_decoder: NodeLevelDecoder):
        super(CompositeDecoder, self).__init__()
        self.adj_decoder = adj_decoder
        self.feature_decoder = feature_decoder
    
    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        self.adj_decoder.save(os.path.join(path, "adj"))
        self.feature_decoder.save(os.path.join(path, "feature"))
    
    def load(self, path):
        self.adj_decoder.load(os.path.join(path, "adj"))
        self.feature_decoder.load(os.path.join(path, "feature"))
    
    def loss_keys(self):
        return self.adj_decoder.loss_keys() + self.feature_decoder.loss_keys()
    
    def get_params(self):
        return {
            "type": "CompositeDecoder",
            "adj": self.adj_decoder.get_params(),
            "feature": self.feature_decoder.get_params()
        }
    
    def decode(self, representation):
        # with more than 128 dimension the cutoff factor is computed incorrectly
        split = min(representation.x.shape[1] // 2, 128)
        adj_representation = representation.x[:, :split]
        adj_graph = self.adj_decoder.decode(TupleGraph(
            x=adj_representation, batch=representation.batch, num_nodes=representation.num_nodes,
            edge_index=representation.edge_index, adj=representation.adj, y=representation.y
            ))
        adj = torch.clone(adj_graph.adj).detach()
        adj[adj > 0.5] = 1
        adj[adj <= 0.5] = 0
        edge_index, _ = dense_to_sparse(adj)

        feature_representation = representation.x[:, split:]
        feature = self.feature_decoder.decode(TupleGraph(x=feature_representation, edge_index=edge_index))

        return TupleGraph(x=feature.x, adj=adj_graph.adj, num_nodes=representation.num_nodes, batch=representation.batch, y=representation.y)
    
    def loss(self, representation):
        # with more than 128 dimension the cutoff factor is computed incorrectly
        split = min(representation.x.shape[1] // 2, 128)
        adj_representation = representation.x[:, :split]
        adj_graph, adj_loss = self.adj_decoder.loss(TupleGraph(
            x=adj_representation, batch=representation.batch, num_nodes=representation.num_nodes,
            edge_index=representation.edge_index, adj=representation.adj, y=representation.y
            ))
        adj = torch.clone(adj_graph.adj).detach()
        adj[adj > 0.5] = 1
        adj[adj <= 0.5] = 0
        edge_index, _ = dense_to_sparse(adj)

        feature_representation = representation.x[:, split:]
        feature, feature_loss = self.feature_decoder.loss(TupleGraph(x=feature_representation, edge_index=edge_index))
        losses = {**feature_loss, **adj_loss}
        
        graph = TupleGraph(x=feature.x, adj=adj_graph.adj, num_nodes=representation.num_nodes, batch=representation.batch, y=representation.y)
        return (graph, losses)