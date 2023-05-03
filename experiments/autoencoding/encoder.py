import os
from abc import ABC, abstractmethod, ABCMeta
from typing import Tuple, Dict, List

import torch

from .helper_types import LatentRepresentation, GraphLatentRepresentation, NodesLatentRepresentation
from .helper_types import Graph, Loss

from .pooling import Pooling


class Encoder(torch.nn.Module, ABC):
    @abstractmethod
    def encode(self, graph: Graph) -> LatentRepresentation:
        pass

    @abstractmethod
    def loss(self, graph: Graph) -> Tuple[LatentRepresentation, Dict[str, Loss]]:
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


class GraphLevelEncoder(Encoder, metaclass=ABCMeta):
    @abstractmethod
    def encode(self, graph: Graph) -> GraphLatentRepresentation:
        pass

    @abstractmethod
    def loss(self, graph: Graph) -> Tuple[GraphLatentRepresentation, Dict[str, Loss]]:
        pass


class NodeLevelEncoder(Encoder, metaclass=ABCMeta):
    @abstractmethod
    def encode(self, graph: Graph) -> NodesLatentRepresentation:
        pass

    @abstractmethod
    def loss(self, graph: Graph) -> Tuple[NodesLatentRepresentation, Dict[str, Loss]]:
        pass

class CompositeGraphLevelEncoder(GraphLevelEncoder):
    def __init__(self, node_level_encoder: NodeLevelEncoder, pooling: Pooling):
        super(CompositeGraphLevelEncoder, self).__init__()
        self.node_level_encoder = node_level_encoder
        self.pooling = pooling
    
    def encode(self, graph: Graph) -> GraphLatentRepresentation:
        nodes_representation = self.node_level_encoder.encode(graph)
        return self.pooling.pool(nodes_representation, graph.batch)
    
    def loss(self, graph: Graph) -> Tuple[GraphLatentRepresentation, Dict[str, Loss]]:
        nodes_representation, encoder_loss = self.node_level_encoder.loss(graph)
        graph_representation, pool_loss = self.pooling.loss(nodes_representation, graph.batch)
        return (graph_representation, {**encoder_loss, **pool_loss})
    
    def loss_keys(self) -> List[str]:
        return [*self.node_level_encoder.loss_keys(), *self.pooling.loss_keys()]
    
    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        self.node_level_encoder.save(os.path.join(path, "encoder"))
        self.pooling.save(os.path.join(path, "pooling"))
    
    def load(self, path):
        self.node_level_encoder.load(os.path.join(path, "encoder"))
        self.pooling.load(os.path.join(path, "pooling"))
    
    def get_params(self):
        return {
            "type": "CompositeGraphLevel",
            "encoder": self.node_level_encoder.get_params(),
            "pooling": self.pooling.get_params()
        }