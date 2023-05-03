import os.path
from abc import ABC, abstractmethod, ABCMeta
from typing import Tuple, Dict, List

import torch

from .helper_types import LatentRepresentation, GraphLatentRepresentation, NodesLatentRepresentation
from .helper_types import Graph, Loss, TupleGraph
from .encoder import Encoder
from .decoder import Decoder


class AutoEncoderMetric(ABC):
    @abstractmethod
    def reset(self, train=False):
        pass
    
    @abstractmethod
    def get_aggregated(self) -> Dict[str, Loss]:
        pass

    @abstractmethod
    def compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        pass


class AutoEncoder(torch.nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder):
        super(AutoEncoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, graph: Graph) -> LatentRepresentation:
        return self.encoder.encode(graph)
    
    def decode(self, representation: LatentRepresentation) -> Graph:
        return self.decoder.decode(representation)

    def loss(self, ground_truth: Graph) -> Tuple[Graph, Dict[str, Loss]]:
        representation, encoder_loss = self.encoder.loss(ground_truth)
        representation_graph = TupleGraph(
            x=representation, edge_index=ground_truth.edge_index,
            edge_attr=ground_truth.edge_attr,
            batch=ground_truth.batch, num_nodes=ground_truth.num_nodes)
        graph, decoder_loss = self.decoder.loss(representation_graph)
        return (graph, {**encoder_loss, **decoder_loss})
    
    def loss_keys(self) -> List[str]:
        return [*self.encoder.loss_keys(), *self.decoder.loss_keys()]
    
    def save(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        if not os.path.exists(os.path.join(path, "encoder")):
            os.mkdir(os.path.join(path, "encoder"))
        if not os.path.exists(os.path.join(path, "decoder")):
            os.mkdir(os.path.join(path, "decoder"))
        self.encoder.save(os.path.join(path, "encoder"))
        self.decoder.save(os.path.join(path, "decoder"))
    
    def load(self, path):
        self.encoder.load(os.path.join(path, "encoder"))
        self.decoder.load(os.path.join(path, "decoder"))
    
    def get_params(self):
        own_params = {
            "encoder_name": type(self.encoder).__name__,
            "decoder_name": type(self.decoder).__name__
        }
        return {**self.encoder.get_params(), **self.decoder.get_params(), **own_params}