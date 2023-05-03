import os
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import torch

from .encoder import GraphLevelEncoder
from .helper_types import Loss, GraphLatentRepresentation, Graph, Prediction


class GraphRepresentationClassifier(torch.nn.Module, ABC):
    @abstractmethod
    def classify(self, representation: GraphLatentRepresentation) -> Prediction:
        pass

    @abstractmethod
    def loss(self, representation: GraphLatentRepresentation) -> Tuple[Prediction, Dict]:
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


class GraphClassifier(torch.nn.Module):
    def __init__(self, encoder: GraphLevelEncoder, classifier: GraphRepresentationClassifier):
        super(GraphClassifier, self).__init__()
        self.encoder = encoder
        self.classifier = classifier
    
    def get_params(self) -> Dict:
        return {
            "type": "GraphClassifier",
            "encoder": self.encoder.get_params(),
            "classifier": self.classifier.get_params()
        }
    
    def loss_keys(self) -> List[str]:
        return [*self.encoder.loss_keys(), *self.classifier.loss_keys()]
    
    def loss(self, graph) -> Tuple[Prediction, Dict[str, Loss]]:
        representation, encoder_loss = self.encoder.loss(graph)
        pred, classifier_loss = self.classifier.loss(representation)
        return (pred, {**encoder_loss, **classifier_loss})
    
    def classify(self, graph) -> Prediction:
        representation = self.encoder.encode(graph)
        pred = self.classifier.classify(representation)

        return pred
    
    def save(self, path):
        if not os.path.isdir(path):
            os.mkdir(path)
        self.encoder.save(os.path.join(path, "encoder"))
        self.classifier.save(os.path.join(path, "classifier"))
    
    def load(self, path):
        self.encoder.load(os.path.join(path, "encoder"))
        self.classifier.load(os.path.join(path, "classifier"))


class ClassificationMetric(ABC):
    @abstractmethod
    def compute(self, pred: Prediction, ground_truth: Graph) -> Dict[str, Loss]:
        pass
    
    @abstractmethod
    def keys(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_aggregated(self) -> Dict[str, Loss]:
        pass
    
    @abstractmethod
    def reset(self, train=False):
        pass