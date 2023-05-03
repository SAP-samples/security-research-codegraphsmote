from typing import Any
from abc import ABC, ABCMeta
from dataclasses import dataclass


class LatentRepresentation(ABC):
    pass


class GraphLatentRepresentation(LatentRepresentation, metaclass=ABCMeta):
    pass


class NodesLatentRepresentation(LatentRepresentation, metaclass=ABCMeta):
    pass


class Graph(ABC):
    pass


@dataclass
class TupleGraph(Graph):
    x: Any = None
    edge_index: Any = None
    edge_attr: Any = None
    batch: Any = None
    adj: Any = None
    y: Any = None
    num_nodes: Any = None


class Loss(ABC):
    pass


class Prediction(ABC):
    pass