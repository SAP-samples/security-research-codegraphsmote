from abc import ABC, abstractmethod
from typing import Tuple, List, Dict

from .helper_types import GraphLatentRepresentation, NodesLatentRepresentation, Loss


class Pooling(ABC):
    @abstractmethod
    def pool(self, nodes_representation: NodesLatentRepresentation, batch) -> GraphLatentRepresentation:
        pass
    
    @abstractmethod
    def loss(self, nodes_representation: NodesLatentRepresentation, batch) -> Tuple[GraphLatentRepresentation, Dict[str, Loss]]:
        pass
    
    @abstractmethod
    def save(self, path):
        pass
    
    @abstractmethod
    def load(self, path):
        pass
    
    @abstractmethod
    def loss_keys(self) -> List[str]:
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        pass