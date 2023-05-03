from abc import ABC, abstractmethod
from typing import List, Tuple, Optional

import torch
import numpy as np
from torch.utils.data import Subset


class ClassificationDataset(torch.utils.data.Dataset, ABC):
    @abstractmethod
    def get_classes(self) -> List:
        pass
    
    @abstractmethod
    def get_subset_of(self, clas) -> Subset:
        pass
    
    @abstractmethod
    def get_input_size(self) -> int:
        pass

    @abstractmethod
    def get_edge_size(self) -> Optional[int]:
        pass
    
    @abstractmethod
    def get_graph_identifiers(self) -> List:
        pass
    
    @abstractmethod
    def get_params(self):
        pass
    
    def split(self, factor = 0.8) -> Tuple[Subset, Subset]:
        assert factor > 0
        assert factor < 1

        number_graphs = len(self)

        all_indices = list(range(0, number_graphs))
        np.random.shuffle(all_indices)

        train_indices = all_indices[:int(number_graphs * factor)]
        test_indices = all_indices[int(number_graphs * factor):]

        minimum_train_graphs = max(factor - 0.1, 0.05)
        minimum_test_graphs = max(1 - factor - 0.1, 0.05)

        assert len(train_indices) > minimum_train_graphs * number_graphs
        assert len(test_indices) > minimum_test_graphs * number_graphs
        assert set(train_indices).isdisjoint(set(test_indices))

        train_dataset = Subset(self, train_indices)
        test_dataset = Subset(self, test_indices)

        return train_dataset, test_dataset