from collections import defaultdict

import numpy as np
from torch.utils.data import Subset

from . import ClassificationDataset


class CombinedDataset(ClassificationDataset):
    def __init__(self, datasets):
        self.datasets = datasets

        assert all(self.datasets[0].get_input_size() == dataset.get_input_size() for dataset in self.datasets), "Datasets to be combined need to be of same shape"

        self.indices = list()
        self.class_indices = defaultdict(list)
        self.graph_ids = list()
        for i, dataset in enumerate(self.datasets):
            graph_ids = dataset.get_graph_identifiers()
            for clas in dataset.get_classes():
                indices = dataset.get_subset_of(clas).indices
                self.class_indices[clas].extend(list(range(len(self.indices), len(self.indices)+len(indices))))
                self.indices.extend(list(map(lambda x: (i, x), indices)))
                self.graph_ids.extend([graph_ids[index] for index in indices])
        
        indices = np.random.permutation(len(self.indices))
        self.indices = list(np.asarray(self.indices)[indices])
        self.graph_ids = list(np.asarray(self.graph_ids)[indices])

        reverse_permutation = {val:i for i,val in enumerate(indices)}
        for clas in self.class_indices.keys():
            self.class_indices[clas] = [reverse_permutation[index] for index in self.class_indices[clas]]
    
    def __len__(self):
        return len(self.graph_ids)
    
    def __getitem__(self, index):
        di, i = self.indices[index]
        return self.datasets[di][i]
    
    def get_edge_size(self):
        if all(d.get_edge_size() == self.datasets[0].get_edge_size() for d in self.datasets):
            return self.datasets[0].get_edge_size()
        return None

    def get_classes(self):
        return self.class_indices.keys()
    
    def get_subset_of(self, clas) -> Subset:
        return Subset(self, self.class_indices[clas])

    def get_input_size(self):
        return self.datasets[0].get_input_size()
    
    def get_graph_identifiers(self):
        return self.graph_ids
    
    def get_params(self):
        return {
            "type": "Combined",
            "datasets": [dataset.get_params() for dataset in self.datasets],
        }