import numpy as np
from torch.utils.data import Subset

from . import ClassificationDataset


class ReducedDataset(ClassificationDataset):
    def __init__(self, dataset, percentage):
        self.dataset = dataset

        number_graphs = len(self.dataset)

        # assumes base dataset splits into subsets
        train_set, test_set = self.dataset.split()
        train_indices = train_set.indices
        test_indices = test_set.indices

        classes = self.dataset.get_classes()
        train_indices_set = set(train_indices)
        test_indices_set = set(test_indices)
        def train_intersect(indices):
            return train_indices_set & set(indices)
        def test_intersect(indices):
            return test_indices_set & set(indices)
        class_sizes = [len(train_intersect(self.dataset.get_subset_of(clas).indices)) for clas in classes]
        print("Class sizes", class_sizes)

        self.class_indices = {}
        self.graph_ids = []
        self.indices = []
        graph_ids = self.dataset.get_graph_identifiers()
        for clas in classes:
            subset = self.dataset.get_subset_of(clas).indices
            subset = list(train_intersect(subset))
            np.random.shuffle(subset)
            # ensure at least one element per class so that it is learnable
            remaining = min(int(len(subset) * percentage) + 1, len(subset))
            subset = subset[:remaining]
            # self.class_indices[clas] = subset
            self.class_indices[clas] = list(range(len(self.indices), len(self.indices)+len(subset)))
            self.indices.extend(subset)
            self.graph_ids.extend([graph_ids[idx] for idx in subset])
        
        print("Class sizes", [len(clas) for clas in self.class_indices.values()])
        
        self.train_indices = list(range(0, len(self.indices)))
        for clas in classes:
            subset = self.dataset.get_subset_of(clas).indices
            subset = list(test_intersect(subset))
            np.random.shuffle(subset)
            # self.class_indices[clas].extend(subset)
            self.class_indices[clas].extend(list(range(len(self.indices), len(self.indices)+len(subset))))
            self.indices.extend(subset)
            self.graph_ids.extend([graph_ids[idx] for idx in subset])
        
        print("Class sizes", [len(clas) for clas in self.class_indices.values()])

        self.test_indices = list(range(len(self.train_indices), len(self.indices)))

        indices = np.random.permutation(len(self.indices))
        self.indices = list(np.asarray(self.indices)[indices])
        self.graph_ids = list(np.asarray(self.graph_ids)[indices])

        reverse_permutation = {val:i for i,val in enumerate(indices)}
        self.train_indices = [reverse_permutation[index] for index in self.train_indices]
        self.test_indices = [reverse_permutation[index] for index in self.test_indices]
        for clas in classes:
            self.class_indices[clas] = [reverse_permutation[index] for index in self.class_indices[clas]]
    
    def get_edge_size(self):
        return self.dataset.get_edge_size()
    
    def __len__(self):
        return len(self.graph_ids)
    
    def __getitem__(self, index):
        return self.dataset[self.indices[index]]
    
    def get_classes(self):
        return self.class_indices.keys()
    
    def get_subset_of(self, clas) -> Subset:
        return Subset(self, self.class_indices[clas])

    def get_input_size(self):
        return self.dataset.get_input_size()
    
    def get_graph_identifiers(self):
        return self.graph_ids
    
    def get_params(self):
        return {
            "type": "Reduced",
            "dataset": self.dataset.get_params()
        }
    
    def split(self, factor=None):
        if factor is None:
            factor = 0.8
        assert factor == 0.8, "Factor must be the same as params, as it is already pre-split"
    
        train_dataset = Subset(self, self.train_indices)
        test_dataset = Subset(self, self.test_indices)
    
        return train_dataset, test_dataset