import os
from collections import defaultdict

import torch
import numpy as np
from tqdm import tqdm

from torch.utils.data import Subset
from torch_geometric.utils import subgraph

from . import ClassificationDataset

class FillupDataset(ClassificationDataset):
    def __init__(self, dataset: ClassificationDataset, fill_from: ClassificationDataset, params):
        self.fill_from = fill_from
        self.params = params
        desired_samples = params.get("desired_samples")
        self.overwrite_cache = params["overwrite_cache"]
        
        self.dataset = dataset
        
        self.x = []
        self.y = []
        
        train, test = self.dataset.split()
        self.train_indices = []
        self.test_indices = []
        self.graph_ids = []
        self.indices = []
        self.class_indices = defaultdict(list)
        graph_ids = self.dataset.get_graph_identifiers()
        with torch.no_grad():
            print("indexing train graphs")
            for index in tqdm(train.indices):
                graph_id = graph_ids[index]
                if type(graph_id) in [str, np.str_] and "/" in graph_id:
                    _, graph_id = os.path.split(graph_id)
                self.graph_ids.append(graph_id)
                self.indices.append(index)
                
                data = self.dataset[index]

                i = len(self.indices) - 1
                y = data.y
                if not type(y) is int:
                    y = y.detach().cpu().numpy()[0].item()
                self.class_indices[int(y)].append(i)
                self.x.append(np.asarray([[index, 0]]))
                self.y.append(y)
                self.train_indices.append(i)
            print("indexing test graphs")
            for index in tqdm(test.indices):
                graph_id = graph_ids[index]
                if type(graph_id) in [str, np.str_] and "/" in graph_id:
                    _, graph_id = os.path.split(graph_id)
                self.graph_ids.append(graph_id)
                self.indices.append(index)
                
                i = len(self.indices) - 1
                y = self.dataset[index].y
                if not type(y) is int:
                    y = y.detach().cpu().numpy()[0].item()
                self.class_indices[int(y)].append(i)
                self.x.append(np.asarray([[index, 0]]))
                self.y.append(y)
                self.test_indices.append(i)
        
        self.original_graph_ids = self.graph_ids

        self.x = np.concatenate(self.x, axis=0)
        self.y = np.asarray(self.y)

        train_x, train_y = self.x[self.train_indices], self.y[self.train_indices]
        test_x, test_y = self.x[self.test_indices], self.y[self.test_indices]

        majority_samples = max(np.sum(train_y == clas) for clas in self.dataset.get_classes())
        minority_samples = min(np.sum(train_y == clas) for clas in self.dataset.get_classes())
        if desired_samples is None:
            desired_samples = min(int(1.1 * majority_samples), int(1.2 * minority_samples))
        desired_samples = min(
            desired_samples,
            *(len(self.fill_from.get_subset_of(clas)) + np.sum(train_y == clas)
                for clas in self.dataset.get_classes())
        )

        if desired_samples < majority_samples:
            all_indices = np.asarray(list(range(len(train_x))))
            new_indices = list()
            for clas in self.dataset.get_classes():
                new_indices.extend(all_indices[train_y == clas][:desired_samples])
            train_x = train_x[new_indices]
            train_y = train_y[new_indices]
        
        extra_x = []
        extra_y = []
        for clas in self.dataset.get_classes():
            clas_indices = self.fill_from.get_subset_of(clas).indices
            clas_samples = np.sum(train_y == clas)
            if clas_samples >= desired_samples:
                continue
            chosen_indices = np.random.choice(clas_indices, desired_samples - clas_samples)
            extra_x.extend(np.asarray([[index, 1]]) for index in chosen_indices)
            extra_y.extend([clas]*(desired_samples - clas_samples))

        self.original_indices = list(range(0, len(train_x)))

        train_x = np.concatenate((train_x, np.concatenate(extra_x, axis=0)), axis=0)
        train_y = np.asarray(train_y.tolist() + extra_y)

        self.x = np.concatenate((train_x, test_x), axis=0)
        self.y = np.concatenate((train_y, test_y), axis=0)
        
        self.train_indices = list(range(0, len(train_x)))
        self.test_indices = list(range(len(train_x), len(self.x)))
        assert len(self.train_indices) == len(train_x)
        assert len(self.test_indices) == len(test_x)

        self.graph_ids = [str(a) for a in self.x]

        self.class_indices = defaultdict(list)
        for i, y in enumerate(self.y):
            self.class_indices[int(y)].append(i)
    
    def get_edge_size(self):
        return self.dataset.get_edge_size()

    def __len__(self):
        return len(self.graph_ids)

    @torch.no_grad()
    def __getitem__(self, index):
        index, dataset_index = self.x[index]
        index = int(index)
        dataset_index = int(dataset_index)
        if dataset_index == 0:
            return self.dataset[index]
        if dataset_index == 1:
            return self.fill_from[index]
    
    def get_graph_identifiers(self):
        return self.graph_ids
    
    def get_input_size(self):
        return self.dataset.get_input_size()
    
    def get_subset_of(self, clas):
        return Subset(self, self.class_indices[clas])
    
    def get_original_trainset(self):
        return Subset(self, self.original_indices)
    
    def get_classes(self):
        return self.class_indices.keys()
    
    def get_params(self):
        return self.params
    
    def split(self, factor=None):
        assert factor is None, "Factor must be the same as params, as it is already pre-split"

        train_dataset = Subset(self, self.train_indices)
        test_dataset = Subset(self, self.test_indices)

        return train_dataset, test_dataset


class NodeDropDataset(ClassificationDataset):
    def __init__(self, dataset: ClassificationDataset, params):
        self.params = params
        desired_samples = params.get("desired_samples")
        self.overwrite_cache = params["overwrite_cache"]
        
        self.dataset = dataset
        
        self.x = []
        self.y = []
        
        train, test = self.dataset.split()
        self.train_indices = []
        self.test_indices = []
        self.graph_ids = []
        self.indices = []
        self.class_indices = defaultdict(list)
        graph_ids = self.dataset.get_graph_identifiers()
        with torch.no_grad():
            print("indexing train graphs")
            for index in tqdm(train.indices):
                graph_id = graph_ids[index]
                if type(graph_id) in [str, np.str_] and "/" in graph_id:
                    _, graph_id = os.path.split(graph_id)
                self.graph_ids.append(graph_id)
                self.indices.append(index)
                
                data = self.dataset[index]

                i = len(self.indices) - 1
                y = data.y
                if not type(y) is int:
                    y = y.detach().cpu().numpy()[0].item()
                self.class_indices[int(y)].append(i)
                self.x.append(np.asarray([[index, 0]]))
                self.y.append(y)
                self.train_indices.append(i)
            print("indexing test graphs")
            for index in tqdm(test.indices):
                graph_id = graph_ids[index]
                if type(graph_id) in [str, np.str_] and "/" in graph_id:
                    _, graph_id = os.path.split(graph_id)
                self.graph_ids.append(graph_id)
                self.indices.append(index)
                
                i = len(self.indices) - 1
                y = self.dataset[index].y
                if not type(y) is int:
                    y = y.detach().cpu().numpy()[0].item()
                self.class_indices[int(y)].append(i)
                self.x.append(np.asarray([[index, 0]]))
                self.y.append(y)
                self.test_indices.append(i)
        
        self.original_graph_ids = self.graph_ids

        self.x = np.concatenate(self.x, axis=0)
        self.y = np.asarray(self.y)

        train_x, train_y = self.x[self.train_indices], self.y[self.train_indices]
        test_x, test_y = self.x[self.test_indices].tolist(), self.y[self.test_indices].tolist()

        majority_samples = max(np.sum(train_y == clas) for clas in self.dataset.get_classes())
        minority_samples = min(np.sum(train_y == clas) for clas in self.dataset.get_classes())
        if desired_samples is None:
            desired_samples = min(int(1.1 * majority_samples), int(1.2 * minority_samples))

        if desired_samples < majority_samples:
            all_indices = np.asarray(list(range(len(train_x))))
            new_indices = list()
            for clas in self.dataset.get_classes():
                new_indices.extend(all_indices[train_y == clas][:desired_samples])
            train_x = train_x[new_indices]
            train_y = train_y[new_indices]
        
        extra_x = []
        extra_y = []
        for clas in self.dataset.get_classes():
            clas_samples = np.sum(train_y == clas)
            if clas_samples >= desired_samples:
                continue
            clas_indices = np.where(train_y == clas)
            if isinstance(clas_indices, tuple):
                clas_indices = clas_indices[0]
            chosen_indices = np.random.choice(clas_indices, desired_samples - clas_samples)
            extra_x.extend((
                (
                    index, 
                    np.random.choice(
                        self.dataset[index].num_nodes,
                        np.random.binomial(n=self.dataset[index].num_nodes-1,p=0.9)+1
                    ).tolist()
                )
                for index in chosen_indices
                ))
            extra_y.extend([clas]*(desired_samples - clas_samples))

        self.original_indices = list(range(0, len(train_x)))

        train_x = train_x.tolist() + extra_x
        train_y = train_y.tolist() + extra_y

        self.x = train_x + test_x
        self.y = train_y + test_y
        
        self.train_indices = list(range(0, len(train_x)))
        self.test_indices = list(range(len(train_x), len(self.x)))
        assert len(self.train_indices) == len(train_x)
        assert len(self.test_indices) == len(test_x)

        self.graph_ids = [str(a) if a[1] != 0 else self.dataset.get_graph_identifiers()[a[0]] for a in self.x]

        self.class_indices = defaultdict(list)
        for i, y in enumerate(self.y):
            self.class_indices[int(y)].append(i)
    
    def get_edge_size(self):
        return self.dataset.get_edge_size()

    def __len__(self):
        return len(self.graph_ids)

    @torch.no_grad()
    def __getitem__(self, index):
        index, dataset_index = self.x[index]
        index = int(index)
        if isinstance(dataset_index, list):
            graph = self.dataset[index]
            
            return self.node_drop(graph, dataset_index)
        if dataset_index == 0:
            return self.dataset[index]
    
    def node_drop(self, graph, subset):
        graph.num_nodes = len(subset)

        graph.x = graph.x[subset, ...]
        graph.edge_index, _ = subgraph(
            subset,
            graph.edge_index,
            relabel_nodes=True
        )

        return graph
    
    def get_graph_identifiers(self):
        return self.graph_ids
    
    def get_input_size(self):
        return self.dataset.get_input_size()
    
    def get_subset_of(self, clas):
        return Subset(self, self.class_indices[clas])
    
    def get_original_trainset(self):
        return Subset(self, self.original_indices)
    
    def get_classes(self):
        return self.class_indices.keys()
    
    def get_params(self):
        return self.params
    
    def split(self, factor=None):
        assert factor is None, "Factor must be the same as params, as it is already pre-split"

        train_dataset = Subset(self, self.train_indices)
        test_dataset = Subset(self, self.test_indices)

        return train_dataset, test_dataset


class EdgeDropDataset(ClassificationDataset):
    def __init__(self, dataset: ClassificationDataset, params):
        self.params = params
        desired_samples = params.get("desired_samples")
        self.overwrite_cache = params["overwrite_cache"]
        
        self.dataset = dataset
        
        self.x = []
        self.y = []
        
        train, test = self.dataset.split()
        self.train_indices = []
        self.test_indices = []
        self.graph_ids = []
        self.indices = []
        self.class_indices = defaultdict(list)
        graph_ids = self.dataset.get_graph_identifiers()
        with torch.no_grad():
            print("indexing train graphs")
            for index in tqdm(train.indices):
                graph_id = graph_ids[index]
                if type(graph_id) in [str, np.str_] and "/" in graph_id:
                    _, graph_id = os.path.split(graph_id)
                self.graph_ids.append(graph_id)
                self.indices.append(index)
                
                data = self.dataset[index]

                i = len(self.indices) - 1
                y = data.y
                if not type(y) is int:
                    y = y.detach().cpu().numpy()[0].item()
                self.class_indices[int(y)].append(i)
                self.x.append(np.asarray([[index, 0]]))
                self.y.append(y)
                self.train_indices.append(i)
            print("indexing test graphs")
            for index in tqdm(test.indices):
                graph_id = graph_ids[index]
                if type(graph_id) in [str, np.str_] and "/" in graph_id:
                    _, graph_id = os.path.split(graph_id)
                self.graph_ids.append(graph_id)
                self.indices.append(index)
                
                i = len(self.indices) - 1
                y = self.dataset[index].y
                if not type(y) is int:
                    y = y.detach().cpu().numpy()[0].item()
                self.class_indices[int(y)].append(i)
                self.x.append(np.asarray([[index, 0]]))
                self.y.append(y)
                self.test_indices.append(i)
        
        self.original_graph_ids = self.graph_ids

        self.x = np.concatenate(self.x, axis=0)
        self.y = np.asarray(self.y)

        train_x, train_y = self.x[self.train_indices], self.y[self.train_indices]
        test_x, test_y = self.x[self.test_indices].tolist(), self.y[self.test_indices].tolist()

        majority_samples = max(np.sum(train_y == clas) for clas in self.dataset.get_classes())
        minority_samples = min(np.sum(train_y == clas) for clas in self.dataset.get_classes())
        if desired_samples is None:
            desired_samples = min(int(1.1 * majority_samples), int(1.2 * minority_samples))

        if desired_samples < majority_samples:
            all_indices = np.asarray(list(range(len(train_x))))
            new_indices = list()
            for clas in self.dataset.get_classes():
                new_indices.extend(all_indices[train_y == clas][:desired_samples])
            train_x = train_x[new_indices]
            train_y = train_y[new_indices]
        
        extra_x = []
        extra_y = []
        for clas in self.dataset.get_classes():
            clas_samples = np.sum(train_y == clas)
            if clas_samples >= desired_samples:
                continue
            clas_indices = np.where(train_y == clas)
            if isinstance(clas_indices, tuple):
                clas_indices = clas_indices[0]
            chosen_indices = np.random.choice(clas_indices, desired_samples - clas_samples)
            extra_x.extend((
                (
                    index, 
                    np.random.choice(
                        len(self.dataset[index].edge_index),
                        np.random.binomial(n=len(self.dataset[index].edge_index)-1,p=0.9)+1
                    ).tolist()
                )
                for index in chosen_indices
                ))
            extra_y.extend([clas]*(desired_samples - clas_samples))

        self.original_indices = list(range(0, len(train_x)))

        train_x = train_x.tolist() + extra_x
        train_y = train_y.tolist() + extra_y

        self.x = train_x + test_x
        self.y = train_y + test_y
        
        self.train_indices = list(range(0, len(train_x)))
        self.test_indices = list(range(len(train_x), len(self.x)))
        assert len(self.train_indices) == len(train_x)
        assert len(self.test_indices) == len(test_x)

        self.graph_ids = [str(a) if a[1] != 0 else self.dataset.get_graph_identifiers()[a[0]] for a in self.x]

        self.class_indices = defaultdict(list)
        for i, y in enumerate(self.y):
            self.class_indices[int(y)].append(i)
    
    def get_edge_size(self):
        return self.dataset.get_edge_size()

    def __len__(self):
        return len(self.graph_ids)

    @torch.no_grad()
    def __getitem__(self, index):
        index, dataset_index = self.x[index]
        index = int(index)
        if isinstance(dataset_index, list):
            graph = self.dataset[index]
            
            return self.edge_drop(graph, dataset_index)
        if dataset_index == 0:
            return self.dataset[index]
    
    def edge_drop(self, graph, subset):
        graph.edge_index = graph.edge_index[:, subset]

        return graph
    
    def get_graph_identifiers(self):
        return self.graph_ids
    
    def get_input_size(self):
        return self.dataset.get_input_size()
    
    def get_subset_of(self, clas):
        return Subset(self, self.class_indices[clas])
    
    def get_original_trainset(self):
        return Subset(self, self.original_indices)
    
    def get_classes(self):
        return self.class_indices.keys()
    
    def get_params(self):
        return self.params
    
    def split(self, factor=None):
        assert factor is None, "Factor must be the same as params, as it is already pre-split"

        train_dataset = Subset(self, self.train_indices)
        test_dataset = Subset(self, self.test_indices)

        return train_dataset, test_dataset