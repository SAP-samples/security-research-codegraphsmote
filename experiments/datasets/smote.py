import os
import gzip
import pickle
import dataclasses
from collections import defaultdict

import torch
import numpy as np
import scipy.stats
import torch_geometric
from tqdm import tqdm
from torch.utils.data import Subset
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from scipy.spatial.distance import minkowski
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import dense_to_sparse, to_dense_adj

from . import ClassificationDataset
from ..autoencoding.autoencoder import AutoEncoder
from ..autoencoding.helper_types import TupleGraph


MAX_NODES = 750

class PaddedDataset(ClassificationDataset):
    def __init__(self, params, dataset):
        self.indices = []
        self.train_indices = []
        self.test_indices = []
        self.graph_ids = []
        self.class_indices = defaultdict(list)
        if params.get("num_nodes") is None:
            params["num_nodes"] = 0
            print("Looking for maximum number of nodes")
            train, test = dataset.split()
            graph_ids = dataset.get_graph_identifiers()
            for i in train.indices:
                data = dataset[i]
                if data.num_nodes > MAX_NODES:
                    continue
                params["num_nodes"] = max(params["num_nodes"], data.num_nodes)
                self.indices.append(i)
                self.graph_ids.append(graph_ids[i])
                index = len(self.indices) - 1
                self.train_indices.append(index)
                self.class_indices[int(data.y)].append(index)
            for i in test.indices:
                data = dataset[i]
                if data.num_nodes > MAX_NODES:
                    continue
                params["num_nodes"] = max(params["num_nodes"], data.num_nodes)
                self.indices.append(i)
                self.graph_ids.append(graph_ids[i])
                index = len(self.indices) - 1
                self.test_indices.append(index)
                self.class_indices[int(data.y)].append(index)
        print(f"Using {params['num_nodes']} nodes")
        
        self.params = params

        self.dataset = dataset
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, index):
        data = self.dataset[self.indices[index]]
        if not self.params.get("dont_pad"):
            padding_size = self.params["num_nodes"] - data.num_nodes
            assert padding_size >= 0
            num_nodes = data.num_nodes
            if padding_size > 0:
                data.x = F.pad(data.x, pad=(0, 0, 0, padding_size))
            data.num_nodes = num_nodes
        return data
    
    def get_graph_identifiers(self):
        return self.graph_ids
    
    def get_input_size(self):
        return self.dataset.get_input_size()
    
    def get_subset_of(self, clas):
        return Subset(self, self.class_indices[clas])
    
    def get_edge_size(self):
        return self.dataset.get_edge_size()

    def get_classes(self):
        return self.class_indices.keys()
    
    def get_params(self):
        return {
            "type": "Padded",
            "num_nodes": self.params["num_nodes"],
            "dataset": self.dataset.get_params(),
        }
    
    def split(self, factor=None):
        if factor is not None:
            raise ValueError("Cannot control factor of PaddedDataset, as it is pre-split")
        
        return Subset(self, self.train_indices), Subset(self, self.test_indices)


def autoencode(model, graph, repetitions):
    num_nodes = graph.num_nodes
    device = graph.x.device
    dtype = graph.x.dtype
    hidden_channels = graph.x.shape[1]

    decoded_adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)
    decoded_features = torch.zeros((num_nodes, hidden_channels), device=device, dtype=dtype)
    for i in range(repetitions):
        latent = model.encode(graph)
        latent = latent["mu"] + torch.randn_like(latent["logstd"]) * torch.exp(latent["logstd"])

        decoded = model.decode(TupleGraph(
            x=latent,
            num_nodes=num_nodes,
            batch=torch.zeros((num_nodes,), device=device, dtype=torch.long)
        ))
        decoded_adj += torch.sigmoid(decoded.adj)
        decoded_features += decoded.x
    
    decoded.adj = decoded_adj / repetitions
    decoded.x = decoded_features / repetitions
    decoded_adj = None
    decoded_features = None
    del decoded_adj
    del decoded_features
    
    rand_sample = torch.rand_like(decoded.adj)
    decoded.adj[decoded.adj >= rand_sample] = 1
    decoded.adj[decoded.adj < rand_sample] = 0
    decoded.edge_index, _ = dense_to_sparse(decoded.adj.detach())
    decoded.adj = None

    return decoded


def pad(latent, num_nodes):
    padding_size = num_nodes - latent.shape[0]
    if padding_size > 0:
        latent = F.pad(latent, pad=(0, 0, 0, padding_size))
    elif padding_size < 0:
        latent = latent[:padding_size, :]
    return latent


def pad_adj(adj, num_nodes):
    padding_size = num_nodes - adj.shape[0]
    if padding_size > 0:
        return F.pad(adj, pad=(0, padding_size, 0,  padding_size))
    elif padding_size < 0:
        return adj[:padding_size, :padding_size]
    return adj


def autoencode_interpolated(model, graph1, graph2, ratio, repetitions):
    num_nodes = int(ratio * graph1.num_nodes + (1 - ratio) * graph2.num_nodes)
    device = graph1.x.device
    dtype = graph1.x.dtype
    hidden_channels = graph1.x.shape[1]

    decoded_adj = torch.zeros((num_nodes, num_nodes), device=device, dtype=dtype)
    decoded_features = torch.zeros((num_nodes, hidden_channels), device=device, dtype=dtype)
    for i in range(repetitions):
        latent1 = model.encode(graph1)
        latent2 = model.encode(graph2)
        latent1 = latent1["mu"] + torch.randn_like(latent1["logstd"]) * torch.exp(latent1["logstd"])
        latent2 = latent2["mu"] + torch.randn_like(latent2["logstd"]) * torch.exp(latent2["logstd"])

        latent1 = pad(latent1, num_nodes)
        latent2 = pad(latent2, num_nodes)

        latent = ratio * latent1 + (1 - ratio) * latent2

        decoded = model.decode(TupleGraph(
            x=latent,
            num_nodes=num_nodes,
            batch=torch.zeros((num_nodes,), device=device, dtype=torch.long)
        ))
        decoded_adj += torch.sigmoid(decoded.adj)
        decoded_features += decoded.x
    
    decoded.adj = decoded_adj / repetitions
    decoded.x = decoded_features / repetitions
    decoded_adj = None
    decoded_features = None
    del decoded_adj
    del decoded_features
    
    rand_sample = torch.rand_like(decoded.adj)
    decoded.adj[decoded.adj >= rand_sample] = 1
    decoded.adj[decoded.adj < rand_sample] = 0
    decoded.edge_index, _ = dense_to_sparse(decoded.adj.detach())
    decoded.adj = None

    return decoded


def encode(model, graph, repetitions):
    latent_all = None
    for i in range(repetitions):
        latent = model.encode(graph)
        latent = latent["mu"] + torch.randn_like(latent["logstd"]) * torch.exp(latent["logstd"])
        if latent_all is None:
            latent_all = latent
        else:
            latent_all += latent
    
    latent_all = latent_all / repetitions

    return latent_all


class AutoencodedDataset(ClassificationDataset):
    def __init__(self, model: AutoEncoder, dataset: ClassificationDataset, params):
        self.model = model
        self.model.eval()
        self.params = params
        
        if self.params.get("repetitions") is None:
            self.params["repetitions"] = 10
        if self.params.get("non_categorical_features") is None:
            # assumes categorical features start
            self.params["non_categorical_features"] = 0
        self.overwrite_cache = params["overwrite_cache"]
        self.cache_dir = os.path.join("cache", params["cache_dir"])
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)

        self.dataset = PaddedDataset({"dont_pad": True}, dataset)
        
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
                
                i = len(self.indices) - 1
                y = self.dataset[index].y
                if not type(y) is int:
                    y = y.detach().cpu().numpy()[0].item()
                self.class_indices[int(y)].append(i)
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
                self.test_indices.append(i)
    
    def __len__(self):
        return len(self.graph_ids)

    @torch.no_grad()
    def _get_encoded_by_graph_id(self, graph_id, graph=None):
        encoded_cache = self._encoded_cache_path_for(f"{graph_id}")
        if os.path.exists(encoded_cache) and not self.overwrite_cache:
            try:
                with gzip.open(encoded_cache, "r") as f:
                    return pickle.load(f)
            except EOFError:
                print(f"Re-loading {graph_id}")
        if graph is None:
            index = self.dataset.get_graph_identifiers().index(graph_id)
            graph = self.dataset[index]
        encoded = encode(self.model, graph, self.params["repetitions"]).cpu()
        with gzip.open(encoded_cache, "w") as f:
            pickle.dump(encoded, f)
        return encoded
    
    def get_edge_size(self):
        return self.dataset.get_edge_size()

    @torch.no_grad()
    def _get_encoded(self, index):
        data = self.dataset[self.indices[index]]
        y = data.y
        if not type(y) is int:
            y = y.detach().cpu().numpy()[0].item()
        graph_id = self.graph_ids[index]
        return self._get_encoded_by_graph_id(graph_id, data), y, False
    
    @torch.no_grad()
    def get_reconstruction_metrics(self, index, metrics):
        device = next(self.model.parameters()).device
        real = self.dataset[self.indices[index]].to(device)
        real.batch = torch.zeros((real.num_nodes,), device=real.x.device, dtype=torch.long)
        reconstructed, ret = self.model.loss(real)

        y = real.y
        if not type(y) is int:
            y = y.detach().cpu().numpy()[0].item()

        for metric in metrics:
            metric.reset()
            ret = {**ret, **metric.compute(reconstructed, real, aggregated=True)}
        
        return ret, y
    
    def _cache_path_for(self, index):
        if type(index) in [str, np.str_] and "/" in index:
            _, index = os.path.split(index)
        return os.path.join(self.cache_dir, f"{index}.pt.gz")
    
    def _encoded_cache_path_for(self, index):
        if type(index) in [str, np.str_] and "/" in index:
            _, index = os.path.split(index)
        index = f"encoded_{index}"
        return os.path.join(self.cache_dir, f"{index}.pt.gz")
    
    @torch.no_grad()
    def __getitem__(self, index):
        graph_id = self.graph_ids[index]
        cache_path = self._cache_path_for(graph_id)
        if os.path.isfile(cache_path) and not self.overwrite_cache:
            try:
                with gzip.open(cache_path, "r") as f:
                    data = torch_geometric.data.Data.from_dict(pickle.load(f))
                    return data
            except EOFError:
                print(f"Re-loading {graph_id}")
        
        data = self.dataset[self.indices[index]]
        y = data.y
        if not type(y) is int:
            y = y.detach().cpu().numpy()[0].item()
        x = autoencode(self.model, data, self.params["repetitions"])
        x.x[..., :-self.params["non_categorical_features"]] = torch.softmax(x.x[..., :-self.params["non_categorical_features"]], dim=-1)
        x.y = y
        with gzip.open(cache_path, "w") as f:
            pickle.dump(dataclasses.asdict(x), f)
        return torch_geometric.data.Data.from_dict(dataclasses.asdict(x))
    
    def get_graph_identifiers(self):
        return self.graph_ids
    
    def get_input_size(self):
        return self.dataset.get_input_size()
    
    def get_subset_of(self, clas):
        return Subset(self, self.class_indices[clas])
    
    def get_classes(self):
        return self.class_indices.keys()
    
    def get_params(self):
        return self.params
    
    def split(self, factor=None):
        if factor is not None:
            raise ValueError("Cannot control factor of AutoencodedDataset, as it is pre-split")
        return Subset(self, self.train_indices), Subset(self, self.test_indices)


class SMOTEDataset(ClassificationDataset):
    def __init__(self, model: AutoEncoder, dataset: ClassificationDataset, params):
        self.model = model
        self.model.eval()
        self.params = params
        desired_samples = params.get("desired_samples")
        self.sample_deterministic = params.get("sample_type") != "random"
        if self.params.get("repetitions") is None:
            self.params["repetitions"] = 10
        if self.params.get("non_categorical_features") is None:
            # assumes categorical features start
            self.params["non_categorical_features"] = 0
        self.overwrite_cache = params["overwrite_cache"]
        self.cache_dir = os.path.join("cache", params["cache_dir"])
        if not os.path.exists(self.cache_dir):
            os.mkdir(self.cache_dir)
        
        self.dataset = PaddedDataset({"dont_pad": True}, dataset)
        
        self.x = []
        self.y = []
        
        train, test = self.dataset.split()
        self.train_indices = []
        self.test_indices = []
        self.graph_ids = []
        self.indices = []
        self.class_indices = defaultdict(list)
        graph_ids = self.dataset.get_graph_identifiers()
        encoded_train = []
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
                self.x.append(np.asarray([[index, 0, -1]]))
                self.y.append(y)
                self.train_indices.append(i)

                encoded_cache = self._encoded_cache_path_for(f"{graph_id}")
                if os.path.exists(encoded_cache) and not self.overwrite_cache:
                    try:
                        with gzip.open(encoded_cache, "r") as f:
                            encoded_train.append(pickle.load(f))
                            continue
                    except EOFError:
                        print(f"Re-loading {graph_id}")
                encoded = encode(model, data, self.params["repetitions"]).cpu()
                with gzip.open(encoded_cache, "w") as f:
                    pickle.dump(encoded, f)
                encoded_train.append(encoded)
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
                self.x.append(np.asarray([[index, 0, -1.0]]))
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

        if desired_samples < majority_samples:
            all_indices = np.asarray(list(range(len(train_x))))
            new_indices = list()
            for clas in self.dataset.get_classes():
                new_indices.extend(all_indices[train_y == clas][:desired_samples])
            train_x = train_x[new_indices]
            train_y = train_y[new_indices]

        def _comparison_stub(u, v):
            u = encoded_train[int(u[0])]
            v = encoded_train[int(v[0])]
            num_nodes = max(u.shape[0], v.shape[0])
            u_pad = pad(u, num_nodes).view(-1)
            v_pad = pad(v, num_nodes).view(-1)
            return minkowski(u_pad, v_pad)
        neighbors = NearestNeighbors(metric=_comparison_stub)
        smote = SMOTE(sampling_strategy= {clas: desired_samples for clas in self.dataset.get_classes()}, k_neighbors=neighbors)
        self.smote = smote
        def _interpolation_stub(X, nn_data, nn_num, rows, cols, steps):
            # https://github.com/scikit-learn-contrib/imbalanced-learn/blob/6176807c9c5d68126a79771b6c0fce329f632d2f/imblearn/over_sampling/_smote/base.py#L108
            # assumes X is array with 3 entries: first index, second index and ratio
            neighbors = nn_data[nn_num[rows, cols]]
            ret = np.zeros_like(neighbors)
            ret[..., 0] = X[rows, 0]
            ret[..., 1] = neighbors[..., 0]
            ret[..., 2] = steps[..., 0]
            return ret
        smote._generate_samples = _interpolation_stub

        self.original_indices = list(range(0, len(train_x)))

        train_x, train_y = smote.fit_resample(train_x, train_y)

        self.x = np.concatenate((train_x, test_x), axis=0)
        self.y = np.concatenate((train_y, test_y), axis=0)
        
        self.train_indices = list(range(0, len(train_x)))
        self.test_indices = list(range(len(train_x), len(self.x)))
        assert len(self.train_indices) == len(train_x)
        assert len(self.test_indices) == len(test_x)

        self.graph_ids = [self._get_cache_id_for(x) for x in self.x]

        self.class_indices = defaultdict(list)
        for i, y in enumerate(self.y):
            self.class_indices[int(y)].append(i)
    
    def _get_cache_id_for(self, x):
        # x is array with shape (3,)
        # [first_index, second_index, ratio]
        # ratio < 0 means no second_index, non-interpolated
        if x[2] < 0:
            return self.original_graph_ids[int(x[0])]
        else:
            ratio = x[2]
            first_graph_id = self.original_graph_ids[int(x[0])]
            second_graph_id = self.original_graph_ids[int(x[1])]
            if first_graph_id > second_graph_id:
                first_graph_id, second_graph_id = second_graph_id, first_graph_id
                ratio = 1 - ratio
            return f"{first_graph_id}__I__{second_graph_id}__I__{ratio}"
    
    def get_edge_size(self):
        return self.dataset.get_edge_size()

    def __len__(self):
        return len(self.graph_ids)
    
    def _cache_path_for(self, index):
        if type(index) in [str, np.str_] and "/" in index:
            _, index = os.path.split(index)
        return os.path.join(self.cache_dir, f"{index}.pt.gz")
    
    def _encoded_cache_path_for(self, index):
        if type(index) in [str, np.str_] and "/" in index:
            _, index = os.path.split(index)
        index = f"encoded_{index}"
        return os.path.join(self.cache_dir, f"{index}.pt.gz")

    @torch.no_grad()
    def _get_encoded_by_graph_id(self, graph_id, graph=None):
        encoded_cache = self._encoded_cache_path_for(f"{graph_id}")
        if os.path.exists(encoded_cache) and not self.overwrite_cache:
            try:
                with gzip.open(encoded_cache, "r") as f:
                    return pickle.load(f)
            except EOFError:
                print(f"Re-loading {graph_id}")
        if graph is None:
            index = self.dataset.get_graph_identifiers().index(graph_id)
            graph = self.dataset[index]
        encoded = encode(self.model, graph, self.params["repetitions"]).cpu()
        with gzip.open(encoded_cache, "w") as f:
            pickle.dump(encoded, f)
        return encoded

    @torch.no_grad()
    def _get_encoded(self, index):
        first_index, second_index, ratio = self.x[index]
        if ratio < 0:
            data = self.dataset[self.indices[int(first_index)]]
            y = data.y
            if not type(y) is int:
                y = y.detach().cpu().numpy()[0].item()
            graph_id = self.graph_ids[index]
            return self._get_encoded_by_graph_id(graph_id, data), y, False
        else:
            data1 = self.dataset[self.indices[int(first_index)]]
            data2 = self.dataset[self.indices[int(second_index)]]
            assert data1.y == data2.y, f"{data1.y} != {data2.y}"
            y = data1.y
            if not type(y) is int:
                y = y.detach().cpu().numpy()[0].item()
            first_graph_id = self.original_graph_ids[int(first_index)]
            second_graph_id = self.original_graph_ids[int(second_index)]
            
            num_nodes = int(ratio * data1.num_nodes + (1 - ratio) * data2.num_nodes)

            latent1 = self._get_encoded_by_graph_id(first_graph_id, data1)
            latent2 = self._get_encoded_by_graph_id(second_graph_id, data2)
            latent1 = pad(latent1, num_nodes)
            latent2 = pad(latent2, num_nodes)

            return ratio * latent1 + (1 - ratio) * latent2, y, True

    @torch.no_grad()
    def __getitem__(self, index):
        cache_id = self.graph_ids[index]
        cache_path = self._cache_path_for(cache_id)
        
        first_index, second_index, ratio = self.x[index]
        first_index = int(first_index)
        second_index = int(second_index)
        
        if os.path.isfile(cache_path) and not self.overwrite_cache and ratio >= 0: # TODO: remove ratio
            try:
                with gzip.open(cache_path, "r") as f:
                    data = torch_geometric.data.Data.from_dict(pickle.load(f))
                    return data
            except EOFError:
                print(f"Re-loading {cache_id}")
        
        x = None
        if ratio < 0:
            data = self.dataset[self.indices[first_index]]
            y = data.y
            if not type(y) is int:
                y = y.detach().cpu().numpy()[0].item()
            # x = autoencode(self.model, data, self.params["repetitions"])
            # x.x[..., :-self.params["non_categorical_features"]] = torch.softmax(x.x[..., :-self.params["non_categorical_features"]], dim=-1)
            num_nodes = data.x.shape[0]
            data = TupleGraph(
                x=data.x,
                num_nodes=num_nodes,
                edge_index=data.edge_index,
                batch=torch.zeros((num_nodes,), device=data.x.device, dtype=torch.long),
                y=y,
                adj=None
            )
            return torch_geometric.data.Data.from_dict(dataclasses.asdict(data))
            x.y = y
        else:
            data1 = self.dataset[self.indices[first_index]]
            data2 = self.dataset[self.indices[second_index]]
            assert data1.y == data2.y, f"{data1.y} != {data2.y}"
            y = data1.y
            if not type(y) is int:
                y = y.detach().cpu().numpy()[0].item()
            x = autoencode_interpolated(self.model, data1, data2, ratio, self.params["repetitions"])
            x.x[..., :-self.params["non_categorical_features"]] = torch.softmax(x.x[..., :-self.params["non_categorical_features"]], dim=-1)
            x.y = y
        with gzip.open(cache_path, "w") as f:
            pickle.dump(dataclasses.asdict(x), f)
        return torch_geometric.data.Data.from_dict(dataclasses.asdict(x))
    
    def get_graph_identifiers(self):
        return self.graph_ids
    
    def get_original_trainset(self):
        return Subset(self, self.original_indices)
    
    def get_input_size(self):
        return self.dataset.get_input_size()
    
    def get_subset_of(self, clas):
        return Subset(self, self.class_indices[clas])
    
    def get_classes(self):
        return self.class_indices.keys()
    
    def get_params(self):
        return self.params
    
    def split(self, factor=None):
        assert factor is None, "Factor must be the same as params, as it is already pre-split"

        train_dataset = Subset(self, self.train_indices)
        test_dataset = Subset(self, self.test_indices)

        return train_dataset, test_dataset