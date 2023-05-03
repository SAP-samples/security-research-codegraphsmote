import os
import sys
import copy
import pickle
from typing import List, Dict
from collections import defaultdict

import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch.utils.data import Subset
from tqdm import tqdm

from .datasets import ClassificationDataset
from .datasets.utils import split_dataset_by_ids
from .earlystopping import EarlyStopping
from .utils import CorrectedSummaryWriter, flatten_hparams

EPSILON = sys.float_info.epsilon
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DatasetWithGraphIds(torch.utils.data.Dataset):
    def __init__(self, dataset: ClassificationDataset) -> None:
        super().__init__()
        self.dataset = dataset
        self.graph_ids = dataset.get_graph_identifiers()
    
    def __getitem__(self, index):
        data = self.dataset[index]
        data.graph_id = self.graph_ids[index]
        return data
    
    def __len__(self):
        return len(self.dataset)
    
    def from_subset(subset: Subset):
        return Subset(DatasetWithGraphIds(subset.dataset), subset.indices)

class Trainer():
    def __init__(self, 
                 model,
                 losses: List,
                 metrics: List,
                 dataset: ClassificationDataset,
                 params: Dict):
        self.model = model
        self.il_model1 = copy.deepcopy(model)
        self.il_model2 = copy.deepcopy(model)
        self.losses = losses
        self.metrics = metrics
        self.dataset = dataset
        self.params = params

        self.train_irreducible = self.params.get("train_irreducible", False)
        self.training_on_smote = dataset.get_params().get("type") == "SMOTE"

        if self.params.get("l2") is None:
            self.params["l2"] = 0

        self.global_step = 0
        self.model.to(device)
        self.il_model1.to(device)
        self.il_model2.to(device)
        self.LOG_BASE_DIR = "results/" + params["experiment_name"] + "/"
        if not os.path.exists(self.LOG_BASE_DIR):
            os.mkdir(self.LOG_BASE_DIR)
        
        self._init_writer()
        self.train_set, self.test_set = self.dataset.split()
        self._init_dataloader()

        self.irreducible_losses = None
    
    def _init_writer(self):
        # find empty run dir
        number = 1
        while os.path.exists(os.path.join(self.LOG_BASE_DIR, f"run{number}")):
            number += 1
        self.writer = CorrectedSummaryWriter(os.path.join(self.LOG_BASE_DIR, f"run{number}"))

        hparams = flatten_hparams({**self.model.get_params(), **self.dataset.get_params(), **self.params})

        metrics = self.model.loss_keys()
        for metric in self.metrics:
            metrics.extend(metric.keys())
        for loss in self.losses:
            metrics.extend(loss.keys())
        self.writer.add_hparams(hparams, {**{f"{metric}/train": None for metric in metrics}, 
                                          **{f"{metric}/test": None for metric in metrics}})  # dummy values for metrics
    
    def _init_dataloader(self):
        self.train_loader, self.test_loader = self.build_loaders(self.train_set, self.test_set)
    
    def build_loaders(self, train_set, test_set):
        self.train_set = DatasetWithGraphIds.from_subset(train_set)
        self.test_set = DatasetWithGraphIds.from_subset(test_set)
        train_loader = DataLoader(self.train_set, batch_size=self.params["batch_size"],
                                shuffle=True, num_workers=4, prefetch_factor=1, drop_last=len(self.train_set) > self.params["batch_size"])
        test_loader = DataLoader(self.test_set, batch_size=self.params["batch_size"],
                                shuffle=False, num_workers=4, prefetch_factor=1)
        
        return train_loader, test_loader
    
    def save(self, model=None, train_set=None, test_set=None, path=None):
        if path is None:
            path = self.LOG_BASE_DIR
        if model is None:
            model = self.model
        if train_set is None:
            train_set = self.train_set
        if test_set is None:
            test_set = self.test_set
        if not os.path.exists(path):
            os.makedirs(path)
        self.save_model(model, path)
        train_ids = [self.dataset.get_graph_identifiers()[idx] for idx in train_set.indices]
        test_ids = [self.dataset.get_graph_identifiers()[idx] for idx in test_set.indices]
        with open(os.path.join(path, "train_files.pkl"), "wb") as f:
            pickle.dump(train_ids, f)
        with open(os.path.join(path, "test_files.pkl"), "wb") as f:
            pickle.dump(test_ids, f)
    
    def save_model(self, model, path=None):
        if path is None:
            path = self.LOG_BASE_DIR
        if not os.path.exists(path):
            os.makedirs(path)
        model.save(path)
    
    def load_checkpoint(self):
        path = os.path.join(self.LOG_BASE_DIR, "checkpoint")
        self.load_model(self.model, path=path)

    def load(self, path=None):
        if path is None:
            path = self.LOG_BASE_DIR
        self.load_model(self.model, path=path)
        with open(os.path.join(path, "train_files.pkl"), "rb") as f:
            train_ids = pickle.load(f)
        with open(os.path.join(path, "test_files.pkl"), "rb") as f:
            test_ids = pickle.load(f)
        self.train_set, self.test_set = split_dataset_by_ids(self.dataset, [train_ids, test_ids])

        self._init_dataloader()
    
    def load_model(self, model, path=None):
        if path is None:
            path = self.LOG_BASE_DIR
        model.load(path)
    
    def calc_loss_on(self, model, data, aggregated=True):
        loss_values = list()
        metric_values = {}

        model_pred, model_loss = model.loss(data)
        metric_values = {**metric_values, **model_loss}
        loss_values.extend(model_loss.values())
        for loss in self.losses:
            losses = loss.compute(model_pred, data, aggregated=aggregated)
            metric_values = {**metric_values, **losses}
            loss_values.extend(losses.values())
        if aggregated:
            metric_values["Loss"] = sum(loss_values)
        else:
            metric_values["Loss"] = [sum(s) for s in zip(*loss_values)]

        for metric in self.metrics:
            metrics = metric.compute(model_pred, data, aggregated=True)
            metric_values = {**metric_values, **metrics}
        
        return metric_values


    def train_model_on(self, model, train_loader, test_loader, name="", rho_loss=False):
        save_path = os.path.join(self.LOG_BASE_DIR, name)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.params["learning_rate"], weight_decay=self.params["l2"])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.params["milestones"],
                                                         gamma=self.params["gamma"])
        early_stopping = EarlyStopping(patience=self.params["early_stopping_patience"], verbose=True, 
                                       path=os.path.join(save_path, "checkpoint"))
        
        for epoch in range(self.params["epochs"]):
            model.train()
            metric_values = defaultdict(float)
            progress_bar = tqdm(train_loader, mininterval=1)
            for metric in self.losses + self.metrics:
                metric.reset(train=True)
            n = 0
            for data in progress_bar:
                data.to(device)
                optimizer.zero_grad()

                n_this = len(data)
                self.global_step += n_this
                n += n_this

                metric_values_this = self.calc_loss_on(model, data, aggregated=not rho_loss)
                if rho_loss:
                    irreducible_losses = torch.stack([self.irreducible_losses[graph_id] for graph_id in data.graph_id])
                    losses = torch.stack(metric_values_this["Loss"]) - irreducible_losses
                    losses, _ = torch.sort(losses, descending=True)
                    cutoff = min(int(len(losses)*0.2 + 1), len(losses))
                    losses = losses[:cutoff]
                    loss_value = torch.sum(losses)
                    loss_value.backward()
                else:
                    loss_value = metric_values_this["Loss"]
                    loss_value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                if rho_loss:
                    # aggregate losses
                    for key in metric_values_this.keys():
                        if key != "Loss" and key not in sum(self.model.loss_keys(), *[loss.keys() for loss in self.losses]):
                            continue
                        value = metric_values_this[key]
                        if torch.is_tensor(value):
                            value = torch.mean(value.float())
                        elif type(value) is list:
                            value = sum(value) / len(value)
                        else:
                            print(f"Error: Aggregation not known for {value} with type {type(value)}")
                        metric_values_this[key] = value
                
                for (key, value) in metric_values_this.items():
                    if torch.is_tensor(value):
                        value = value.detach().item()
                    metric_values[key] += value*n_this
                    self.writer.add_scalar(f"{key}/train", value, self.global_step)

                description = f"Epoch {epoch+1}"
                for (key, value) in metric_values.items():
                    if key == "Loss":
                        continue
                    if key not in self.model.loss_keys():
                        continue
                    value = value/(n+EPSILON)
                    description += f" {key}: {value:.3f}"
                for metric in self.losses + self.metrics:
                    for (key, value) in metric.get_aggregated().items():
                        description += f" {key}: {value:.3f}"
                progress_bar.set_description(description, refresh=False)

            scheduler.step()
            test_results = self.evaluate_model_on(model, test_loader)
            if self.params.get("verbose"):
                train_results = self.evaluate_model_on(model, train_loader)
                print("Train " + str(train_results))
                print("Test " + str(test_results))
            early_stopping(test_results["Loss"], lambda p: self.save_model(model, p))
            if early_stopping.early_stop:
                self.save_model(model, path=save_path)
                return
            if self.params.get("save_every_epoch"):
                path = os.path.join(self.LOG_BASE_DIR, f"epoch{epoch}")
                print(f"saving to {path}")
                self.save_model(path=save_path)

    @torch.no_grad()
    def evaluate_model_on(self, model, loader, name=""):
        model.eval()
        metric_values = defaultdict(float)
        
        n = 0
        for metric in self.losses + self.metrics:
            metric.reset(train=(loader == self.train_loader))
        for data in loader:
            data.to(device)

            n_this = len(data)
            n += n_this

            metric_values_this = self.calc_loss_on(model, data)
            
            for (key, value) in metric_values_this.items():
                if torch.is_tensor(value):
                    value = value.detach().item()
                metric_values[key] += value*n_this
        for (key, value) in metric_values.items():
            if key != "Loss" and key not in model.loss_keys():
                continue
            value = value/(n+EPSILON)
            metric_values[key] = value
            self.writer.add_scalar(f"{key}/test", value, self.global_step)
        for metric in self.losses + self.metrics:
            try:
                for (key, value) in metric.get_aggregated().items():
                    metric_values[key] = value
                    self.writer.add_scalar(f"{key}/test", value, self.global_step)
            except:
                for key in metric.keys():
                    metric_values[key] = float("nan")
                    self.writer.add_scalar(f"{key}/test", float("nan"), self.global_step)
        return dict(metric_values)

    def build_irreducible(self):
        
        # Building training sets
        indices1 = list()
        indices2 = list()

        relevant_indices_set = set(self.train_set.indices)
        if self.training_on_smote:
            relevant_indices_set = set(idx for idx in relevant_indices_set if self.dataset.x[idx][2] < 0)
        def relevant_intersect(indices):
            return relevant_indices_set & set(indices)
        for clas in self.dataset.get_classes():
            indices = relevant_intersect(self.dataset.get_subset_of(clas).indices)
            indices = list(indices)
            np.random.shuffle(indices)
            midpoint = int(len(indices)/2)
            indices1.extend(indices[:midpoint])
            indices2.extend(indices[midpoint:])

        indices_set = set(indices1 + indices2)
        assert len(set(indices1) & set(indices2)) == 0
        assert len(indices_set & relevant_indices_set) == len(indices_set)
        assert len(indices_set) == len(relevant_indices_set)
        assert len(indices_set) == len(indices1 + indices2)

        train_set1 = Subset(self.dataset, indices1)
        train_set2 = Subset(self.dataset, indices2)
        graph_ids = self.dataset.get_graph_identifiers()
        graph_ids1 = [str(graph_ids[index]) for index in train_set1.indices]
        graph_ids2 = [str(graph_ids[index]) for index in train_set2.indices]

        print("Training irreducible model 1")
        train_loader, test_loader = self.build_loaders(train_set1, train_set2)
        self.train_model_on(self.il_model1, train_loader, test_loader, name="IL_MODEL1", rho_loss=False)
        self.load_model(self.il_model1, os.path.join(self.LOG_BASE_DIR, "IL_MODEL1", "checkpoint"))
        
        print("Training irreducible model 2")
        train_loader, test_loader = self.build_loaders(train_set2, train_set1)
        self.train_model_on(self.il_model2, train_loader, test_loader, name="IL_MODEL2", rho_loss=False)
        self.load_model(self.il_model2, os.path.join(self.LOG_BASE_DIR, "IL_MODEL2", "checkpoint"))

        self.il_model1.eval()
        self.il_model2.eval()

        print("Calculating irreducible loss values")
        self.irreducible_losses = {}
        loader = DataLoader(DatasetWithGraphIds.from_subset(self.train_set), batch_size=1,
                                shuffle=False, num_workers=4, prefetch_factor=1, drop_last=False)
        def calc_loss_on(model, data):
            metrics = self.calc_loss_on(model, data, aggregated=True)
            
            return metrics["Loss"].detach()
        for data in tqdm(loader, mininterval=1):
            data.to(device)
            graph_id = data.graph_id[0] # remove list wrapper from batching
            if graph_id in graph_ids1:
                self.irreducible_losses[graph_id] = calc_loss_on(self.il_model2, data)
            elif graph_id in graph_ids2:
                self.irreducible_losses[graph_id] = calc_loss_on(self.il_model1, data)
            elif self.training_on_smote:
                self.irreducible_losses[graph_id] = (calc_loss_on(self.il_model1, data) + calc_loss_on(self.il_model2, data)) / 2
            else:
                print(f"ERROR for graph id {graph_id}")
        

    def train(self):
        if self.irreducible_losses is None and self.train_irreducible:
            self.build_irreducible()

        self.train_model_on(self.model, self.train_loader, self.test_loader, rho_loss=self.train_irreducible)
        self.save()

    @torch.no_grad()
    def evaluate(self, train_set=False):
        loader = self.train_loader if train_set else self.test_loader

        return self.evaluate_model_on(self.model, loader)
    
    def cleanup(self):
        if not hasattr(self, "writer"):
            return
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __del__(self):
        self.cleanup()