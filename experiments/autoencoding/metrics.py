import math
from abc import abstractmethod, ABCMeta
from typing import Dict, List
from collections import defaultdict

import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.utils import (add_remaining_self_loops, to_undirected, to_dense_adj, degree)
from torch_geometric.utils import negative_sampling, remove_self_loops
from torch_geometric.nn import global_mean_pool, global_add_pool
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

from .autoencoder import AutoEncoderMetric
from .helper_types import Graph, Loss


EPS = 1e-15


def safe_ap(y, pred):
    try:
        return average_precision_score(y, pred)
    except ValueError:
        # if only one value is present, the ap score is undefined
        # use the accuracy in that case
        return accuracy_score(y > 0.5, pred > 0.5) # TODO
    

def safe_roc_auc(y, pred):
    try:
        return roc_auc_score(y, pred)
    except ValueError:
        # if only one value is present, the auc score is undefined
        # use the accuracy in that case
        return accuracy_score(y > 0.5, pred > 0.5) # TODO


class AutoEncoderDummyMetric(AutoEncoderMetric, metaclass=ABCMeta):
    def __init__(self):
        self.aggregated = defaultdict(float)
        self.n = 0
    
    def get_aggregated(self):
        return {key: value/self.n for (key, value) in self.aggregated.items()}
    
    def compute(self, graph: Graph, ground_truth: Graph, aggregated=True) -> Dict[str, Loss]:
        assert aggregated, "Unaggregated compute not supported right now"
        values = self._compute(graph, ground_truth)

        for (key, value) in values.items():
            if type(value) not in [int, float]:
                value = value.item()
            self.aggregated[key] += value
        self.n += 1
        
        return values
    
    @abstractmethod
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        pass

    def reset(self, train=False):
        self.aggregated = defaultdict(float)
        self.n = 0


class SampledEdgesCrossEntropyLoss(AutoEncoderDummyMetric):
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        if isinstance(ground_truth.batch, torch.Tensor):
            src_batch, dst_batch = ground_truth.batch, ground_truth.batch
        else:
            src_batch, dst_batch = ground_truth.batch[0], ground_truth.batch[1]
        
        def edge_index_decode(edge_index):
            return torch.sigmoid(graph.adj[edge_index[0], edge_index[1]])

        pos_edge_index, _ = remove_self_loops(to_undirected(ground_truth.edge_index))
        pos_loss = -torch.log(edge_index_decode(pos_edge_index) + EPS)
        
        num_neg_samples = pos_edge_index.size(1)
        split = degree(src_batch[pos_edge_index[0]], dtype=torch.long).tolist()
        pos_losses = torch.split(pos_loss, split, dim=0)
        
        pos_edge_index, _ = add_remaining_self_loops(pos_edge_index)
        neg_edge_index = negative_sampling(pos_edge_index, ground_truth.num_nodes, force_undirected=True, num_neg_samples=num_neg_samples, method="dense")
        neg_loss = -torch.log(1 - edge_index_decode(neg_edge_index) + EPS)
        
        split = degree(src_batch[neg_edge_index[0]], dtype=torch.long).tolist()
        neg_losses = torch.split(neg_loss, split, dim=0)
        
        pos_loss = torch.stack([torch.mean(loss) for loss in pos_losses])
        neg_loss = torch.stack([torch.mean(loss) for loss in neg_losses])
        return {"EdgCE": torch.mean(pos_loss + neg_loss)}
    
    def keys(self) -> List[str]:
        return ["EdgCE"]

class SampledDirectedEdgesCrossEntropyLoss(AutoEncoderDummyMetric):
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        if isinstance(ground_truth.batch, torch.Tensor):
            src_batch, dst_batch = ground_truth.batch, ground_truth.batch
        else:
            src_batch, dst_batch = ground_truth.batch[0], ground_truth.batch[1]
        
        def edge_index_decode(edge_index):
            return torch.sigmoid(graph.adj[edge_index[0], edge_index[1]])

        pos_edge_index, _ = remove_self_loops(ground_truth.edge_index)
        pos_loss = -torch.log(edge_index_decode(pos_edge_index) + EPS)
        
        num_neg_samples = pos_edge_index.size(1)
        split = degree(src_batch[pos_edge_index[0]], dtype=torch.long).tolist()
        pos_losses = torch.split(pos_loss, split, dim=0)
        
        pos_edge_index, _ = add_remaining_self_loops(pos_edge_index)
        neg_edge_index = negative_sampling(pos_edge_index, ground_truth.num_nodes, force_undirected=False, num_neg_samples=num_neg_samples, method="dense")
        neg_loss = -torch.log(1 - edge_index_decode(neg_edge_index) + EPS)
        
        split = degree(src_batch[neg_edge_index[0]], dtype=torch.long).tolist()
        neg_losses = torch.split(neg_loss, split, dim=0)
        
        pos_loss = torch.stack([torch.mean(loss) for loss in pos_losses])
        neg_loss = torch.stack([torch.mean(loss) for loss in neg_losses])
        return {"EdgCE": torch.mean(pos_loss + neg_loss)}
    
    def keys(self) -> List[str]:
        return ["EdgCE"]


class AdjacencyCrossEntropyLoss(AutoEncoderDummyMetric):
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        edge_index = add_remaining_self_loops(ground_truth.edge_index)[0]
        edge_index = to_undirected(edge_index)
        adj = to_dense_adj(edge_index)[0] # removes batch wrapper
        adj_pred = graph.adj

        alpha = torch.sum(adj == 0) / torch.sum(adj == 1)
        if torch.sum(adj == 0) == 0 or torch.sum(adj == 1) == 0:
            alpha = 1.0
        losses = F.binary_cross_entropy_with_logits(adj_pred, adj.float(), pos_weight=alpha, reduction="none")
        if ground_truth.batch is not None:
            losses = torch.mean(global_mean_pool(losses, ground_truth.batch), dim=1)
        else:
            losses = torch.mean(losses)
        return {"AdjCE": torch.mean(losses)}
    
    def keys(self) -> List[str]:
        return ["AdjCE"]


class AdjacencySumDifference(AutoEncoderDummyMetric):
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        edge_index = add_remaining_self_loops(ground_truth.edge_index)[0]
        adj = to_dense_adj(edge_index)[0] # removes batch wrapper
        adj_pred = None
        if hasattr(graph, "adj") and graph.adj is not None:
            adj_pred = graph.adj
        else:
            adj_pred = to_dense_adj(graph.edge_index)[0]

        pred_sums = unbatch(torch.sigmoid(adj_pred), ground_truth.batch)
        real_sums = unbatch(adj, ground_truth.batch)
        pred_sums = [torch.sum(x, dim=1, keepdim=True) for x in pred_sums]
        real_sums = [torch.sum(x, dim=1, keepdim=True) for x in real_sums]
        pred_sums = torch.stack([torch.mean(x) for x in pred_sums])
        real_sums = torch.stack([torch.mean(x) for x in real_sums])
        return {"AdjSum": 0.1*torch.mean(torch.abs(pred_sums - real_sums))}
    
    def keys(self) -> List[str]:
        return ["AdjSum"]


class DirectedAdjacencyCrossEntropyLoss(AutoEncoderDummyMetric):
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        edge_index = add_remaining_self_loops(ground_truth.edge_index)[0]
        adj = to_dense_adj(edge_index)[0] # removes batch wrapper
        adj_pred = None
        if hasattr(graph, "adj") and graph.adj is not None:
            adj_pred = graph.adj
        else:
            adj_pred = to_dense_adj(graph.edge_index)[0]

        alpha = torch.sum(adj < 0.5) / torch.sum(adj >= 0.5)
        losses = F.binary_cross_entropy_with_logits(adj_pred, adj.float(), pos_weight=alpha, reduction="none")
        if ground_truth.batch is not None:
            losses = torch.mean(global_mean_pool(losses, ground_truth.batch), dim=1)
        else:
            losses = torch.mean(losses)
        return {"AdjCE": torch.mean(losses)}
    
    def keys(self) -> List[str]:
        return ["AdjCE"]


class FeaturesMSE(AutoEncoderDummyMetric):
    def __init__(self, start_index=None, end_index=None, *args, **kwargs):
        super(*args, **kwargs)
        self.start_index = start_index
        self.end_index = end_index
    
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        real_x = ground_truth.x
        x = graph.x 
        if self.end_index is not None:
            real_x = real_x[..., :self.end_index]
            x = x[..., :self.end_index]
        if self.start_index is not None:
            real_x = real_x[..., self.start_index:]
            x = x[..., self.start_index:]
        x = F.normalize(x, p=2, dim=-1) # TODO: really normalize? If yes, parametrize
        # x = x * std + mu
        # real_x = real_x * std + mu
        return {"XMSE": F.mse_loss(x, real_x)}
    
    def keys(self) -> List[str]:
        return ["XMSE"]


class FeaturesCos(AutoEncoderDummyMetric):
    def __init__(self, start_index=None, end_index=None, *args, **kwargs):
        super(*args, **kwargs)
        self.start_index = start_index
        self.end_index = end_index
    
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        real_x = ground_truth.x
        x = graph.x
        if self.end_index is not None:
            real_x = real_x[..., :self.end_index]
            x = x[..., :self.end_index]
        if self.start_index is not None:
            real_x = real_x[..., self.start_index:]
            x = x[..., self.start_index:]
        x = F.normalize(x, p=2, dim=-1) # TODO: really normalize? If yes, parametrize
        # x = x * std + mu
        # real_x = real_x * std + mu
        return {"XCOS": F.cosine_embedding_loss(x, real_x, torch.ones(len(real_x), device=real_x.device))}
    
    def keys(self) -> List[str]:
        return ["XCOS"]


class FeaturesCE(AutoEncoderDummyMetric):
    def __init__(self, start_index=None, end_index=None, *args, **kwargs):
        super(*args, **kwargs)
        self.start_index = start_index
        self.end_index = end_index

    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        real_x = ground_truth.x
        x = graph.x
        if self.end_index is not None:
            real_x = real_x[..., :self.end_index]
            x = x[..., :self.end_index]
        if self.start_index is not None:
            real_x = real_x[..., self.start_index:]
            x = x[..., self.start_index:]
        return {"XCE": F.cross_entropy(x, real_x)}
    
    def keys(self) -> List[str]:
        return ["XCE"]


class AdjacencyAUC(AutoEncoderDummyMetric):
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        edge_index = add_remaining_self_loops(ground_truth.edge_index)[0]
        edge_index = to_undirected(edge_index)
        adj = to_dense_adj(edge_index)[0] # removes batch wrapper
        adj_pred = None
        if hasattr(graph, "adj") and graph.adj is not None:
            adj_pred = torch.sigmoid(graph.adj)
        else:
            adj_pred = to_dense_adj(graph.edge_index)[0]

        adjs = unbatch(adj.detach(), ground_truth.batch)
        adj_preds = unbatch(adj_pred.detach(), ground_truth.batch)

        ys = [adj.reshape(-1).cpu() for adj in adjs]
        preds = [adj_pred.reshape(-1).cpu() for adj_pred in adj_preds]

        values = [safe_roc_auc(y, pred) for (y, pred) in zip(ys, preds)]
        return {"AdjAUC": float(np.mean(values))}
    
    def keys(self) -> List[str]:
        return ["AdjAUC"]

class AdjacencyBinaryMetricsGPU(AutoEncoderDummyMetric):
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        edge_index = add_remaining_self_loops(ground_truth.edge_index)[0]
        edge_index = to_undirected(edge_index)
        adj = to_dense_adj(edge_index)
        adj = adj[0] # removes batch wrapper
        adj_pred = None
        if hasattr(graph, "adj") and graph.adj is not None:
            adj_pred = torch.sigmoid(graph.adj)
        else:
            adj_pred = to_dense_adj(graph.edge_index)[0]

        adjs = unbatch(adj.detach(), ground_truth.batch)
        adj_preds = unbatch(adj_pred.detach(), ground_truth.batch)

        ys = [adj.reshape(-1) for adj in adjs]
        preds = [adj_pred.reshape(-1) for adj_pred in adj_preds]

        ys = [y > 0.5 for y in ys]
        preds = [pred > 0.5 for pred in preds]

        def metrics(y, pred):
            metric_values = {}
            correct = (y == pred)
            incorrect = torch.logical_not(correct)
            

            tp = torch.sum(torch.logical_and(correct, pred)).cpu()
            tn = torch.sum(torch.logical_and(correct, torch.logical_not(pred))).cpu()
            fp = torch.sum(torch.logical_and(incorrect, pred)).cpu()
            fn = torch.sum(torch.logical_and(incorrect, torch.logical_not(pred))).cpu()
            
            metric_values["AdjPrec"] = tp / (tp + fp)
            metric_values["AdjRec"] = tp / (tp + fn)
            metric_values["AdjF1"] = 2 * (metric_values["AdjPrec"] * metric_values["AdjRec"]) / \
                (metric_values["AdjPrec"] + metric_values["AdjRec"])
            
            metric_values["AdjNegPrec"] = tn / (tn + fn)
            metric_values["AdjNegRec"] = tn / (tn + fp)
            metric_values["AdjBAcc"] = (metric_values["AdjRec"] + metric_values["AdjNegRec"]) / 2

            for key, value in metric_values.items():
                if math.isnan(float(value)):
                    metric_values[key] = 0.0
            
            return metric_values

        metric_values = defaultdict(float)
        for (y, pred) in zip(ys, preds):
            for (key, value) in metrics(y, pred).items():
                metric_values[key] += value
        metric_values = {key: value/len(ys) for (key, value) in metric_values.items()}
        return metric_values
    
    def keys(self) -> List[str]:
        return ["AdjF1", "AdjBAcc", "AdjPrec", "AdjRec", "AdjNegPrec", "AdjNegRec"]


class AdjacencySelectedBinaryMetricsGPU(AdjacencyBinaryMetricsGPU):
    def __init__(self, *keys):
        super(AdjacencySelectedBinaryMetricsGPU, self).__init__()
        assert all(key in super(AdjacencySelectedBinaryMetricsGPU, self).keys() for key in keys)
        self.selected_keys = keys
    
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        super_values = super(AdjacencySelectedBinaryMetricsGPU, self).compute(graph, ground_truth)
        return {key: super_values[key] for key in self.selected_keys}
    
    def keys(self) -> List[str]:
        return self.selected_keys


class AdjacencyAccuracyGPU(AutoEncoderDummyMetric):
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        edge_index = add_remaining_self_loops(ground_truth.edge_index)[0]
        edge_index = to_undirected(edge_index)
        adj = to_dense_adj(edge_index)
        adj = adj[0] # removes batch wrapper
        adj_pred = None
        if hasattr(graph, "adj") and graph.adj is not None:
            adj_pred = torch.sigmoid(graph.adj)
        else:
            adj_pred = to_dense_adj(graph.edge_index)[0]

        adjs = unbatch(adj.detach(), ground_truth.batch)
        adj_preds = unbatch(adj_pred.detach(), ground_truth.batch)

        ys = [adj.reshape(-1) for adj in adjs]
        preds = [adj_pred.reshape(-1) for adj_pred in adj_preds]

        def accuracy(y, pred):
            value = torch.sum(y == pred) / y.shape[0]
            return value.cpu()

        values = [accuracy(y > 0.5, pred > 0.5) for (y, pred) in zip(ys, preds)]
        return {"AdjAcc": float(np.mean(values))}
    
    def keys(self) -> List[str]:
        return ["AdjAcc"]


class SampledEdgesAP(AutoEncoderDummyMetric):
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        if isinstance(ground_truth.batch, torch.Tensor):
            src_batch, dst_batch = ground_truth.batch, ground_truth.batch
        else:
            src_batch, dst_batch = ground_truth.batch[0], ground_truth.batch[1]
        
        def edge_index_decode(edge_index):
            return torch.sigmoid(graph.adj[edge_index[0], edge_index[1]])

        pos_edge_index, _ = remove_self_loops(to_undirected(ground_truth.edge_index))
        pos_pred = edge_index_decode(pos_edge_index)
        
        num_neg_samples = pos_edge_index.size(1)
        pos_edge_index, _ = add_remaining_self_loops(pos_edge_index)
        neg_edge_index = negative_sampling(pos_edge_index, ground_truth.num_nodes, force_undirected=True, num_neg_samples=num_neg_samples, method="dense")
        neg_pred = edge_index_decode(neg_edge_index)
        
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))
        pred = torch.cat([pos_pred, neg_pred], dim=0).detach().cpu()
        y = torch.cat([pos_y, neg_y], dim=0).detach().cpu()

        return {"EdgAP": safe_ap(y, pred)}
    
    def keys(self) -> List[str]:
        return ["EdgAP"]


class SampledDirectedEdgesAP(AutoEncoderDummyMetric):
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        if isinstance(ground_truth.batch, torch.Tensor):
            src_batch, dst_batch = ground_truth.batch, ground_truth.batch
        else:
            src_batch, dst_batch = ground_truth.batch[0], ground_truth.batch[1]
        
        def edge_index_decode(edge_index):
            return torch.sigmoid(graph.adj[edge_index[0], edge_index[1]])

        pos_edge_index, _ = remove_self_loops(ground_truth.edge_index)
        pos_pred = edge_index_decode(pos_edge_index)
        
        num_neg_samples = pos_edge_index.size(1)
        pos_edge_index, _ = add_remaining_self_loops(pos_edge_index)
        neg_edge_index = negative_sampling(pos_edge_index, ground_truth.num_nodes, force_undirected=False, num_neg_samples=num_neg_samples, method="dense")
        neg_pred = edge_index_decode(neg_edge_index)
        
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))
        pred = torch.cat([pos_pred, neg_pred], dim=0).detach().cpu()
        y = torch.cat([pos_y, neg_y], dim=0).detach().cpu()

        return {"EdgAP": safe_ap(y, pred)}
    
    def keys(self) -> List[str]:
        return ["EdgAP"]


class SampledEdgesAUC(AutoEncoderDummyMetric):
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        if isinstance(ground_truth.batch, torch.Tensor):
            src_batch, dst_batch = ground_truth.batch, ground_truth.batch
        else:
            src_batch, dst_batch = ground_truth.batch[0], ground_truth.batch[1]
        
        def edge_index_decode(edge_index):
            return torch.sigmoid(graph.adj[edge_index[0], edge_index[1]])

        pos_edge_index, _ = remove_self_loops(to_undirected(ground_truth.edge_index))
        pos_pred = edge_index_decode(pos_edge_index)
        
        num_neg_samples = pos_edge_index.size(1)
        pos_edge_index, _ = add_remaining_self_loops(pos_edge_index)
        neg_edge_index = negative_sampling(pos_edge_index, ground_truth.num_nodes, force_undirected=True, num_neg_samples=num_neg_samples, method="dense")
        neg_pred = edge_index_decode(neg_edge_index)
        
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))
        pred = torch.cat([pos_pred, neg_pred], dim=0).detach().cpu()
        y = torch.cat([pos_y, neg_y], dim=0).detach().cpu()

        return {"EdgAUC": safe_roc_auc(y, pred)}
    
    def keys(self) -> List[str]:
        return ["EdgAUC"]


class SampledDirectedEdgesAUC(AutoEncoderDummyMetric):
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        if isinstance(ground_truth.batch, torch.Tensor):
            src_batch, dst_batch = ground_truth.batch, ground_truth.batch
        else:
            src_batch, dst_batch = ground_truth.batch[0], ground_truth.batch[1]
        
        def edge_index_decode(edge_index):
            return torch.sigmoid(graph.adj[edge_index[0], edge_index[1]])

        pos_edge_index, _ = remove_self_loops(ground_truth.edge_index)
        pos_pred = edge_index_decode(pos_edge_index)
        
        num_neg_samples = pos_edge_index.size(1)
        pos_edge_index, _ = add_remaining_self_loops(pos_edge_index)
        neg_edge_index = negative_sampling(pos_edge_index, ground_truth.num_nodes, force_undirected=False, num_neg_samples=num_neg_samples, method="dense")
        neg_pred = edge_index_decode(neg_edge_index)
        
        pos_y = pos_pred.new_ones(pos_pred.size(0))
        neg_y = neg_pred.new_zeros(neg_pred.size(0))
        pred = torch.cat([pos_pred, neg_pred], dim=0).detach().cpu()
        y = torch.cat([pos_y, neg_y], dim=0).detach().cpu()

        return {"EdgAUC": safe_roc_auc(y, pred)}
    
    def keys(self) -> List[str]:
        return ["EdgAUC"]


class AdjacencyAP(AutoEncoderDummyMetric):
    def _compute(self, graph: Graph, ground_truth: Graph) -> Dict[str, Loss]:
        edge_index = add_remaining_self_loops(ground_truth.edge_index)[0]
        edge_index = to_undirected(edge_index)
        adj = to_dense_adj(edge_index)[0] # removes batch wrapper
        adj_pred = None
        if hasattr(graph, "adj") and graph.adj is not None:
            adj_pred = torch.sigmoid(graph.adj)
        else:
            adj_pred = to_dense_adj(graph.edge_index)[0]

        adjs = unbatch(adj.detach(), ground_truth.batch)
        adj_preds = unbatch(adj_pred.detach(), ground_truth.batch)

        ys = [adj.reshape(-1).cpu() for adj in adjs]
        preds = [adj_pred.reshape(-1).cpu() for adj_pred in adj_preds]

        values = [safe_ap(y, pred) for (y, pred) in zip(ys, preds)]
        return {"AdjAP": float(np.mean(values))}
    
    def keys(self) -> List[str]:
        return ["AdjAP"]

# original source: https://github.com/pyg-team/pytorch_geometric/blob/5a4f8687f628fb4deda1d8708a7e84831c611bc2/torch_geometric/utils/unbatch.py#L9
def unbatch(src, batch, dim: int = 0):
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.
    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)
    :rtype: :class:`List[Tensor]`
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    indices = [0] + list(np.cumsum(sizes))
    return [src[indices[i]:indices[i+1], indices[i]:indices[i+1]] for i in range(len(sizes))]

def unbatch_x(src, batch):
    r"""Splits :obj:`src` according to a :obj:`batch`
    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)
    :rtype: :class:`List[Tensor]`
    """
    sizes = degree(batch, dtype=torch.long).tolist()
    indices = [0] + list(np.cumsum(sizes))
    return [src[indices[i]:indices[i+1], :] for i in range(len(sizes))]