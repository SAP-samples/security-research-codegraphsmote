import math
from typing import List, Dict

import torch
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score, average_precision_score

from .classifier import ClassificationMetric
from .helper_types import Prediction, Graph, Loss


class CrossEntropyLoss(ClassificationMetric):
    def __init__(self):
        self.aggregated = 0
        self.len = 0

    def reset(self, train=False):
        self.aggregated = 0
        self.len = 0

    def compute(self, pred: Prediction, graph: Graph, aggregated=True) -> Dict[str, Loss]:
        y = graph.y
        if isinstance(y, list):
            y = torch.as_tensor(y, device=pred.device)
        ce = None
        if len(pred.shape) == 1:
            # binary classification
            ce = F.binary_cross_entropy_with_logits(pred, y.float(), reduction="mean" if aggregated else "none")
        else:
            ce = F.cross_entropy(pred, y, reduction="mean" if aggregated else "none")
        self.aggregated += ce.item() if aggregated else torch.mean(ce).item()
        self.len += 1
        return {"CE": ce}
    
    def get_aggregated(self):
        return {"CE": self.aggregated / self.len}
    
    def keys(self) -> List[str]:
        return ["CE"]


class ClassAccuracy(ClassificationMetric):
    def __init__(self):
        self.correct = 0
        self.all = 0
    
    def reset(self, train=False):
        self.correct = 0
        self.all = 0

    def compute(self, pred: Prediction, graph: Graph, aggregated=True) -> Dict[str, Loss]:
        y = graph.y
        if isinstance(y, list):
            y = torch.as_tensor(y, device=pred.device)
        correct, all_ = None, None
        if len(pred.shape) == 1:
            # binary classification
            pred = torch.sigmoid(pred)
            correct = ((pred > 0.5) == (y > 0.5))
            all_ = len(y)
        else:
            pred = F.softmax(pred, dim=-1)
            pred = torch.argmax(pred, dim=-1)
            correct = (pred == y)
            all_ = len(y)
        self.correct += torch.sum(correct).item()
        self.all += all_
        if aggregated:
            return {"Acc": torch.sum(correct) / all_}
        return {"Acc": correct}
    
    def get_aggregated(self):
        return {"Acc": self.correct / self.all}
    
    def keys(self) -> List[str]:
        return ["Acc"]


class ClassAUC(ClassificationMetric):
    def __init__(self):
        self.pred = []
        self.y = []
    
    def reset(self, train=False):
        self.pred = []
        self.y = []

    def compute(self, pred: Prediction, graph: Graph, aggregated=True) -> Dict[str, Loss]:
        assert aggregated, "Area under Curve can be computed only for aggregated"
        y = graph.y
        if isinstance(y, list):
            y = torch.as_tensor(y, device=pred.device)
        y = y.detach().cpu().numpy()
        if len(pred.shape) == 1:
            # binary classification
            pred = torch.sigmoid(pred).detach().cpu().numpy()
        else:
            pred = F.softmax(pred, dim=-1).detach().cpu().numpy()
        self.pred.extend(pred)
        self.y.extend(y)
        try:
            return {"AUC": roc_auc_score(y, pred)}
        except:
            return {"AUC": float("nan")}
    
    def get_aggregated(self):
        return {"AUC": roc_auc_score(self.y, self.pred)}
    
    def keys(self) -> List[str]:
        return ["AUC"]


class ClassAP(ClassificationMetric):
    def __init__(self):
        self.pred = []
        self.y = []
    
    def reset(self, train=False):
        self.pred = []
        self.y = []

    def compute(self, pred: Prediction, graph: Graph, aggregated=True) -> Dict[str, Loss]:
        assert aggregated, "Average Precision Score can be computed only for aggregated"
        y = graph.y
        if isinstance(y, list):
            y = torch.as_tensor(y, device=pred.device)
        y = y.detach().cpu().numpy()
        if len(pred.shape) == 1:
            # binary classification
            pred = torch.sigmoid(pred).detach().cpu().numpy()
        else:
            pred = F.softmax(pred, dim=-1).detach().cpu().numpy()
        self.pred.extend(pred)
        self.y.extend(y)
        try:
            return {"AP": average_precision_score(y, pred)}
        except:
            return {"AP": float("nan")}
    
    def get_aggregated(self):
        return {"AP": average_precision_score(self.y, self.pred)}
    
    def keys(self) -> List[str]:
        return ["AP"]


class ClassPositivesNegatives(ClassificationMetric):
    def __init__(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
    
    def reset(self, train=False):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
    
    def get_aggregated(self):
        if isinstance(self.tp, list):
            return {
                "F1": ClassPositivesNegatives.f1(self.tp, self.tn, self.fp, self.fn),
                "Prec": ClassPositivesNegatives.prec(self.tp, self.tn, self.fp, self.fn),
                "Rec": ClassPositivesNegatives.rec(self.tp, self.tn, self.fp, self.fn),
                "BAcc": ClassPositivesNegatives.bacc(self.tp, self.tn, self.fp, self.fn),
                "MCC": ClassPositivesNegatives.mcc(self.tp, self.tn, self.fp, self.fn),
                "TP": sum(self.tp),
                "TN": sum(self.tn),
                "FP": sum(self.fp),
                "FN": sum(self.fn)
            }
        tp = self.tp
        tn = self.tn
        fp = self.fp
        fn = self.fn
        return {
            "F1": ClassPositivesNegatives.f1(tp, tn, fp, fn),
            "Prec": ClassPositivesNegatives.prec(tp, tn, fp, fn),
            "Rec": ClassPositivesNegatives.rec(tp, tn, fp, fn),
            "BAcc": ClassPositivesNegatives.bacc(tp, tn, fp, fn),
            "MCC": ClassPositivesNegatives.mcc(tp, tn, fp, fn),
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn
        }
    
    def f1(tp, tn, fp, fn):
        if isinstance(tp, list):
            # multiclass
            # macro averaging
            num_classes = len(tp)
            recs = [ClassPositivesNegatives.rec(tp[c], tn[c], fp[c], fn[c]) for c in range(num_classes)]
            precs = [ClassPositivesNegatives.prec(tp[c], tn[c], fp[c], fn[c]) for c in range(num_classes)]
            def _f1(prec, rec):
                if (prec + rec) == 0:
                    return float("NaN")
                return 2*(prec*rec) / (prec+rec)
            f1s = [_f1(precs[c], recs[c]) for c in range(num_classes)]
            return sum(f1s) / num_classes
        prec = ClassPositivesNegatives.prec(tp, tn, fp, fn)
        rec = ClassPositivesNegatives.rec(tp, tn, fp, fn)
        if (prec + rec) == 0:
            return float("NaN")
        return 2 * (prec * rec) / (prec + rec)
    
    def bacc(tp, tn, fp, fn):
        if isinstance(tp, list):
            # multiclass
            # macro averaging
            num_classes = len(tp)
            recs = [ClassPositivesNegatives.rec(tp[c], tn[c], fp[c], fn[c]) for c in range(num_classes)]
            nrecs = [ClassPositivesNegatives.rec(tn[c], tp[c], fn[c], fp[c]) for c in range(num_classes)]
            baccs = [(recs[c] + nrecs[c]) / 2 for c in range(num_classes)]
            return sum(baccs) / num_classes
        rec = ClassPositivesNegatives.rec(tp, tn, fp, fn)
        nrec = ClassPositivesNegatives.rec(tn, tp, fn, fp)
        return (rec + nrec) / 2

    def mcc(tp, tn, fp, fn):
        if isinstance(tp, list):
            # multiclass
            num_classes = len(tp)
            # correct * samples - true_vec * positive_vec
            # -------------------------------------------
            # sqrt(samples*sample - positive*positive) * sqrt(samples*samples - true_vec*true_vec)
            correct = sum(tp)
            samples = (sum(tp) + sum(tn) + sum(fp) + sum(fn)) / num_classes
            true_pos = sum((tp[c] + fn[c])*(tp[c] + fp[c]) for c in range(num_classes))
            true_true = sum((tp[c] + fn[c])**2 for c in range(num_classes))
            pos_pos = sum((tp[c] + fp[c])**2 for c in range(num_classes))
            denominator = (math.sqrt(samples*samples - pos_pos) * math.sqrt(samples*samples - true_true))
            if denominator == 0:
                return -1
            return (correct * samples - true_pos) / denominator
        
        return (tp*tn - fp*fn) / \
            (math.sqrt((tp + fp)*(tp + fn)*(tn + fp)*(tn + fn)) + 1e-12)
    
    def prec(tp, tn, fp, fn):
        if isinstance(tp, list):
            # multiclass
            # macro averaging
            num_classes = len(tp)
            precs = [ClassPositivesNegatives.prec(tp[c], tn[c], fp[c], fn[c]) for c in range(num_classes)]
            return sum(precs) / num_classes
        if tp + fp == 0:
            return float("NaN")
        return (tp) / (tp + fp)
    
    def rec(tp, tn, fp, fn):
        if isinstance(tp, list):
            # multiclass
            # macro averaging
            num_classes = len(tp)
            recs = [ClassPositivesNegatives.rec(tp[c], tn[c], fp[c], fn[c]) for c in range(num_classes)]
            return sum(recs) / num_classes
        if tp + fn == 0:
            return float("NaN")
        return (tp) / (tp + fn)

    def compute(self, pred: Prediction, graph: Graph, aggregated=True) -> Dict[str, Loss]:
        if not aggregated:
            raise NotImplementedError("Not aggregated values are not implemented yet")
        y = graph.y
        if isinstance(y, list):
            y = torch.as_tensor(y, device=pred.device)
        if len(pred.shape) == 1:
            # binary classification
            pred = torch.sigmoid(pred)
            pred = pred > 0.5
            y = y > 0.5

            correct = (y == pred)
            incorrect = torch.logical_not(correct)
            
            tp = torch.sum(torch.logical_and(correct, pred)).cpu()
            tn = torch.sum(torch.logical_and(correct, torch.logical_not(pred))).cpu()
            fp = torch.sum(torch.logical_and(incorrect, pred)).cpu()
            fn = torch.sum(torch.logical_and(incorrect, torch.logical_not(pred))).cpu()

            self.tp += tp.item()
            self.tn += tn.item()
            self.fp += fp.item()
            self.fn += fn.item()

            return {
                "F1": ClassPositivesNegatives.f1(tp, tn, fp, fn),
                "Prec": ClassPositivesNegatives.prec(tp, tn, fp, fn),
                "Rec": ClassPositivesNegatives.rec(tp, tn, fp, fn),
                "BAcc": ClassPositivesNegatives.bacc(tp, tn, fp, fn),
                "MCC": ClassPositivesNegatives.mcc(tp, tn, fp, fn),
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn
            }
        else:
            # multiclass
            self._maybe_init_multi(pred.shape)
            num_classes = pred.shape[1]
            pred = F.softmax(pred, dim=-1)
            pred = torch.argmax(pred, dim=-1)
            for c in range(num_classes):
                positive = (pred == c)
                negative = torch.logical_not(positive)
                
                correct = (y == pred)
                incorrect = torch.logical_not(correct)

                self.tp[c] += torch.sum(torch.logical_and(positive, correct)).cpu().item()
                self.tn[c] += torch.sum(torch.logical_and(negative, y != c)).cpu().item()
                self.fp[c] += torch.sum(torch.logical_and(positive, incorrect)).cpu().item()
                self.fn[c] += torch.sum(torch.logical_and(negative, y == c)).cpu().item()
            return {
                "F1": ClassPositivesNegatives.f1(self.tp, self.tn, self.fp, self.fn),
                "Prec": ClassPositivesNegatives.prec(self.tp, self.tn, self.fp, self.fn),
                "Rec": ClassPositivesNegatives.rec(self.tp, self.tn, self.fp, self.fn),
                "BAcc": ClassPositivesNegatives.bacc(self.tp, self.tn, self.fp, self.fn),
                "MCC": ClassPositivesNegatives.mcc(self.tp, self.tn, self.fp, self.fn),
                "TP": sum(self.tp),
                "TN": sum(self.tn),
                "FP": sum(self.fp),
                "FN": sum(self.fn)
            }
        raise NotImplementedError("Multi class metrics not implemented yet")
    
    def _maybe_init_multi(self, shape):
        if isinstance(self.tp, list):
            return
        num_classes = shape[1]
        self.tp = [0 for _ in range(num_classes)]
        self.tn = [0 for _ in range(num_classes)]
        self.fp = [0 for _ in range(num_classes)]
        self.fn = [0 for _ in range(num_classes)]
    
    def keys(self) -> List[str]:
        return ["F1", "Prec", "Rec", "BAcc", "MCC", "TP", "TN", "FP", "FN"]


class BestClassPositivesNegatives(ClassificationMetric):
    def __init__(self):
        self.preds = []
        self.ys = []
        self.train = False
        self.train_ys = []
        self.train_preds = []
    
    def reset(self, train=False):
        self.preds = []
        self.ys = []
        self.train = train
        if train:
            self.train_ys = []
            self.train_preds = []
    
    def get_aggregated(self):
        
        return {
            "BF1": self.best_value(ClassPositivesNegatives.f1),
            "BPrec": self.best_value(ClassPositivesNegatives.prec),
            "BRec": self.best_value(ClassPositivesNegatives.rec),
            "BBAcc": self.best_value(ClassPositivesNegatives.bacc),
            "BMCC": self.best_value(ClassPositivesNegatives.mcc)
        }
    
    def best_value(self, metric):
        threshold = 0.5
        if len(self.train_ys) > 0:
            assert len(self.train_ys) == len(self.train_preds)
            thresholds = torch.linspace(0.01, 0.99, steps=50).tolist()
            threshold = max(
                (
                    (metric(*self.positives_negatives(threshold, self.train_ys, self.train_preds)), threshold) 
                    for threshold in thresholds
                ),
                key=lambda t: t[0]
                )[1]
        return metric(*self.positives_negatives(threshold, self.ys, self.preds))

    def positives_negatives(self, threshold, ys, preds):
        pred = torch.tensor(preds) > threshold
        y = torch.tensor(ys) > threshold

        correct = (y == pred)
        incorrect = torch.logical_not(correct)
        
        tp = torch.sum(torch.logical_and(correct, pred)).cpu().item()
        tn = torch.sum(torch.logical_and(correct, torch.logical_not(pred))).cpu().item()
        fp = torch.sum(torch.logical_and(incorrect, pred)).cpu().item()
        fn = torch.sum(torch.logical_and(incorrect, torch.logical_not(pred))).cpu().item()
        return tp, tn, fp, fn

    def compute(self, pred: Prediction, graph: Graph, aggregated=True) -> Dict[str, Loss]:
        if not aggregated:
            raise NotImplementedError("Not aggregated values are not implemented yet")
        y = graph.y
        if isinstance(y, list):
            y = torch.as_tensor(y, device=pred.device)
        self.ys.extend(y.detach().cpu().tolist())
        if self.train:
            self.train_ys.extend(y.detach().cpu().tolist())
        if len(pred.shape) == 1:
            pred = torch.sigmoid(pred)
            self.preds.extend(pred.detach().cpu().tolist())
            if self.train:
                self.train_preds.extend(pred.detach().cpu().tolist())

            pred = pred > 0.5
            y = y > 0.5

            correct = (y == pred)
            incorrect = torch.logical_not(correct)
            
            tp = torch.sum(torch.logical_and(correct, pred)).cpu()
            tn = torch.sum(torch.logical_and(correct, torch.logical_not(pred))).cpu()
            fp = torch.sum(torch.logical_and(incorrect, pred)).cpu()
            fn = torch.sum(torch.logical_and(incorrect, torch.logical_not(pred))).cpu()

            return {
                "BF1": ClassPositivesNegatives.f1(tp, tn, fp, fn),
                "BPrec": ClassPositivesNegatives.prec(tp, tn, fp, fn),
                "BRec": ClassPositivesNegatives.rec(tp, tn, fp, fn),
                "BBAcc": ClassPositivesNegatives.bacc(tp, tn, fp, fn),
                "BMCC": ClassPositivesNegatives.mcc(tp, tn, fp, fn)
            }
        raise NotImplementedError("Multi class metrics not implemented yet")
    
    def keys(self) -> List[str]:
        return ["BF1", "BPrec", "BRec", "BBAcc", "BMCC"]