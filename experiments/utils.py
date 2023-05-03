import os
import sys
from pathlib import Path

import torch

from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Logger(object):
    def __init__(self, path):
        self.path = Path(path)
        if not os.path.exists(self.path.parent):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = None
        self.stdout = sys.stdout
    
    def __enter__(self):
        self.file = open(self.path, "a")
        self.stdout = sys.stdout
        sys.stdout = self
        return self
    
    def __exit__(self, *args):
        sys.stdout = self.stdout
        self.file.close()
        self.file = None

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)

    def flush(self):
        self.stdout.flush()


class CorrectedSummaryWriter(SummaryWriter):
    def add_hparams(self, hparam_dict, metric_dict, hparam_domain_discrete=None):
        torch._C._log_api_usage_once("tensorboard.logging.add_hparams")
        if type(hparam_dict) is not dict or type(metric_dict) is not dict:
            raise TypeError('hparam_dict and metric_dict should be dictionary.')
        exp, ssi, sei = hparams(hparam_dict, metric_dict, hparam_domain_discrete)

        self.file_writer.add_summary(exp)
        self.file_writer.add_summary(ssi)
        self.file_writer.add_summary(sei)
        for k, v in metric_dict.items():
            if v is not None:
                self.add_scalar(k, v)


def flatten_hparams(raw_hparams):
    hparams = dict()

    def _sanitize_value(value):
        if type(value) in [int, float, str, bool]:
            return value
        if type(value).__module__ == "numpy":
            return value.item()
        print(f"Warning: skipping value {value} with type {type(value)}")
        return None

    for key, value in raw_hparams.items():
        if type(value) is dict:
            for inner_key, inner_value in flatten_hparams(value).items():
                hparams[key+"_"+inner_key] = inner_value
        elif type(value) in [list, tuple]:
            for index, inner_value in enumerate(value):
                inner_value = _sanitize_value(inner_value)
                if inner_value is None:
                    continue
                hparams[f"{key}{index}"] = inner_value
        else:
            value = _sanitize_value(value)
            if value is None:
                continue
            hparams[key] = value
    return hparams