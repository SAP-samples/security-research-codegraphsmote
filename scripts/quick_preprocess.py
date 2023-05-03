import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiprocessing import Pool

import numpy as np
import torch
from tqdm import tqdm

from params import AUTOENCODER_DATASET, AUTOENCODER_TRAINER_PARAMS
from params import CLASSIFICATION_TRAINER_PARAMS, CLASSIFICATION_DATASET
from utils import get_dataset

USE_AUTOENCODER = False
MULTIPROCESSING = False

dataset = None
if USE_AUTOENCODER:
    np.random.seed(AUTOENCODER_TRAINER_PARAMS["seed"])
    torch.manual_seed(AUTOENCODER_TRAINER_PARAMS["seed"])

    print(f"starting {AUTOENCODER_DATASET['name']}")
    AUTOENCODER_DATASET["overwrite_cache"] = False
    dataset = get_dataset(AUTOENCODER_DATASET)
else:
    np.random.seed(CLASSIFICATION_TRAINER_PARAMS["seed"])
    torch.manual_seed(CLASSIFICATION_TRAINER_PARAMS["seed"])

    print(f"starting {CLASSIFICATION_DATASET['name']}")
    CLASSIFICATION_DATASET["overwrite_cache"] = False
    dataset = get_dataset(CLASSIFICATION_DATASET)

def execute(index):
    # _ = dataset[index]
    g = dataset[index]
    assert g.x.shape[1] == dataset.get_input_size(), f"{g.x.shape[1]} != {dataset.get_input_size()}"
    return None

if MULTIPROCESSING:
    with Pool() as p:
        for _ in tqdm(p.imap(execute, range(len(dataset))), total=len(dataset)):
            pass
else:
    for i in tqdm(range(len(dataset))):
        execute(i)
