import json

import torch
import numpy as np

from experiments.utils import Logger
from experiments.rho_trainer import Trainer
from experiments.autoencoding.classification_metrics import *
from experiments.datasets.reduced import ReducedDataset

from params import CLASSIFIER_PARAMS, CLASSIFICATION_TRAINER_PARAMS, CLASSIFICATION_DATASET
from utils import get_dataset, get_classification_model

RUNS = 10
PERCENTAGES = [1.0, 0.8, 0.6, 0.4, 0.2]

def set_seed(seed):
    CLASSIFICATION_TRAINER_PARAMS["seed"] = seed
    np.random.seed(CLASSIFICATION_TRAINER_PARAMS["seed"])
    torch.manual_seed(CLASSIFICATION_TRAINER_PARAMS["seed"])

def write_result(split, variant, classifier, dataset, trainer, result, params):
    with open("results.csv", "a") as f:
        f.write(f"{split};{variant};{classifier};{dataset};{trainer};{json.dumps(params)};{json.dumps(result)}\n")

CLASSIFICATION_DATASET["overwrite_cache"] = False
full_dataset = get_dataset(CLASSIFICATION_DATASET)
CLASSIFIER_PARAMS["features"] = full_dataset.get_input_size()
CLASSIFIER_PARAMS["edge_dim"] = full_dataset.get_edge_size()
CLASSIFIER_PARAMS["classes"] = len(full_dataset.get_classes())
if CLASSIFIER_PARAMS["classes"] == 2:
    CLASSIFIER_PARAMS["classes"] = 1

for percentage in PERCENTAGES:
    EXPERIMENT_NAME = f'{CLASSIFIER_PARAMS["name"]}_{CLASSIFICATION_DATASET["name"]+str(int(100*percentage))}_{CLASSIFICATION_TRAINER_PARAMS["name"]}'

    with Logger("results/" + EXPERIMENT_NAME + "/out.log"):
        for i in range(RUNS):
            print(f"Starting run {i} of {EXPERIMENT_NAME}")
            set_seed(CLASSIFICATION_TRAINER_PARAMS["seed"] + 1)
            dataset = ReducedDataset(full_dataset, percentage)
            model = get_classification_model(CLASSIFIER_PARAMS)

            losses = [CrossEntropyLoss()]
            metrics = [ClassAccuracy(), ClassPositivesNegatives(), ClassAUC(), ClassAP(), BestClassPositivesNegatives()]
            CLASSIFICATION_TRAINER_PARAMS["experiment_name"] = EXPERIMENT_NAME
            trainer = Trainer(model, losses, metrics, dataset, CLASSIFICATION_TRAINER_PARAMS)

            trainer.train()
            params = {**CLASSIFICATION_DATASET, **CLASSIFIER_PARAMS, \
                **CLASSIFICATION_TRAINER_PARAMS, **{"percentage": percentage}}
            write_result(
                "Train", "End", CLASSIFIER_PARAMS["name"], CLASSIFICATION_DATASET["name"]+str(int(100*percentage)),
                CLASSIFICATION_TRAINER_PARAMS["name"], trainer.evaluate(train_set=True), params
            )
            write_result(
                "Test", "End", CLASSIFIER_PARAMS["name"], CLASSIFICATION_DATASET["name"]+str(int(100*percentage)),
                CLASSIFICATION_TRAINER_PARAMS["name"], trainer.evaluate(train_set=False), params
            )
            trainer.load_checkpoint()
            write_result(
                "Train", "Checkpoint", CLASSIFIER_PARAMS["name"], CLASSIFICATION_DATASET["name"]+str(int(100*percentage)),
                CLASSIFICATION_TRAINER_PARAMS["name"], trainer.evaluate(train_set=True), params
            )
            write_result(
                "Test", "Checkpoint", CLASSIFIER_PARAMS["name"], CLASSIFICATION_DATASET["name"]+str(int(100*percentage)),
                CLASSIFICATION_TRAINER_PARAMS["name"], trainer.evaluate(train_set=False), params
            )

            trainer.cleanup()

            del trainer
            del model
            del dataset