import json

import torch
import numpy as np

from torch_geometric.loader import DataLoader

from experiments.utils import Logger
from experiments.rho_trainer import Trainer
from experiments.autoencoding.classification_metrics import *

from params import CLASSIFICATION_DATASET, CLASSIFIER_PARAMS, CLASSIFICATION_TRAINER_PARAMS
from utils import get_dataset, get_classification_model

RUNS = 10

from params import *

CROSSEVALUATION_DATASETS = [QEMU_FFMPEG_PARAMS]
# CROSSEVALUATION_DATASETS = []

def set_seed(seed):
    CLASSIFICATION_TRAINER_PARAMS["seed"] = seed
    np.random.seed(CLASSIFICATION_TRAINER_PARAMS["seed"])
    torch.manual_seed(CLASSIFICATION_TRAINER_PARAMS["seed"])

EXPERIMENT_NAME = f'{CLASSIFIER_PARAMS["name"]}_{CLASSIFICATION_DATASET["name"]}_{CLASSIFICATION_TRAINER_PARAMS["name"]}'

def write_result(split, variant, classifier, dataset, trainer, result, params):
    with open("results/results.csv", "a") as f:
        f.write(f"{split};{variant};{classifier};{dataset};{trainer};{json.dumps(params)};{json.dumps(result)}\n")

def cross_eval(trainer, common_params, datasets, variant):
    prev_loader = trainer.test_loader
    for dataset_params in datasets:
        dataset_params["overwrite_cache"] = False
        dataset = get_dataset(dataset_params)
        trainer.test_loader = DataLoader(dataset, batch_size=trainer.params["batch_size"],
                                      shuffle=False, num_workers=4, prefetch_factor=1)
        write_result(split=dataset_params["name"], variant=variant, 
            result=trainer.evaluate(train_set=False), **common_params)
    trainer.test_loader = prev_loader

with Logger("results/" + EXPERIMENT_NAME + "/out.log"):
    for i in range(RUNS):
        print(f"Starting run {i} of {EXPERIMENT_NAME}")
        set_seed(CLASSIFICATION_TRAINER_PARAMS["seed"] + 1)
        CLASSIFICATION_DATASET["overwrite_cache"] = False
        dataset = get_dataset(CLASSIFICATION_DATASET)
        CLASSIFIER_PARAMS["features"] = dataset.get_input_size()
        CLASSIFIER_PARAMS["edge_dim"] = dataset.get_edge_size()
        CLASSIFIER_PARAMS["classes"] = len(dataset.get_classes())
        if CLASSIFIER_PARAMS["classes"] == 2:
            CLASSIFIER_PARAMS["classes"] = 1
        model = get_classification_model(CLASSIFIER_PARAMS)

        losses = [CrossEntropyLoss()]
        metrics = [ClassAccuracy(), ClassPositivesNegatives(), ClassAUC(), ClassAP(), BestClassPositivesNegatives()]
        CLASSIFICATION_TRAINER_PARAMS["experiment_name"] = EXPERIMENT_NAME
        trainer = Trainer(model, losses, metrics, dataset, CLASSIFICATION_TRAINER_PARAMS)

        trainer.train()
        params = {**CLASSIFICATION_DATASET, **CLASSIFIER_PARAMS, **CLASSIFICATION_TRAINER_PARAMS}
        common_params = {
            "classifier": CLASSIFIER_PARAMS["name"],
            "dataset": CLASSIFICATION_DATASET["name"],
            "trainer": CLASSIFICATION_TRAINER_PARAMS["name"],
            "params": params
        }
        write_result(
            "Train", "End", CLASSIFIER_PARAMS["name"], CLASSIFICATION_DATASET["name"],
            CLASSIFICATION_TRAINER_PARAMS["name"], trainer.evaluate(train_set=True), params
        )
        write_result(
            "Test", "End", CLASSIFIER_PARAMS["name"], CLASSIFICATION_DATASET["name"],
            CLASSIFICATION_TRAINER_PARAMS["name"], trainer.evaluate(train_set=False), params
        )
        cross_eval(trainer, common_params, CROSSEVALUATION_DATASETS, "End")
        trainer.load_checkpoint()
        write_result(
            "Train", "Checkpoint", CLASSIFIER_PARAMS["name"], CLASSIFICATION_DATASET["name"],
            CLASSIFICATION_TRAINER_PARAMS["name"], trainer.evaluate(train_set=True), params
        )
        write_result(
            "Test", "Checkpoint", CLASSIFIER_PARAMS["name"], CLASSIFICATION_DATASET["name"],
            CLASSIFICATION_TRAINER_PARAMS["name"], trainer.evaluate(train_set=False), params
        )
        cross_eval(trainer, common_params, CROSSEVALUATION_DATASETS, "Checkpoint")

        trainer.cleanup()

        del trainer
        del model
        del dataset