import json

import torch
import numpy as np

from torch_geometric.loader import DataLoader

from experiments.utils import Logger
from experiments.rho_trainer import Trainer
from experiments.autoencoding.classification_metrics import *
from experiments.datasets.reduced import ReducedDataset
from experiments.datasets.fillup import NodeDropDataset, EdgeDropDataset
from experiments.datasets.vulnerability_dataset import get_cwe_subsets

from params import CLASSIFIER_PARAMS, CLASSIFICATION_TRAINER_PARAMS, CLASSIFICATION_DATASET
from utils import get_dataset, get_classification_model


RUNS = 10
PERCENTAGES = [0.2, 0.4, 0.6, 0.8, 1.0]

MODIFIER = "CONSTANT" # one of "", "SCALING", "CONSTANT", "CONSTANT_DOUBLED"


assert CLASSIFICATION_DATASET["type"] in ["NodeDrop", "EdgeDrop"], "This script is only valid for Drop-Datasets"

DatasetClass = NodeDropDataset if CLASSIFICATION_DATASET["type"] == "NodeDrop" else EdgeDropDataset

def set_seed(seed):
    CLASSIFICATION_TRAINER_PARAMS["seed"] = seed
    np.random.seed(CLASSIFICATION_TRAINER_PARAMS["seed"])
    torch.manual_seed(CLASSIFICATION_TRAINER_PARAMS["seed"])


def write_result(split, variant, classifier, dataset, trainer, result, params):
    with open("results.csv", "a") as f:
        f.write(f"{split};{variant};{classifier};{dataset};{trainer};{json.dumps(params)};{json.dumps(result)}\n")


CLASSIFICATION_DATASET["overwrite_cache"] = False
CLASSIFICATION_DATASET["dataset"]["overwrite_cache"] = CLASSIFICATION_DATASET["overwrite_cache"]
full_dataset = get_dataset(CLASSIFICATION_DATASET["dataset"])
CLASSIFIER_PARAMS["features"] = full_dataset.get_input_size()
CLASSIFIER_PARAMS["edge_dim"] = full_dataset.get_edge_size()
CLASSIFIER_PARAMS["classes"] = len(full_dataset.get_classes())
if CLASSIFIER_PARAMS["classes"] == 2:
    CLASSIFIER_PARAMS["classes"] = 1

desired_samples = None
if MODIFIER == "SCALING":
    train_factor = 0.8
    majority_samples = max(len(full_dataset.get_subset_of(clas)) for clas in full_dataset.get_classes()) * train_factor
    minority_samples = min(len(full_dataset.get_subset_of(clas)) for clas in full_dataset.get_classes()) * train_factor
    def desired_samples_fun(percentage):
        majority_factor = 1.1 * (1 / (percentage**0.2))
        minority_factor = 1.2 * (1 / (percentage**0.2))
        return min(int(majority_factor * majority_samples * percentage), int(minority_factor * minority_samples * percentage))
    desired_samples = desired_samples_fun
    CLASSIFICATION_DATASET["desired_samples"] = desired_samples
elif MODIFIER == "CONSTANT":
    train_factor = 0.8
    majority_samples = max(len(full_dataset.get_subset_of(clas)) for clas in full_dataset.get_classes()) * train_factor
    minority_samples = min(len(full_dataset.get_subset_of(clas)) for clas in full_dataset.get_classes()) * train_factor
    def desired_samples_fun(_):
        majority_factor = 1.1
        minority_factor = 1.2
        return min(int(majority_factor * majority_samples), int(minority_factor * minority_samples))
    desired_samples = desired_samples_fun
    CLASSIFICATION_DATASET["desired_samples"] = desired_samples
elif MODIFIER == "CONSTANT_DOUBLED":
    train_factor = 0.8
    majority_samples = max(len(full_dataset.get_subset_of(clas)) for clas in full_dataset.get_classes()) * train_factor
    minority_samples = min(len(full_dataset.get_subset_of(clas)) for clas in full_dataset.get_classes()) * train_factor
    def desired_samples_fun(_):
        majority_factor = 2.0
        minority_factor = 2.0
        return min(int(majority_factor * majority_samples), int(minority_factor * minority_samples))
    desired_samples = desired_samples_fun
    CLASSIFICATION_DATASET["desired_samples"] = desired_samples
elif MODIFIER == "":
    pass
else:
    raise NotImplementedError(f"Modifier {MODIFIER} unknown")

def test_patchdb(trainer, common_params, variant):
    if not "PATCHDB" in CLASSIFICATION_DATASET["name"].upper():
        return
    prev_loader = trainer.test_loader
    for cwe, dataset in get_cwe_subsets(trainer.dataset, trainer.test_set).items():
        trainer.test_loader = DataLoader(dataset, batch_size=trainer.params["batch_size"],
                                      shuffle=False, num_workers=4, prefetch_factor=1)
        write_result(split=cwe, variant=variant, 
            result=trainer.evaluate(train_set=False), **common_params)
    trainer.test_loader = prev_loader

for percentage in PERCENTAGES:
    EXPERIMENT_NAME = f'{CLASSIFIER_PARAMS["name"]}_{CLASSIFICATION_DATASET["name"]+MODIFIER+str(int(100*percentage))}_{CLASSIFICATION_TRAINER_PARAMS["name"]}'

    with Logger("results/" + EXPERIMENT_NAME + "/out.log"):
        for i in range(RUNS):
            print(f"Starting run {i} of {EXPERIMENT_NAME}")
            set_seed(CLASSIFICATION_TRAINER_PARAMS["seed"] + 1)
            reduced_dataset = ReducedDataset(full_dataset, percentage)
            if MODIFIER != "":
                CLASSIFICATION_DATASET["desired_samples"] = desired_samples(percentage)
            dataset = DatasetClass(reduced_dataset, CLASSIFICATION_DATASET)
            model = get_classification_model(CLASSIFIER_PARAMS)

            losses = [CrossEntropyLoss()]
            metrics = [ClassAccuracy(), ClassPositivesNegatives(), ClassAUC(), ClassAP(), BestClassPositivesNegatives()]
            CLASSIFICATION_TRAINER_PARAMS["experiment_name"] = EXPERIMENT_NAME
            trainer = Trainer(model, losses, metrics, dataset, CLASSIFICATION_TRAINER_PARAMS)

            trainer.train()

            original_loader = DataLoader(dataset.get_original_trainset(), batch_size=trainer.params["batch_size"],
                                      shuffle=False, num_workers=4, prefetch_factor=1)
            train_loader = trainer.train_loader
            params = {**CLASSIFICATION_DATASET, **CLASSIFIER_PARAMS, \
                **CLASSIFICATION_TRAINER_PARAMS, **{"percentage": percentage}}
            common_params = {
                "classifier": CLASSIFIER_PARAMS["name"],
                "dataset": CLASSIFICATION_DATASET["name"],
                "trainer": CLASSIFICATION_TRAINER_PARAMS["name"],
                "params": params,
            }
            write_result(
                "Train", "End", CLASSIFIER_PARAMS["name"], CLASSIFICATION_DATASET["name"]+MODIFIER+str(int(100*percentage)),
                CLASSIFICATION_TRAINER_PARAMS["name"], trainer.evaluate(train_set=True), params
            )
            trainer.train_loader = original_loader
            write_result(
                "TrainOriginal", "End", CLASSIFIER_PARAMS["name"], CLASSIFICATION_DATASET["name"]+MODIFIER+str(int(100*percentage)),
                CLASSIFICATION_TRAINER_PARAMS["name"], trainer.evaluate(train_set=True), params
            )
            write_result(
                "Test", "End", CLASSIFIER_PARAMS["name"], CLASSIFICATION_DATASET["name"]+MODIFIER+str(int(100*percentage)),
                CLASSIFICATION_TRAINER_PARAMS["name"], trainer.evaluate(train_set=False), params
            )
            test_patchdb(trainer, common_params, "End")
            trainer.train_loader = train_loader
            trainer.load_checkpoint()
            write_result(
                "Train", "Checkpoint", CLASSIFIER_PARAMS["name"], CLASSIFICATION_DATASET["name"]+MODIFIER+str(int(100*percentage)),
                CLASSIFICATION_TRAINER_PARAMS["name"], trainer.evaluate(train_set=True), params
            )
            trainer.train_loader = original_loader
            write_result(
                "TrainOriginal", "Checkpoint", CLASSIFIER_PARAMS["name"], CLASSIFICATION_DATASET["name"]+MODIFIER+str(int(100*percentage)),
                CLASSIFICATION_TRAINER_PARAMS["name"], trainer.evaluate(train_set=True), params
            )
            write_result(
                "Test", "Checkpoint", CLASSIFIER_PARAMS["name"], CLASSIFICATION_DATASET["name"]+MODIFIER+str(int(100*percentage)),
                CLASSIFICATION_TRAINER_PARAMS["name"], trainer.evaluate(train_set=False), params
            )
            test_patchdb(trainer, common_params, "Checkpoint")

            trainer.cleanup()

            del trainer
            del model
            del dataset