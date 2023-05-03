import torch
import numpy as np

from experiments.utils import Logger
from experiments.rho_trainer import Trainer
from experiments.autoencoding.metrics import *

from params import AUTOENCODER_DATASET, ENCODER_PARAMS, DECODER_PARAMS, AUTOENCODER_TRAINER_PARAMS
from utils import get_dataset, get_ae_model

np.random.seed(AUTOENCODER_TRAINER_PARAMS["seed"])
torch.manual_seed(AUTOENCODER_TRAINER_PARAMS["seed"])

EXPERIMENT_NAME = f'{ENCODER_PARAMS["name"]}_{DECODER_PARAMS["name"]}_{AUTOENCODER_DATASET["name"]}_{AUTOENCODER_TRAINER_PARAMS["name"]}'

with Logger("results/" + EXPERIMENT_NAME + "/out.log"):
    print("starting training run " + EXPERIMENT_NAME)

    AUTOENCODER_DATASET["overwrite_cache"] = False
    dataset = get_dataset(AUTOENCODER_DATASET)
    ENCODER_PARAMS["features"] = dataset.get_input_size()
    ENCODER_PARAMS["edge_dim"] = dataset.get_edge_size()
    print(f"Number of features {ENCODER_PARAMS['features']}")
    model = get_ae_model(ENCODER_PARAMS, DECODER_PARAMS)

    word_embedding_size = None
    if "type" in AUTOENCODER_DATASET and AUTOENCODER_DATASET["type"] == "Combined":
        word_embedding_size = AUTOENCODER_DATASET["datasets"][0]["encoding_params"]["vector_size"]
    if "encoding_params" in AUTOENCODER_DATASET:
        word_embedding_size = AUTOENCODER_DATASET["encoding_params"]["vector_size"]
    if word_embedding_size is None:
        losses = [DirectedAdjacencyCrossEntropyLoss(), FeaturesCE(), AdjacencySumDifference()]
    else:
        losses = [DirectedAdjacencyCrossEntropyLoss(), FeaturesCE(end_index=-word_embedding_size), AdjacencySumDifference(), FeaturesCos(start_index=-word_embedding_size)]
    metrics = [SampledDirectedEdgesAP(), SampledDirectedEdgesAUC()]
    AUTOENCODER_TRAINER_PARAMS["experiment_name"] = EXPERIMENT_NAME
    trainer = Trainer(model, losses, metrics, dataset, AUTOENCODER_TRAINER_PARAMS)

    trainer.train()

    print("Performance at end:")
    print("Train: " + str(trainer.evaluate(train_set=True)))
    print("Test: " + str(trainer.evaluate()))
    trainer.load_checkpoint()
    print("Performance at checkpoint:")
    print("Train: " + str(trainer.evaluate(train_set=True)))
    print("Test: " + str(trainer.evaluate()))

    trainer.cleanup()