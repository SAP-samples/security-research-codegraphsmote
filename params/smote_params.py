from .dataset_params import *
from .encoder_params import *
from .decoder_params import *


REVEAL_AUTOENCODED = {
    "type": "Autoencoded",
    "dataset": REVEAL_DATASET_PARAMS,
    "non_categorical_features": REVEAL_DATASET_PARAMS["encoding_params"]["vector_size"],
    "model": {
        "encoder": GDN_VAE_TRANSFORMER,
        "decoder": COMPOSITE_GDN,
        "checkpoint": "results/GDN_VAE_COMPOSITE_GDN_VAE_CHROMIUM_reveal/checkpoint"
    },
    "cache_dir": "REVEAL_AUTOENCODED",
    "name": "REVEAL_AUTOENCODED"
}

REVEAL_AUTOENCODED_DOWNSAMPLED = {
    "type": "Autoencoded",
    "dataset": REVEAL_DOWNSAMPLED,
    "non_categorical_features": REVEAL_DATASET_PARAMS["encoding_params"]["vector_size"],
    "model": {
        "encoder": GDN_VAE_REVEAL,
        "decoder": COMPOSITE_GDN,
        "checkpoint": "results/GDN_VAE_COMPOSITE_GDN_VAE_CHROMIUM_common/checkpoint"
    },
    "cache_dir": "REVEAL_AUTOENCODED",
    "name": "REVEAL_AUTOENCODED_DOWNSAMPLED"
}

REVEAL_SMOTE = {
    "type": "SMOTE",
    "dataset": REVEAL_DATASET_PARAMS,
    "non_categorical_features": REVEAL_DATASET_PARAMS["encoding_params"]["vector_size"],
    "repetitions": 10,
    "model": {
        "encoder": GDN_VAE_TRANSFORMER,
        "decoder": COMPOSITE_GDN,
        "checkpoint": "results/GDN_VAE_COMPOSITE_GDN_VAE_CHROMIUM_reveal/checkpoint"
    },
    "cache_dir": "REVEAL_AUTOENCODED",
    "name": "REVEAL_SMOTE"
}

REVEAL3_AUTOENCODED = {
    "type": "Autoencoded",
    "dataset": REVEAL3_PARAMS,
    "non_categorical_features": REVEAL3_PARAMS["encoding_params"]["vector_size"],
    "model": {
        "encoder": GDN_VAE_TRANSFORMER,
        "decoder": COMPOSITE_GDN,
        "checkpoint": "results/GDN_VAE_COMPOSITE_GDN_VAE_REVEAL3_TRANSFORMER_reveal/checkpoint"
    },
    "cache_dir": "REVEAL3_AUTOENCODED",
    "name": "REVEAL3_AUTOENCODED"
}

REVEAL3_AUTOENCODED_DOWNSAMPLED = {
    "type": "Downsampled",
    "dataset": REVEAL3_AUTOENCODED,
    "name": "REVEAL3_AUTOENCODED_DOWNSAMPLED"
}

QEMU_FFMPEG_AUTOENCODED = {
    "type": "Autoencoded",
    "dataset": QEMU_FFMPEG_PARAMS,
    "non_categorical_features": QEMU_FFMPEG_PARAMS["datasets"][0]["encoding_params"]["vector_size"],
    "model": {
        "encoder": GDN_VAE_TRANSFORMER,
        "decoder": COMPOSITE_GDN,
        "checkpoint": "results/GDN_VAE_COMPOSITE_GDN_VAE_REVEAL_DEVIGN_reveal/checkpoint"
    },
    "cache_dir": "QEMU_FFMPEG_AUTOENCODED",
    "name": "QEMU_FFMPEG_AUTOENCODED"
}

REVEAL3_SMOTE = {
    "type": "SMOTE",
    "dataset": REVEAL3_TRANSFORMER,
    "non_categorical_features": REVEAL3_TRANSFORMER["encoding_params"]["vector_size"],
    "model": {
        "encoder": GDN_VAE_TRANSFORMER,
        "decoder": COMPOSITE_GDN,
        "checkpoint": "results/GDN_VAE_COMPOSITE_GDN_VAE_REVEAL3_TRANSFORMER_reveal/checkpoint"
    },
    "cache_dir": "REVEAL3_AUTOENCODED",
    "repetitions": 5,
    "name": "REVEAL3_SMOTE"
}

PATCHDB_SMOTE = {
    "type": "SMOTE",
    "dataset": PATCHDB_TRANSFORMER,
    "non_categorical_features": PATCHDB_TRANSFORMER["encoding_params"]["vector_size"],
    "repetitions": 10,
    "model": {
        "encoder": GDN_VAE_TRANSFORMER,
        "decoder": COMPOSITE_GDN,
        "checkpoint": "results/GDN_VAE_COMPOSITE_GDN_VAE_PATCHDB_TRANSFORMER_reveal/checkpoint"
    },
    "cache_dir": "PATCHDB_AUTOENCODED",
    "name": "PATCHDB_SMOTE"
}

PATCHDB_AUTOENCODED = {
    "type": "Autoencoded",
    "dataset": PATCHDB_TRANSFORMER,
    "non_categorical_features": PATCHDB_TRANSFORMER["encoding_params"]["vector_size"],
    "repetitions": 10,
    "model": {
        "encoder": GDN_VAE_TRANSFORMER,
        "decoder": COMPOSITE_GDN,
        "checkpoint": "results/GDN_VAE_COMPOSITE_GDN_VAE_PATCHDB_TRANSFORMER_reveal/checkpoint"
    },
    "cache_dir": "PATCHDB_AUTOENCODED",
    "name": "PATCHDB_AUTOENCODED"
}

FFMPEG_AUTOENCODED_DOWNSAMPLED = {
    "type": "Autoencoded",
    "dataset": FFMPEG_DOWNSAMPLED,
    "non_categorical_features": FFMPEG_PARAMS["encoding_params"]["vector_size"],
    "model": {
        "encoder": GDN_VAE,
        "decoder": COMPOSITE_GDN,
        "checkpoint": "results/GDN_VAE_COMPOSITE_GDN_VAE_FFMPEG_reveal/checkpoint"
    },
    "cache_dir": "FFMPEG_AUTOENCODED",
    "name": "FFMPEG_AUTOENCODED_DOWNSAMPLED"
}

FFMPEG_SMOTE = {
    "type": "SMOTE",
    "dataset": FFMPEG_PARAMS,
    "non_categorical_features": FFMPEG_PARAMS["encoding_params"]["vector_size"],
    "model": {
        "encoder": GDN_VAE,
        "decoder": COMPOSITE_GDN,
        "checkpoint": "results/GDN_VAE_COMPOSITE_GDN_VAE_FFMPEG_reveal/checkpoint"
    },
    "cache_dir": "FFMPEG_AUTOENCODED",
    "name": "FFMPEG_SMOTE"
}

QEMU_FFMPEG_SMOTE = {
    "type": "SMOTE",
    "dataset": QEMU_FFMPEG_PARAMS,
    "non_categorical_features": FFMPEG_PARAMS["encoding_params"]["vector_size"],
    "model": {
        "encoder": GDN_VAE,
        "decoder": COMPOSITE_GDN,
        "checkpoint": "results/GDN_VAE_COMPOSITE_GDN_VAE_QEMU_FFMPEG_reveal/checkpoint"
    },
    "cache_dir": "QEMU_FFMPEG_AUTOENCODED",
    "name": "QEMU_FFMPEG_SMOTE"
}