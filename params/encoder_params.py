SMALL_GAT_HPARAMS = {
    "hidden_channels": 4,
    "num_layers": 1,
    "layer_type": "GAT",
    "autoencoder_type": "AE",
    "name": "SMAL_GAT"
}

SMALL_GCN_HPARAMS = {
    "num_layers": 2,
    "hidden_channels": 64,
    "layer_type": "GCN",
    "autoencoder_type": "AE",
    "name": "SMAL_GCN"
}

BIG_GAT = {
    "num_layers": 2,
    "hidden_channels": 32,
    "layer_type": "GAT",
    "autoencoder_type": "AE",
    "name": "BIG_GAT"
}

GAT_VAE = {
    "autoencoder_type": "VAE",
    "common": {
        "num_layers": 2,
        "hidden_channels": 32,
        "layer_type": "GAT",
        "norm_type": "GraphNorm",
    },
    "mu": {
        "num_layers": 1,
        "hidden_channels": 32,
        "layer_type": "GAT",
        "norm_type": "GraphNorm",
    },
    "logstd": {
        "num_layers": 1,
        "hidden_channels": 32,
        "layer_type": "GAT",
        "norm_type": "GraphNorm",
    },
    "name": "GAT_VAE"
}

GDN_VAE = {
    "autoencoder_type": "VAE",
    "common": {
        "num_layers": 2,
        "hidden_channels": 64,
        "layer_type": "GDN",
        "norm_type": "None", # TODO: was "None"
        "random_embedding_size": 16,
    },
    "mu": {
        "num_layers": 1,
        "hidden_channels": 64,
        "layer_type": "GDN",
        "norm_type": "None", # TODO: was "None"
    },
    "logstd": {
        "num_layers": 1,
        "hidden_channels": 64,
        "layer_type": "GDN",
        "norm_type": "None", # TODO: was "None"
    },
    "name": "GDN_VAE"
}

GDN_VAE_TRANSFORMER = {
    "autoencoder_type": "VAE",
    "common": {
        "num_layers": 2,
        "hidden_channels": 384,
        "layer_type": "GDN",
        "norm_type": "GraphNorm", # TODO: was "None"
        "random_embedding_size": 16,
    },
    "mu": {
        "num_layers": 1,
        "hidden_channels": 384,
        "layer_type": "GDN",
        "norm_type": "GraphNorm", # TODO: was "None"
    },
    "logstd": {
        "num_layers": 1,
        "hidden_channels": 384,
        "layer_type": "GDN",
        "norm_type": "GraphNorm", # TODO: was "None"
    },
    "name": "GDN_VAE"
}

BIG_GDN_VAE = {
    "autoencoder_type": "VAE",
    "common": {
        "num_layers": 2,
        "hidden_channels": 128,
        "layer_type": "GDN",
        "norm_type": "None",
        "random_embedding_size": 16,
    },
    "mu": {
        "num_layers": 1,
        "hidden_channels": 128,
        "layer_type": "GDN",
        "norm_type": "None",
    },
    "logstd": {
        "num_layers": 1,
        "hidden_channels": 128,
        "layer_type": "GDN",
        "norm_type": "None",
    },
    "name": "GDN_VAE"
}

GDN_VAE_REVEAL = {
    "autoencoder_type": "VAE",
    "common": {
        "num_layers": 2,
        "hidden_channels": 64,
        "layer_type": "GDN",
        "norm_type": "None",
        "random_embedding_size": 1,
    },
    "mu": {
        "num_layers": 1,
        "hidden_channels": 64,
        "layer_type": "GDN",
        "norm_type": "None",
    },
    "logstd": {
        "num_layers": 1,
        "hidden_channels": 64,
        "layer_type": "GDN",
        "norm_type": "None",
    },
    "name": "GDN_VAE"
}

ReGAE_ENCODER = {
    "type": "ReGAE",
    "hidden_channels": 64,
    "patch_dim": 8,
    "name": "ReGAE_ENCODER"
}