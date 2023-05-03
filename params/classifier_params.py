GAT_CLASSIFIER = {
    "type": "GraphClassifier",
    "name": "GAT_CLASSIFIER",
    "encoder": {
        "type": "GraphComposite",
        "pooling": {
            "type": "max"
        },
        "encoder": {
            "num_layers": 1,
            "hidden_channels": 8,
            "dropout": 0.2,
            "layer_type": "GAT",
            "norm_type": "BatchNorm"
        }
    },
    "classifier": {
        "layer_type": "MLP",
        "num_layers": 1
    }
}

BASELINE_GCN_CLASSIFIER = {
    "type": "GraphClassifier",
    "name": "BASELINE_GCN",
    "encoder": {
        "type": "GraphComposite",
        "pooling": {
            "type": "mean"
        },
        "encoder": {
            "num_layers": 3,
            "hidden_channels": 128,
            "layer_type": "GCN",
            "norm_type": "None"
        }
    },
    "classifier": {
        "layer_type": "MLP",
        "dropout": 0.5,
        "num_layers": 3
    }
}

BASELINE_GIN_CLASSIFIER = {
    "type": "GraphClassifier",
    "name": "BASELINE_GIN",
    "encoder": {
        "type": "GraphComposite",
        "pooling": {
            "type": "sum"
        },
        "encoder": {
            "num_layers": 3,
            "hidden_channels": 128,
            "layer_type": "GIN",
            "norm_type": "None"
        }
    },
    "classifier": {
        "layer_type": "MLP",
        "dropout": 0.5,
        "num_layers": 3
    }
}

BASELINE_GIN_EPS_CLASSIFIER = {
    "type": "GraphClassifier",
    "name": "BASELINE_GIN_EPS",
    "encoder": {
        "type": "GraphComposite",
        "pooling": {
            "type": "sum"
        },
        "encoder": {
            "num_layers": 3,
            "hidden_channels": 128,
            "layer_type": "GIN",
            "norm_type": "None",
            "train_eps": True
        }
    },
    "classifier": {
        "layer_type": "MLP",
        "dropout": 0.5,
        "num_layers": 3
    }
}

GGNN_CLASSIFIER = {
    "type": "GraphClassifier",
    "name": "BASELINE_GGNN",
    "encoder": {
        "type": "GraphComposite",
        "pooling": {
            "type": "max"
        },
        "encoder": {
            "num_layers": 6,
            "hidden_channels": 200,
            "layer_type": "GGNN"
        }
    },
    "classifier": {
        "layer_type": "MLP",
        "dropout": 0.2,
        "num_layers": 3
    }
}

REVEAL_CLASSIFIER = {
    "type": "GraphClassifier",
    "name": "REVEAL_CLASSIFIER",
    "encoder": {
        "type": "GraphComposite",
        "pooling": {
            "type": "sum"
        },
        "encoder": {
            "num_layers": 8,
            "hidden_channels": 200,
            "layer_type": "GGNN"
        }
    },
    "classifier": {
        "layer_type": "MLP",
        "dropout": 0.2,
        "num_layers": 5,
        "hidden_sizes": [256, 128, 256]
    }
}

DEVIGN_CLASSIFIER = {
    "type": "GraphClassifier",
    "name": "DEVIGN_CLASSIFIER",
    "variant": "devign"
}

GDN_CLASSIFIER = {
    "type": "GraphClassifier",
    "name": "GDN",
    "encoder": {
        "type": "GraphComposite",
        "pooling": {
            "type": "mean"
        },
        "encoder": {
            "num_layers": 3,
            "hidden_channels": 128,
            "layer_type": "GDN",
            "norm_type": "None"
        }
    },
    "classifier": {
        "layer_type": "MLP",
        "dropout": 0.5,
        "num_layers": 3
    }
}

ReGAE_CLASSIFIER = {
    "type": "GraphClassifier",
    "name": "ReGAE_CLASSIFIER",
    "encoder": {
        "type": "ReGAE",
        "hidden_channels": 64,
        "patch_dim": 8,
        "name": "ReGAE_ENCODER"
    },
    "classifier": {
        "layer_type": "MLP",
        "dropout": 0.5,
        "num_layers": 3
    }
}