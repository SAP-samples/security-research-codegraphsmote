COMMON_PARAMETERS = {
    "batch_size": 8,
    "learning_rate": 1e-3,
    "milestones": [100, 200],
    "gamma": 1e-1,
    "epochs": 1000,
    "seed": 1234,
    "early_stopping_patience": 30,
    "name": "common"
}

REVEAL_PARAMETERS = {
    "batch_size": 1,
    "learning_rate": 1e-3,
    "milestones": [100, 200],
    "gamma": 1e-1,
    "epochs": 1000,
    "seed": 1234,
    "early_stopping_patience": 30,
    "name": "reveal"
}

COMMON_ALWAYS_SAVE = {
    "batch_size": 8,
    "learning_rate": 1e-3,
    "milestones": [100, 200],
    "gamma": 1e-1,
    "epochs": 1000,
    "seed": 1234,
    "early_stopping_patience": 30,
    "save_every_epoch": True,
    "name": "common"
}

REGULARIZED_PARAMS = {
    "batch_size": 64,
    "l2": 1e-1,
    "learning_rate": 1e-3,
    "milestones": [100, 200],
    "gamma": 1e-1,
    "epochs": 1000,
    "seed": 1234,
    "early_stopping_patience": 10,
    "name": "regularized"
}

BASELINE_TRAINER = {
    "batch_size": 128,
    "l2": 1e-4,
    "learning_rate": 5e-4,
    "milestones": [1000, 1001],
    "gamma": 1.0,
    "epochs": 500,
    "seed": 1234,
    "early_stopping_patience": 50,
    "name": "baseline"
}

DEVIGN_TRAINER = {
    "batch_size": 128,
    "l2": 1e-2,
    "learning_rate": 1e-4,
    "milestones": [1000, 1001],
    "gamma": 1.0,
    "epochs": 500,
    "seed": 1234,
    "early_stopping_patience": 10,
    "name": "devign"
}

RHO_LOSS_TRAINER = {
    "batch_size": 128,
    "l2": 1e-4,
    "learning_rate": 5e-4,
    "milestones": [1000, 1001],
    "gamma": 1.0,
    "epochs": 500,
    "seed": 1234,
    "train_irreducible": True,
    "early_stopping_patience": 50,
    "name": "rholoss"
}

PATIENT_TRAINER = {
    "batch_size": 128,
    "l2": 1e-4,
    "learning_rate": 5e-4,
    "milestones": [1000, 1001],
    "gamma": 1.0,
    "epochs": 10_000,
    "seed": 1234,
    "early_stopping_patience": 500,
    "name": "patient"
}