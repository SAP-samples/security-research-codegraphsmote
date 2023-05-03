from .dataset_params import *
from .decoder_params import *
from .encoder_params import *
from .trainer_params import *
from .smote_params import *
from .classifier_params import *


AUTOENCODER_DATASET = REVEAL3_TRANSFORMER
DECODER_PARAMS = COMPOSITE_GDN_VAE
ENCODER_PARAMS = GDN_VAE_TRANSFORMER
AUTOENCODER_TRAINER_PARAMS = REVEAL_PARAMETERS

CLASSIFICATION_DATASET = REVEAL3_SMOTE
CLASSIFICATION_TRAINER_PARAMS = BASELINE_TRAINER
CLASSIFIER_PARAMS = REVEAL_CLASSIFIER