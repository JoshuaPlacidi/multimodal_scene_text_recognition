import config

import os
import time
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
torch.manual_seed(config.RANDOM_SEED)

from model import get_model
from training_functions import train, validate, evaluate

print('--- Experiment: ' + config.EXPERIMENT)
print('  - Encoder:', config.ENCODER)
print('  - Decoder:', config.DECODER)
print('  - Devices:', config.DEVICE_IDS)

model = get_model(config.SAVED_MODEL)

train(model=model, dataset='synth', validation_steps=2000, iteration_limit=300000)