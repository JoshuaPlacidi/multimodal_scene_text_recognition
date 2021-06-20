import config


import sys
# if len(sys.argv) != 2:
#     raise Exception('Invalid arguments: correct ussage "python run.py [dataset] [mode]" \nExample: python run.py cocotext train')
# elif

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import time

import torch
torch.manual_seed(config.RANDOM_SEED)

from model import get_model
from training_functions import train, validate, evaluate, run_validation

print('--- Experiment: ' + config.EXPERIMENT)
print('  - Encoder:', config.ENCODER)
print('  - Decoder:', config.DECODER)
print('  - Devices:', config.DEVICE_IDS)

# Evaluation
# model = get_model('./results/models/sig_gated_pre_encoder.pt')
# evaluate(model=model)


# Train
model = get_model(config.SAVED_MODEL)

train(model=model, dataset='textocr', validation_steps=2000, iteration_limit=None)

# score, results_df = run_validation(model=model, dataset='cocotext')
# print(score)
# print(results_df.head())
# results_df.to_csv('./results/val_df.csv')