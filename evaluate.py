import config

print('--- Running Evaluation')
print('  - Model:')
print('  - Encoder:', config.ENCODER)
print('  - Decoder:', config.DECODER)
print('  - Devices:', config.DEVICE_IDS)

from model import get_model

import torch
from torch.utils.data import Dataset, ConcatDataset, Subset
import torchvision.transforms as transforms

import string
from tqdm import tqdm
import pandas as pd

from utils import AttnLabelConverter
character = string.printable[:-6]
converter = AttnLabelConverter(character)

import string

from coco_dataset import LmdbDataset, get_cocotext_single_image_data

from PIL import Image
from io import StringIO
import numpy as np
from matplotlib.pyplot import imshow
import numpy as np
import time


saved_model = './results/models/pre_encoder_mlp_overlap_vinvl_resize_linear.pt'

model = get_model(saved_model)
model.eval()

import json
with open('./annotations/features/' + config.SEMANTIC_SOURCE.lower() + '_classes.txt') as f:
    obj_class_labels = f.read().splitlines()

# with open('./results/sem_errors.txt') as f:
#     error_ids = f.read().splitlines()

#val_data = get_val_data()

# def get_val_score(model, print_samples=False):
#     print('  - Running Validation')
#     model.eval()

#     total = 0
#     correct = 0

#     device = config.PRIMARY_DEVICE
#     batch_max_length = config.MAX_TEXT_LENGTH

#     error_annos = []

#     with open('./results/sem_errors.txt') as f:
#         base_errors_ids = f.read().splitlines()

#     with torch.no_grad():
#         for anno_batch, img_batch, text_batch, overlap_batch, scene_batch in tqdm(val_data):
#             image_tensor = img_batch
#             text = text_batch

#             image = image_tensor.to(device)
#             length_for_pred = torch.IntTensor([batch_max_length] * len(img_batch)).to(device)
#             text_for_pred = torch.LongTensor(config.BATCH_SIZE, batch_max_length + 1).fill_(0).to(device)

#             overlap_batch = overlap_batch.to(device)
#             scene_batch = scene_batch.to(device)

#             preds = model(image, text_for_pred, overlap_batch, scene_batch, is_train=False)


#             _, preds_index = preds.max(2)
#             preds_str = converter.decode(preds_index, length_for_pred)

#             preds_prob = torch.nn.functional.softmax(preds, dim=2)
#             preds_max_prob, _ = preds_prob.max(dim=2)

#             if print_samples:
#                 #if config.SEMANTIC_FORM == 'BERT': print('  - Overlap Objs:', tokenizer.decode(overlap_batch[0]))
#                 print('  - Ground truth:', text_batch[0])
#                 print('  - Prediction:  ', preds_str[0], '\n\n')

#             #time.sleep(10)

#             for anno, img, text, pred, pred_max_prob in zip(anno_batch, img_batch, text_batch, preds_str, preds_max_prob):
                
#                 if anno in base_errors_ids:
#                 pred_EOS = pred.find('[s]')
#                 pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])

#                 if(text == str(pred)):
#                     correct += 1
#                 else:
#                     error_annos.append(anno.item())

#                 total += 1

#     return round(correct*100/total,5), error_annos

# score, error_annos = get_val_score(model)

with open('./results/base_error_ids.txt') as f:
    base_errors_ids = f.read().splitlines()

corrections = 0
total = 0

val_data = get_cocotext_single_image_data(return_loader=False)
print('  - Running evaluation on', len(val_data), 'images')

with torch.no_grad():
    for anno, image, label, overlap, scene in tqdm(val_data):
        # print(anno, base_errors_ids[0])
        # print(type(anno), type(base_errors_ids[0]))
        if str(anno) in base_errors_ids:

            text_in = torch.LongTensor(config.BATCH_SIZE, config.MAX_TEXT_LENGTH + 1).fill_(0).to(config.PRIMARY_DEVICE)
            
            ength_for_pred = torch.IntTensor([config.MAX_TEXT_LENGTH]).to(config.PRIMARY_DEVICE)

            overlap_list = list(overlap.numpy())
            overlap_tags = [obj_class_labels[int(i)-1] for i in overlap_list if i != 0]

            scene_list = list(scene.numpy())
            scene_tags = [obj_class_labels[int(i)-1] for i in scene_list if i != 0]
            #scene_tags = list(zip(scene_tags, [i for i in range(len(scene_tags))]))
            

            # # Add batch dimension
            image = image.unsqueeze(0).to(config.PRIMARY_DEVICE)
            overlap = overlap.unsqueeze(0).to(config.PRIMARY_DEVICE)
            scene = scene.unsqueeze(0).to(config.PRIMARY_DEVICE)

            # #print('over in', overlap.shape)
            pred = model(input=image, text=text_in, overlap=overlap, scene=scene, is_train=False)

            length_for_pred = torch.IntTensor([config.MAX_TEXT_LENGTH]).to(config.PRIMARY_DEVICE)

            _, pred_index = pred.max(2)
            pred_str = converter.decode(pred_index, length_for_pred)[0]
            pred_str = pred_str[:pred_str.find('[s]')]

            # print(overlap_tags)
            # print(pred_str)
            # print(label)

            if label == pred_str:
                corrections += 1
                print(label, pred_str)
                print('  - Correction!\n', round(corrections*100/total,3))
                #time.sleep(15)
                #corrections.append(anno)
            total += 1

print(saved_model)
print('Corrections:', corrections)
print('Total:', total)
            

# with open('./results/sem_errors.txt', 'w') as f:
#     for item in error_annos:
#         f.write("%s\n" % item)