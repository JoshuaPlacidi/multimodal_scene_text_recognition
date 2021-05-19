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

from coco_dataset import LmdbDataset, get_val_data

from PIL import Image
from io import StringIO
import numpy as np
from matplotlib.pyplot import imshow
import numpy as np
import time


saved_model = './results/models/scratch.pt'

model = get_model(saved_model)
model.eval()

import json
with open('./annotations/features/' + config.SEMANTIC_SOURCE.lower() + '_classes.txt') as f:
    obj_class_labels = f.read().splitlines()

with open('./annotations/error_anno_ids.txt') as f:
    error_ids = f.read().splitlines()

val_data = get_val_data()

errors = []

print('  - Running evaluation on', len(val_data), 'images')
#time.sleep(15)

corrections = 0
total = 0

ids = [17164, 156586, 14847, 27009, 105279, 103047, 101651, 123201, 77915, 21877, 155509, 159040, 43365, 168639, 38327]

with torch.no_grad():
    for image, label, scene, overlap in tqdm(val_data):
        # print(type(error_ids[0]), type(anno['id']))
        # break
        if True:#str(anno['id']) not in error_ids:
            #print(anno)
            text_in = torch.LongTensor(config.BATCH_SIZE, config.MAX_TEXT_LENGTH + 1).fill_(0).to(config.PRIMARY_DEVICE)
            
            length_for_pred = torch.IntTensor([config.MAX_TEXT_LENGTH]).to(config.PRIMARY_DEVICE)

            overlap_list = list(overlap.numpy())
            overlap_tags = [obj_class_labels[int(i)-1] for i in overlap_list if i != 0]

            scene_list = list(scene.numpy())
            scene_tags = [obj_class_labels[int(i)-1] for i in scene_list if i != 0]
            scene_tags = list(zip(scene_tags, [i for i in range(len(scene_tags))]))
            

            # Add batch dimension
            image = image.unsqueeze(0).to(config.PRIMARY_DEVICE)
            overlap = overlap.unsqueeze(0).to(config.PRIMARY_DEVICE)
            scene = scene.unsqueeze(0).to(config.PRIMARY_DEVICE)

            #print('over in', overlap.shape)
            pred = model(input=image, text=text_in, overlap=overlap, scene=scene, is_train=False)

            _, pred_index = pred.max(2)
            pred_str = converter.decode(pred_index, length_for_pred)[0]
            pred_str = pred_str[:pred_str.find('[s]')]
            print(pred_str, label)
            #print(anno['id'], ': ' + label + ' : ' + pred_str)
            # if label != pred_str:
            #     # print(anno['id'])
            #     # print(scene_tags)
            #     #print(overlap_tags)
            #     anno['pred'] = pred_str
            #     errors.append(anno)
            #     #time.sleep(30)
                
            #     corrections += 1
            # #     # print('Corrections:', corrections)
            # #     # print('Total', total)
            #     print('Corrections:', round(corrections * 100/total, 2))
            total += 1

print('Corrections:', corrections)
print('Total:', len(val_data))
            

            

#         if pred_str != label:
#             anno['pred'] = pred_str
#             errors.append(anno)

with open('./results/sem_errors.txt', 'w') as f:
    for item in errors:
        f.write("%s\n" % item)

# print('  - Saved to file')

# datasets_paths = ["CUTE80", "IC03_860", "IC03_867", "IC13_857", "IC13_1015", "IC15_1811", "IC15_2077", "IIIT5k_3000", "SVT", "SVTP"]



# results_df = pd.DataFrame(columns=datasets_paths)
# results_df.loc['Correct'] = [0] * len(datasets_paths)
# results_df.loc['Total'] = [0] * len(datasets_paths)
# results_df.loc['Score'] = [0] * len(datasets_paths)

# print(results_df)

# for dataset_name in datasets_paths:
#     print('   - Evaluating on ' + dataset_name)
#     cur_dataset_path = config.DEEP_TEXT_DATASET_PATH + 'evaluation/' + dataset_name + '/'
#     #cur_dataset_path = '/data_ssd1/jplacidi/deep_text_datasets/evaluation/CUTE80/'
#     cur_dataset = LmdbDataset(cur_dataset_path)
#     for image, label, overlap, scene in tqdm(cur_dataset):
#         text_in = torch.LongTensor(config.BATCH_SIZE, config.MAX_TEXT_LENGTH + 1).fill_(0).to(config.PRIMARY_DEVICE)
        
#         length_for_pred = torch.IntTensor([config.MAX_TEXT_LENGTH]).to(config.PRIMARY_DEVICE)
#         image = image.unsqueeze(0).to(config.PRIMARY_DEVICE)

#         overlap = torch.zeros(1).to(config.PRIMARY_DEVICE)
#         scene = torch.zeros(1).to(config.PRIMARY_DEVICE)

#         pred = model(input=image, text=text_in, overlap=overlap, scene=scene, is_train=False)

#         _, pred_index = pred.max(2)
#         pred_str = converter.decode(pred_index, length_for_pred)[0]
#         pred_str = pred_str[:pred_str.find('[s]')]
        
#         results_df.at['Total', dataset_name] += 1

#         if label == pred_str: results_df.at['Correct', dataset_name] += 1

#     score =  results_df.at['Correct', dataset_name] / results_df.at['Total', dataset_name]

#     results_df.at['Score', dataset_name] = round(score*100, 3)

#     print(results_df)