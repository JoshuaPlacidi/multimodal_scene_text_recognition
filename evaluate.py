import config
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

from coco_dataset import LmdbDataset

from PIL import Image
from io import StringIO
import numpy as np
from matplotlib.pyplot import imshow
import numpy as np

model = get_model(config.SAVED_MODEL).eval()

datasets_paths = ["CUTE80", "IC03_860", "IC03_867", "IC13_857", "IC13_1015", "IC15_1811", "IC15_2077", "IIIT5k_3000", "SVT", "SVTP"]



results_df = pd.DataFrame(columns=datasets_paths)
results_df.loc['Correct'] = [0] * len(datasets_paths)
results_df.loc['Total'] = [0] * len(datasets_paths)

print(results_df)

for dataset_name in datasets_paths:
    print('   - Running evaluation on ' + dataset_name)
    cur_dataset_path = config.DEEP_TEXT_DATASET_PATH + 'evaluation/' + dataset_name + '/'
    #cur_dataset_path = '/data_ssd1/jplacidi/deep_text_datasets/evaluation/CUTE80/'
    cur_dataset = LmdbDataset(cur_dataset_path)
    for image, label in tqdm(cur_dataset):
        text_in = torch.LongTensor(config.BATCH_SIZE, config.MAX_TEXT_LENGTH + 1).fill_(0).to(config.PRIMARY_DEVICE)
        
        length_for_pred = torch.IntTensor([config.MAX_TEXT_LENGTH]).to(config.PRIMARY_DEVICE)
        image = image.unsqueeze(0).to(config.PRIMARY_DEVICE)

        overlap = torch.zeros(1).to(config.PRIMARY_DEVICE)
        scene = torch.zeros(1).to(config.PRIMARY_DEVICE)

        pred = model(input=image, text=text_in, overlap=overlap, scene=scene, is_train=False)

        _, pred_index = pred.max(2)
        pred_str = converter.decode(pred_index, length_for_pred)[0]
        pred_str = pred_str[:pred_str.find('[s]')]
        
        results_df.at['Total', dataset_name] += 1

        if label == pred_str: results_df.at['Correct', dataset_name] += 1

    print(results_df)

print(results_df)