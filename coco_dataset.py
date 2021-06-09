import json
import string

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizer

import coco_text
import config

import lmdb
import six
import re
import time

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

class TextOCR_Dataset(Dataset):
    def __init__(self, set='train'):
        self.annotations = []
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((32,100))    

        # Load annotations
        self.annotations = get_textocr_annos(set)
                       
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):

        # load sample
        anno = self.annotations[index]
        img, label, overlap, scene = get_sample(anno)
        img = self.resize(img)
        img = self.to_tensor(img)

        return img, label, overlap, scene

def get_textocr_datasets():
    print('  - Loading data from TextOCR dataset')

    train_data = TextOCR_Dataset(set='train')
    val_data = TextOCR_Dataset(set='val')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=0)

    print('  - ' + str(len(train_loader)) + ' training batches')
    print('  - ' + str(len(val_loader)) + ' val batches')

    return train_loader, val_loader

class COCOText_Dataset(Dataset):
    def __init__(self, set='train'):
        self.annotations = []
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((32,100))    

        # Load annotations
        self.annotations = get_cocotext_annos(set)
                       
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):

        # load sample
        anno = self.annotations[index]
        img, label, overlap, scene = get_sample(anno)
        img = self.resize(img)
        img = self.to_tensor(img)

        return img, label, overlap, scene

class COCOText_Validation_Dataset(Dataset):
    def __init__(self, set='val'):
        self.annotations = []
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((32,100))    

        # Load object annotations
        self.annotations = get_cocotext_annos(set)
                       
    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        anno = self.annotations[index]
        img, label, overlap, scene = get_sample(anno)

        # Load annotation and image
        img = self.resize(img)
        img = self.to_tensor(img)

        return anno['id'], img, label, overlap, scene


def get_cocotext_single_image_data(return_loader=True):
    if return_loader:
        val_loader = torch.utils.data.DataLoader(COCOText_Validation_Dataset(), batch_size=config.BATCH_SIZE, shuffle=False,num_workers=0)
        return val_loader
    else:
        return COCOText_Validation_Dataset()

def char_test():
    return COCOText_Dataset()

def get_cocotext_datasets():
    print('  - Loading data from coco-text dataset')

    train_data = COCOText_Dataset(set='train')
    val_data = COCOText_Dataset(set='val')

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=0)

    print('  - ' + str(len(train_loader)) + ' training batches')
    print('  - ' + str(len(val_loader)) + ' val batches')

    return train_loader, val_loader

def get_synth_datasets():
    print('  - Loading data from sythetic datasets')

    mj_train_data = LmdbDataset(config.DEEP_TEXT_DATASET_PATH + 'training/MJ/MJ_train/')
    mj_test_data = LmdbDataset(config.DEEP_TEXT_DATASET_PATH + 'training/MJ/MJ_test/')
    mj_val_data = LmdbDataset(config.DEEP_TEXT_DATASET_PATH + 'training/MJ/MJ_valid/')
    st_train_data = LmdbDataset(config.DEEP_TEXT_DATASET_PATH + 'training/ST/')

    chained_data = torch.utils.data.ConcatDataset([mj_train_data, mj_test_data, mj_val_data, st_train_data])

    train_loader = torch.utils.data.DataLoader(chained_data, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=0)

    val_data = LmdbDataset(config.DEEP_TEXT_DATASET_PATH + 'validation/')
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=0)

    print('  - ' + str(len(train_loader)) + ' training batches')
    print('  - ' + str(len(val_loader)) + ' val batches')
    return train_loader, val_loader

def get_cocotext_annos(set):
    # Open COCO-Text api
    ct = coco_text.COCO_Text(config.COCO_TEXT_API_PATH)

    # Open text annotations
    with open(config.COCO_TEXT_API_PATH) as f:
        text_annotations = json.load(f)    

    with open('./annotations/features/object_tags.json') as object_annotations: # Load object features
        object_annotations = json.load(object_annotations)

    annotations = []

    for _, anno in tqdm(text_annotations['anns'].items()):
        if anno['legibility'] == 'legible': # If annotation is legibile

            image = ct.loadImgs(ids=anno['image_id']) # Load annotation image data

            if image[0]['set'] == set: # Check if in train or val set
                
                # Load and set annotations image path and its scene and overlap data
                anno['img_path'] = config.IMAGE_PATH + image[0]['file_name']
                objects = object_annotations[str(anno['image_id'])][config.SEMANTIC_SOURCE.lower()]

                if config.SEMANTIC_SOURCE == 'coco' or config.SEMANTIC_SOURCE == 'vg' or config.SEMANTIC_SOURCE == 'vinvl':
                    anno['overlap'] = get_overlap_vec(anno, objects)
                    anno['scene'] = get_scene_vec(objects)
                else:
                    anno['overlap'] = None
                    anno['scene'] = None

                # If set == check annotation is a model compatible string (legal characters, <25 length etc...), if val just check language is english
                if set == 'train':
                    if check_anno(anno['utf8_string']):
                        annotations.append(anno)
                else:
                    if anno['language'] == 'english':
                        annotations.append(anno)

    return annotations

def get_textocr_annos(set):
    if set == 'train':
        anno_path = config.TEXTOCR_PATH + "TextOCR_train.json"
    elif set == 'val':
        anno_path = config.TEXTOCR_PATH + "TextOCR_val.json"
    elif set == 'test':
        anno_path = config.TEXTOCR_PATH + "TextOCR_test.json"
    else:
        raise Exception("TextOCR set:", set, "not recognized")

    # Open text annotations
    with open(anno_path) as f:
        text_annotations = json.load(f)    

    with open('./annotations/features/open_images_vinvl_features.json') as object_annotations: # Load object features
        object_annotations = json.load(object_annotations)

    annotations = []

    for _, anno in tqdm(text_annotations['anns'].items()):
        if anno['legibility'] == 'legible': # If annotation is legibile

            image = text_annotations["imgs"][anno["image_id"]] # Load annotation image data

            if image['set'] == set: # Check if in train or val set
                
                # Load and set annotations image path and its scene and overlap data
                anno['img_path'] = config.IMAGE_PATH + image['file_name']
                objects = object_annotations[str(anno['image_id'])]["vinvl"]

                anno['overlap'] = get_overlap_vec(anno, objects)
                anno['scene'] = get_scene_vec(objects)

                # If set == check annotation is a model compatible string (legal characters, <25 length etc...), if val just check language is english
                if set == 'train':
                    if check_anno(anno['utf8_string']):
                        annotations.append(anno)
                else:
                    if anno['language'] == 'english':
                        annotations.append(anno)

    return annotations

def get_sample(anno):
    label = anno['utf8_string']

    img = Image.open(anno['img_path']).convert('L')
    img = img.crop((anno['bbox'][0], anno['bbox'][1], anno['bbox'][0] + anno['bbox'][2], anno['bbox'][1] + anno['bbox'][3]))  

    padded_overlap = torch.zeros(15)
    padded_scene = torch.zeros(52)

    if anno['overlap']:
        overlap = torch.LongTensor(anno['overlap'])
        overlap_len = len(anno['overlap'])
        padded_overlap[:overlap_len] = overlap

    if anno['scene']:
        scene = torch.LongTensor(anno['scene'])
        scene_len = len(anno['scene'])
        padded_scene[:scene_len] = scene


    return img, label, padded_overlap, padded_scene
    
def check_anno(anno_text):
    return anno_text == anno_text.strip().translate({ord(c): None for c in string.printable[-6:]+'/°-'})[0:25]#string.printable[-38:]+'°'})[0:25]

def get_overlap_vec(anno, objects):
    overlap_vec = []
    for obj in objects:
        obj_class = obj['class'] + 1

        if obj_class not in overlap_vec:

            if config.SEMANTIC_ASSIGNMENT == 'resize': # If assigning using resize method
                if overlap_resize(anno, obj):
                    overlap_vec.append(obj_class)

            else:
                if overlap_iou(anno, obj, float(config.SEMANTIC_ASSIGNMENT)): # If assigning using IoU method
                    overlap_vec.append(obj_class)

    return overlap_vec

def get_scene_vec(objects):
    scene_vec = []
    for obj in objects:
        obj_class = obj['class'] + 1
        if obj_class not in scene_vec:
            scene_vec.append(obj_class)
    if len(scene_vec) != len(set(scene_vec)):
        print('Scene error')
    return scene_vec

def get_bert_tokens(anno, classes, max_length, sequence_pad, key, encode_frequency=False):
    sentence = ""
    for k, v in anno[key].items():
        if encode_frequency:
            for _ in range(v):
                sentence += classes[int(k) + 1] + ' [SEP] '
        else:
            sentence += classes[int(k) + 1] + ' [SEP] '

    sentence = sentence[:-7]

    tokens = torch.tensor(tokenizer.encode(sentence, max_length=sequence_pad, padding='max_length', truncation='longest_first'))

    return tokens

def overlap_resize(text, obj):
    box_area = text['bbox'][2] * text['bbox'][3]
    if box_area == 0: box_area = 1
    scale_factor = text['area'] / box_area # mask area / bbox area
    x_mid = (text['bbox'][2]/2) + text['bbox'][0] # width/2 + x
    y_mid = (text['bbox'][3]/2) + text['bbox'][1] # height/2 + y
    new_width = text['bbox'][2] * scale_factor # width * scale factor
    new_height = text['bbox'][3] * scale_factor # height * scale factor
    new_bbox = [x_mid-(new_width/2),y_mid-(new_height/2),new_width,new_height]

    if((obj['bbox'][0] < new_bbox[0]) and (obj['bbox'][1] < new_bbox[1]) and ((obj['bbox'][0] + obj['bbox'][2]) > (new_bbox[0] + new_bbox[2])) and ((obj['bbox'][1] + obj['bbox'][3]) > (new_bbox[1] + new_bbox[3]))):
        return True
    else:
        return False

from shapely.geometry import Polygon
def overlap_iou(text, obj, threshold):
    #def insertion_over_area(text, obj, percent):
    text_poly = Polygon(get_all_coords(text['bbox']))
    obj_poly = Polygon(get_all_coords(obj['bbox']))
    insert = text_poly.intersection(obj_poly).area
    if insert > 0:
        insert_over_area = insert / (text['bbox'][2] * text['bbox'][3]) # insertion of text bb and obj bb / area of text bb
        #print(insert_over_area)
        if insert_over_area >= threshold: 
            return True
    return False

def get_all_coords(coord_array):
    x1 = coord_array[0]
    y1 = coord_array[1]
    x2 = coord_array[2]
    y2 = coord_array[3]
    return[[x1,y1],[x1+x2,y1],[x1+x2,y1+y2],[x1,y1+y2]]

class LmdbDataset(Dataset):

    def __init__(self, root):

        self.root = root
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            raise Exception

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            self.to_tensor = transforms.ToTensor()
            self.resize = transforms.Resize((32,100))

            if False: # Filtering
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192
                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > 26:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{config.CHARS}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
                img = self.resize(img)
                img = self.to_tensor(img)

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{config.CHARS}]'
            label = re.sub(out_of_char, '', label)

            overlap = torch.zeros(1).to(config.PRIMARY_DEVICE)
            scene = torch.zeros(1).to(config.PRIMARY_DEVICE)

        return img, label, overlap, scene