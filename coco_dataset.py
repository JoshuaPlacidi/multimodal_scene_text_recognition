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


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

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


class COCOText_Dataset(Dataset):
    def __init__(self, set='train'):
        
        # Open COCO-Text api
        ct = coco_text.COCO_Text(config.COCO_TEXT_API_PATH)

        self.annotations = []
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((32,100))

        # Open text annotations
        with open(config.COCO_TEXT_API_PATH) as f:
            text_annotations = json.load(f)        

        # Load object annotations

        with open('./annotations/features/object_tags.json') as object_annotations: # Load object features
            self.object_annotations = json.load(object_annotations)

        # Process text annotations
        c = 0
        for _, anno in tqdm(text_annotations['anns'].items()):
            c += 1
            if anno['legibility'] == 'legible': # If annotation is legibile

                image = ct.loadImgs(ids=anno['image_id']) # Load annotation image data

                if image[0]['set'] == set: # Check if in train or val set
                    
                    # Load and set annotations image path and its scene and overlap data
                    anno['img_path'] = config.IMAGE_PATH + image[0]['file_name']
                    objects = self.object_annotations[str(anno['image_id'])][config.SEMANTIC_SOURCE.lower()]
                    anno['overlap'] = self.get_overlap_vec(anno, objects)
                    anno['scene'] = self.get_scene_vec(objects)
                    # print(anno)
                    # print(anno['overlap'], '\n\n')

                    # If set == check annotation is a model compatible string (legal characters, <25 length etc...), if val just check language is english
                    if set == 'train':
                        if check_anno(anno['utf8_string']):
                            self.annotations.append(anno)
                    else:
                        if anno['language'] == 'english':
                            self.annotations.append(anno)
                       
    def __len__(self):
        return len(self.annotations)

    def get_scene_vec(self, objects):
        scene_vec = []
        # for obj in objects:
        #     scene_vec.append(obj['class'])
        return [0]#scene_vec

    def overlap(self, text, obj):
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

    def get_overlap_vec(self, anno, objects):
        overlap_vec = []
        for obj in objects:
            if self.overlap(anno, obj):
                overlap_vec.append(obj['class'])
        return overlap_vec
    
    def __getitem__(self, index):

        # Load annotation and image
        anno = self.annotations[index]
        img = Image.open(anno['img_path']).convert('L')
        img = img.crop((anno['bbox'][0], anno['bbox'][1], anno['bbox'][0] + anno['bbox'][2], anno['bbox'][1] + anno['bbox'][3]))  
        img = self.resize(img)
        img = self.to_tensor(img)
        
        padded_overlap = torch.zeros(20)
        overlap = torch.LongTensor(anno['overlap'])
        padded_overlap[:len(anno['overlap'])] = overlap

        padded_scene = torch.zeros(200)
        scene = torch.LongTensor(anno['scene'])
        padded_scene[:len(anno['scene'])] = scene

        return img, anno['utf8_string'], padded_scene, padded_overlap

def get_cocotext_datasets():
    print('  - Loading data from coco-text dataset')
    # txt_annos_path = "F:/dev/Datasets/COCO/2014/COCO_Text_2014.json"
    # feats_path = './comb_data/VG/area_resize.json'#'./comb_data/VG_area_resize_iou75.json' #new_comb.json  './comb_data/tfidf_area_resize.json'
    # image_path = "F:/dev/Datasets/COCO/2014/images/train2014/"

    train_data = COCOText_Dataset(set='train')
    val_data = COCOText_Dataset(set='val')


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=0)

    print('  - ' + str(len(train_loader)) + ' training batches')
    print('  - ' + str(len(val_loader)) + ' val batches')

    return train_loader, val_loader

class COCOText_Validation_Dataset(Dataset):
    def __init__(self):
        
        # Open COCO-Text api
        ct = coco_text.COCO_Text(config.COCO_TEXT_API_PATH)

        self.annotations = []
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((32,100))

        # Open text annotations
        with open(config.COCO_TEXT_API_PATH) as f:
            text_annotations = json.load(f)        

        # Load object annotations

        with open('./annotations/features/object_tags.json') as object_annotations: # Load object features
            self.object_annotations = json.load(object_annotations)

        # Process text annotations
        c = 0
        for _, anno in tqdm(text_annotations['anns'].items()):
            c += 1
            if anno['legibility'] == 'legible': # If annotation is legibile

                image = ct.loadImgs(ids=anno['image_id']) # Load annotation image data

                if image[0]['set'] == 'val': # Check if in train or val set
                    
                    # Load and set annotations image path and its scene and overlap data
                    anno['img_path'] = config.IMAGE_PATH + image[0]['file_name']
                    objects = self.object_annotations[str(anno['image_id'])][config.SEMANTIC_SOURCE.lower()]
                    anno['overlap'] = self.get_overlap_vec(anno, objects)
                    anno['scene'] = self.get_scene_vec(objects)
                    # print(anno)
                    # print(anno['overlap'], '\n\n')

                    # If set == check annotation is a model compatible string (legal characters, <25 length etc...), if val just check language is english
                    if set == 'train':
                        if check_anno(anno['utf8_string']):
                            self.annotations.append(anno)
                    else:
                        if anno['language'] == 'english':
                            self.annotations.append(anno)
                       
    def __len__(self):
        return len(self.annotations)

    def get_scene_vec(self, objects):
        scene_vec = []
        # for obj in objects:
        #     scene_vec.append(obj['class'])
        return [0]#scene_vec

    def overlap(self, text, obj):
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

    def get_overlap_vec(self, anno, objects):
        overlap_vec = []
        for obj in objects:
            if self.overlap(anno, obj):
                overlap_vec.append(obj['class'])
        return overlap_vec
    
    def __getitem__(self, index):

        # Load annotation and image
        anno = self.annotations[index]
        img = Image.open(anno['img_path']).convert('L')
        img = img.crop((anno['bbox'][0], anno['bbox'][1], anno['bbox'][0] + anno['bbox'][2], anno['bbox'][1] + anno['bbox'][3]))  
        img = self.resize(img)
        img = self.to_tensor(img)
        
        padded_overlap = torch.zeros(20)
        overlap = torch.LongTensor(anno['overlap'])
        padded_overlap[:len(anno['overlap'])] = overlap

        padded_scene = torch.zeros(200)
        scene = torch.LongTensor(anno['scene'])
        padded_scene[:len(anno['scene'])] = scene

        return anno, img, anno['utf8_string'], padded_scene, padded_overlap

def get_val_data():
    #val_loader = torch.utils.data.DataLoader(COCOText_Validation_Dataset(), batch_size=config.BATCH_SIZE, shuffle=False,num_workers=0)
    return COCOText_Validation_Dataset()


class Annotations_Dataset(Dataset):
    def __init__(self, set='train'):
        
        # Open COCO-Text api
        ct = coco_text.COCO_Text(config.COCO_TEXT_API_PATH)

        self.annotations = []
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((32,100))

        # Open text annotations
        with open(config.COCO_TEXT_API_PATH) as f:
            text_annotations = json.load(f)        

        # Load object annotations
        self.classes = ['background']

        if config.SEMANTIC_SOURCE == 'VG': # If using Visual Genome object data
            with open('./annotations/features/vg_frequency.json') as object_annotations: # Load object features
                self.object_annotations = json.load(object_annotations)

            with open('./annotations/features/vg_classes.txt') as VG_labels: # Load visual genome class labels
                for object in VG_labels.readlines():
                    self.classes.append(object.split(',')[0].lower().strip().replace("'", ''))

        elif config.SEMANTIC_SOURCE == 'COCO': # If using MS COCO object data
            with open('./annotations/features/coco_frequency.json') as object_annotations:
                self.object_annotations = json.load(object_annotations)

            with open('./annotations/features/coco_classes.txt') as COCO_labels: # Load MS COCO class labels
                for object in COCO_labels.readlines():
                    self.classes.append(object.split(',')[0].lower().strip().replace("'", ''))

        if config.SEMANTIC_FORM == 'FREQ':
            self.overlap_vector_size = len(self.classes)
            self.scene_vector_size = len(self.classes)

        # Process text annotations
        for _, anno in tqdm(text_annotations['anns'].items()):
            if anno['legibility'] == 'legible': # If annotation is legibile

                image = ct.loadImgs(ids=anno['image_id']) # Load annotation image data

                if image[0]['set'] == set: # Check if in train or val set
                    
                    # Load and set annotations image path and its scene and overlap data
                    anno['img_path'] = config.IMAGE_PATH + image[0]['file_name']
                    anno['scene'] = self.object_annotations[str(anno['image_id'])]['scene']
                    anno['overlap'] = self.object_annotations[str(anno['image_id'])]['overlap'][str(anno['id'])]

                    # If set == check annotation is a model compatible string (legal characters, <25 length etc...), if val just check language is english
                    if set == 'train':
                        if check_anno(anno['utf8_string']):
                            self.annotations.append(anno)
                    else:
                        if anno['language'] == 'english':
                            self.annotations.append(anno)
                       
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):

        # Load annotation and image
        anno = self.annotations[index]
        img = Image.open(anno['img_path']).convert('L')
        img = img.crop((anno['bbox'][0], anno['bbox'][1], anno['bbox'][0] + anno['bbox'][2], anno['bbox'][1] + anno['bbox'][3]))  
        img = self.resize(img)
        img = self.to_tensor(img)
        
        if config.SEMANTIC_FORM == 'BERT':
            overlap = get_bert_tokens(anno, self.classes, 7, 20, 'overlap')
            scene = get_bert_tokens(anno, self.classes, 7, 60, 'scene')

        elif config.SEMANTIC_FORM == 'FREQ':
            #print(anno['overlap'], anno['scene'])

            overlap_padded = torch.zeros(10)
            overlap_objs = torch.LongTensor(get_object_vector(anno['overlap'], self.overlap_vector_size))
            if overlap_objs.shape[0] > 0:
                overlap_objs = overlap_objs
                overlap_padded[:overlap_objs.shape[0]] = overlap_objs

            scene_padded = torch.zeros(20)
            scene_objs = torch.LongTensor(get_object_vector(anno['scene'], self.scene_vector_size))
            if scene_objs.shape[0] > 0:
                scene_objs = scene_objs
                scene_padded[:scene_objs.shape[0]] = scene_objs

            
            #print(overlap_padded.shape)

        else:
            overlap = torch.zeros(1)
            scene = torch.zeros(1)

        return img, anno['utf8_string'], scene_padded, overlap_padded

def get_bert_tokens(anno, classes, max_length, sequence_pad, key, encode_frequency = False):
    # tokens = []
    # for key, val in anno[key].items():
    #     for _ in range(val):
    #         token = torch.tensor(tokenizer.encode(classes[int(key)], max_length=max_length, padding='max_length'))
    #         tokens.append(token) 
    # if not tokens: # If tokens is empty
    #     token = torch.tensor(tokenizer.encode(classes[0], max_length=max_length, padding='max_length'))
    #     tokens.append(token)

    # for _ in range(sequence_pad-len(tokens)):
    #     tokens.append(torch.zeros(max_length))

    # tokens = torch.stack(tokens)

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

# takes a dictionary of type dict[object_label] = count and converts it to a sparse vector
def get_object_vector(dict, length):

    objs_vectors = []

    for key, val in dict.items():
        #for _ in range(val):
        objs_vectors.append(int(key))

    return objs_vectors
    
def check_anno(anno_text):
    return anno_text == anno_text.strip().translate({ord(c): None for c in string.printable[-6:]+'/°-'})[0:25]#string.printable[-38:]+'°'})[0:25]

def get_datasets():
    print('  - Loading data from coco-text dataset')
    # txt_annos_path = "F:/dev/Datasets/COCO/2014/COCO_Text_2014.json"
    # feats_path = './comb_data/VG/area_resize.json'#'./comb_data/VG_area_resize_iou75.json' #new_comb.json  './comb_data/tfidf_area_resize.json'
    # image_path = "F:/dev/Datasets/COCO/2014/images/train2014/"

    train_data = Annotations_Dataset(set='train')
    val_data = Annotations_Dataset(set='val')


    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config.BATCH_SIZE, shuffle=True,num_workers=0)

    print('  - ' + str(len(train_loader)) + ' training batches')
    print('  - ' + str(len(val_loader)) + ' val batches')

    return train_loader, val_loader

def get_syth_datasets():
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