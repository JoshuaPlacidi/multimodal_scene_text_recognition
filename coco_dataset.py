import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import json
import coco_text
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import string
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_ids = [0]

# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = torch.nn.DataParallel(BertModel.from_pretrained('bert-base-uncased'), device_ids=device_ids).to(device)

class COCO_Text_Dataset(Dataset):
    def __init__(self, coco_text_json, combined_coco_json, img_dir, get_set='train'):
        ct = coco_text.COCO_Text(coco_text_json)
        self.annotations = []
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((32,100))
        with open(combined_coco_json) as f:
            data = json.load(f)
        for anno in tqdm(data['combined_annos']):
            image = ct.loadImgs(ids=anno['image_id'])
            if image[0]['set'] == get_set:#anno['legibility'] == 'legible' and anno['language'] == 'english' and 
                anno['set'] = image[0]['set']
                if(get_set == 'train'):
                    # If in train set make sure anno format is compatible for training
                    if check_anno(anno['utf8_string']) and anno['legibility'] == 'legible':# and anno['utf8_string'] != '':
                        anno['img_path'] = img_dir + image[0]['file_name']
                        self.annotations.append(anno)
                    
                elif(get_set == 'val'):
                    if anno['legibility'] == 'legible' and anno['language'] == 'english':
                    # If in val set then dont need to check anno format
                        anno['img_path'] = img_dir + image[0]['file_name']
                        self.annotations.append(anno)  
                       
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        anno = self.annotations[index]
        img = Image.open(anno['img_path']).convert('L')
        img = img.crop((anno['bbox'][0], anno['bbox'][1], anno['bbox'][0] + anno['bbox'][2], anno['bbox'][1] + anno['bbox'][3]))  
        img = self.resize(img)
        img = self.to_tensor(img)
        
        scene_vector = torch.tensor(anno['scene_vector'])
        overlap_vector = torch.tensor(anno['overlap_vector'])
        
        return anno['img_path'], img, anno['utf8_string'], scene_vector, overlap_vector

        #objects = ''
        # if(len(anno['objects']) > 0):
        #     for obj in anno['objects']:
        #         objects += classes_90[obj] + ' '
        # else:
        #     objects = 'background'
        # objects = torch.tensor(tokenizer.encode(str(objects))).unsqueeze(0).to('cuda:0')
        # objects = bert_model(objects)[1]

        #return anno['img_path'], img, anno['utf8_string'], objects, scene_vector, overlap_vector

class Combined_Dataset(Dataset):
    def __init__(self, txt_anns, feats, img_dir, set='train'):
        
        ct = coco_text.COCO_Text(txt_anns)
        self.annotations = []
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize((32,100))

        with open(txt_anns) as f:
            txt_data = json.load(f)

        with open(feats) as f:
            feats = json.load(f)

        for _, anno in tqdm(txt_data['anns'].items()):
            if anno['legibility'] == 'legible':
                image = ct.loadImgs(ids=anno['image_id'])
                if image[0]['set'] == set:
                    anno['img_path'] = img_dir + image[0]['file_name']

                    anno['scene'] = get_object_vec(feats[str(anno['image_id'])]['scene'],1065)
                    anno['overlap'] = get_object_vec(feats[str(anno['image_id'])]['overlap'][str(anno['id'])],972)
                    if set == 'train':
                        if check_anno(anno['utf8_string']):
                            self.annotations.append(anno)
                    else:
                        if anno['language'] == 'english':
                            self.annotations.append(anno)
                       
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        anno = self.annotations[index]
        img = Image.open(anno['img_path']).convert('L')
        img = img.crop((anno['bbox'][0], anno['bbox'][1], anno['bbox'][0] + anno['bbox'][2], anno['bbox'][1] + anno['bbox'][3]))  
        img = self.resize(img)
        img = self.to_tensor(img)
        
        scene_vector = torch.tensor(anno['scene'])
        overlap_vector = torch.tensor(anno['overlap'])
        
        return anno['img_path'], img, anno['utf8_string'], scene_vector, overlap_vector

def get_object_vec(d, length):
    vec = [0]*length
    for key, val in d.items():
        vec[int(key)] = val
    return vec
    
def check_anno(anno_text):
    return anno_text == anno_text.strip().translate({ord(c): None for c in string.printable[-6:]+'/°-'})[0:25]#string.printable[-38:]+'°'})[0:25]

def get_datasets(batch_size):
    print('--- Loading Data')
    txt_annos_path = "F:/dev/Datasets/COCO/2014/COCO_Text_2014.json"
    feats_path = './comb_data/VG/area_resize.json'#'./comb_data/VG_area_resize_iou75.json' #new_comb.json  './comb_data/tfidf_area_resize.json'
    image_path = "F:/dev/Datasets/COCO/2014/images/train2014/"

    train_data = Combined_Dataset(txt_annos_path, feats_path, image_path, set='train')
    val_data = Combined_Dataset(txt_annos_path, feats_path, image_path, set='val')

    train_ldr = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=0)
    val_ldr = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True,num_workers=0)

    print('  - ' + str(len(train_ldr) * batch_size) + ' training samples')
    print('  - ' + str(len(val_ldr) * batch_size) + ' val samples')

    return train_ldr, val_ldr