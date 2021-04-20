DEVICE_IDS=[0]
PRIMARY_DEVICE = 'cuda:' + str(DEVICE_IDS[0])
RANDOM_SEED = 999

COCO_TEXT_API_PATH = './annotations/COCO_Text_2014.json'
IMAGE_PATH = "/datassd1/jplacidi/coco_data/images/train2014/"

SEMANTIC_FORM = 'BERT' # FREQ | ZERO | RAND

BATCH_SIZE = 192

SEMANTIC_SOURCE = 'VG'

EXPERIMENT = 'experiment_name'#e6d1_emb64_SOS_TEST1'#'TEARS_2layers_lr0.0001_TFdecoder'

MAX_TEXT_LENGTH = 25

EPOCHS = 12

SAVED_MODEL = 'TPS-ResNet-BiLSTM-Attn-case-sensitive.pth'#'./results/models/Full_TF_e2_d2_TEARS_0_00001_batch_64_e_50.pt'

EMBED_DIM = 64

ENCODER = 'Transformer' # LSTM | Transformer
DECODER = 'Linear' # LSTM-Atn | Transformer | Linear