# Set devices to use
DEVICE_IDS=[0,1]
PRIMARY_DEVICE = 'cuda:' + str(DEVICE_IDS[0])

#
# Experiment parameters
#
RANDOM_SEED = 999
BATCH_SIZE = 192
EPOCHS = 12
MAX_TEXT_LENGTH = 25
MODEL_SAVE_THRESHOLD = 100 # Once val accuraccy % passes this threshold highest accuraccy models are saved to ./results/models

# Pretained model to use
SAVED_MODEL = 'TPS-ResNet-BiLSTM-Attn-case-sensitive.pth'#'./results/models/Full_TF_e2_d2_TEARS_0_00001_batch_64_e_50.pt'

# Model structure
ENCODER = 'Transformer' # LSTM | Transformer
DECODER = 'Linear' # LSTM-Atn | Transformer | Linear

# Embed dim (only used for transformer encoder currently)
EMBED_DIM = 64

# Semantic vector processing
SEMANTIC_SOURCE = 'VG'
SEMANTIC_FORM = 'ZERO' # BERT | FREQ | ZERO | RAND

# Name of experiement (used to save files)
EXPERIMENT = 'linear_1'#e6d1_emb64_SOS_TEST1'#'TEARS_2layers_lr0.0001_TFdecoder'



# COCO text json file path
COCO_TEXT_API_PATH = './annotations/COCO_Text_2014.json'
# COCO train 2014 image folder path
IMAGE_PATH = "/data_ssd1/jplacidi/coco_data/images/train2014/"

















