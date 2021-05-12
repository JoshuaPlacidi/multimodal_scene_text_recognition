import string

# Set devices to use
DEVICE_IDS=[3,4]
PRIMARY_DEVICE = 'cuda:' + str(DEVICE_IDS[0])

#
# Experiment parameters
#
RANDOM_SEED = 999
BATCH_SIZE = 192
EPOCHS = 10
MAX_TEXT_LENGTH = 25
CHARS = string.printable[:-6]
MODEL_SAVE_THRESHOLD = 0 # Once val accuraccy % passes this threshold highest accuraccy models are saved to ./results/models

# Pretained model to use
SAVED_MODEL = './results/models/new_mlp.pt'#base.pth'#./results/models/SCRATCH_SYTH_e_3.pt'#'base.pth'#


# Model structure
ENCODER = 'Transformer' # LSTM | Transformer
DECODER = 'Transformer' # LSTM | Transformer | Linear

# Netowork Dimensions (only used for transformer encoder currently)
EMBED_DIM = 64
HIDDEN_DIM = 512

# Semantic vector processing
SEMANTIC_SOURCE = 'vinvl' # VG | VinVL
SEMANTIC_FORM = 'FREQ' # BERT | FREQ | ZERO | RAND

# Name of experiement (used to save files)
EXPERIMENT = 'new_mlp'#e6d1_emb64_SOS_TEST1'#'TEARS_2layers_lr0.0001_TFdecoder'


#
# Local Paths
#
# COCO text json file path
COCO_TEXT_API_PATH = './annotations/COCO_Text_2014.json'
# COCO train 2014 image folder path
IMAGE_PATH = "/data_ssd1/jplacidi/coco_data/images/train2014/"
# Deep Text dataset folders
DEEP_TEXT_DATASET_PATH = "/data_ssd1/jplacidi/deep_text_datasets/"

















