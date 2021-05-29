import string

# Set cuda devices to use
DEVICE_IDS=[0,1]
PRIMARY_DEVICE = 'cuda:' + str(DEVICE_IDS[0])

'''
 Experiment paramenters
'''
# Name of experiement (used to save files)
EXPERIMENT = 'oscar_encoder_from_scratch'

# Pretained model to use
SAVED_MODEL = None#'./results/models/scratch.pt'#'./results/models/pre_encoder_mlp_overlap_vinvl_resize_linear.pt'#

RANDOM_SEED = 999
BATCH_SIZE = 192
EPOCHS = 4
MAX_TEXT_LENGTH = 25
CHARS = string.printable[:-6]
MODEL_SAVE_THRESHOLD = 0 # Once val accuraccy % passes this threshold highest accuraccy models are saved to ./results/models


'''
 Model design
'''
ENCODER = 'Oscar' # LSTM  | Transformer   | Oscar
DECODER = 'Transformer' # LSTM  | Transformer   | Linear

# Dimensions
EMBED_DIM = 256
HIDDEN_DIM = 512

# Semantic vector processing
SEMANTIC_VECTOR = 'scene'     # overlap | scene | combined
SEMANTIC_SOURCE = 'vinvl'       # coco  | vg    | vinvl | zero  | rand
SEMANTIC_ASSIGNMENT = 'resize'  # .25   | .50   | .75   | resize (if .25/.50/.75 then using iou assignment)
SEMANTIC_EMBEDDING = 'linear'   # bert  | linear

'''
 Fusion Strategies
'''
PRINT_ATTENTION_SCORES = False
# Encoder
PRE_ENCODER_MLP = False
OSCAR_ENCODER = False
# Decoder
PRE_DECODER_MLP = False
CLS_DECODER_INIT = False       # NOT implemented
MULTIHEAD_PRE_TARGET = False
MULTIHEAD_PRE_MEMORY = False
MULTIHEAD_POST_MEMORY = False


'''
 Local paths
'''
# COCO text json file path
COCO_TEXT_API_PATH = './annotations/COCO_Text_2014.json'
# COCO train 2014 image folder path
IMAGE_PATH = "/data_ssd1/jplacidi/coco_data/images/train2014/"
# Deep Text dataset folders
DEEP_TEXT_DATASET_PATH = "/data_ssd1/jplacidi/deep_text_datasets/"

















