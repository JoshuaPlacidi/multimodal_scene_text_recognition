# Set devices to use
DEVICE_IDS=[1,0]
PRIMARY_DEVICE = 'cuda:' + str(DEVICE_IDS[0])

#
# Experiment parameters
#
RANDOM_SEED = 999
BATCH_SIZE = 192
EPOCHS = 20
MAX_TEXT_LENGTH = 25
MODEL_SAVE_THRESHOLD = 65 # Once val accuraccy % passes this threshold highest accuraccy models are saved to ./results/models

# Pretained model to use
SAVED_MODEL = 'TPS-ResNet-BiLSTM-Attn-case-sensitive.pth'#'./results/models/full_transformer.pt'#'./results/models/Full_TF_e2_d2_TEARS_0_00001_batch_64_e_50.pt'
#'./results/models/transformer_decoder_1_e_16.pt'#


# Model structure
ENCODER = 'Transformer' # LSTM | Transformer
DECODER = 'Transformer' # LSTM-Atn | Transformer | Linear

# Netowork Dimensions (only used for transformer encoder currently)
EMBED_DIM = 64
HIDDEN_DIM = 512

# Semantic vector processing
SEMANTIC_SOURCE = 'COCO'
SEMANTIC_FORM = 'FREQ' # BERT | FREQ | ZERO | RAND

# Name of experiement (used to save files)
EXPERIMENT = 'uni_seq__OandS_no_freq_1'#e6d1_emb64_SOS_TEST1'#'TEARS_2layers_lr0.0001_TFdecoder'



# COCO text json file path
COCO_TEXT_API_PATH = './annotations/COCO_Text_2014.json'
# COCO train 2014 image folder path
IMAGE_PATH = "/data_ssd1/jplacidi/coco_data/images/train2014/"

















