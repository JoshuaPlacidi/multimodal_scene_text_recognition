import config
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
torch.backends.cudnn.enabled = False

from utils import AttnLabelConverter
import time

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import ResNet_FeatureExtractor
from modules.semantic_vectors import Linear_Embedding, Bert_Embedding, Zero, Random
from modules.encoders import BidirectionalLSTM, TF_Encoder, Oscar_Bert
from modules.decoders import Attention, TF_Decoder, Linear_Decoder

character = config.CHARS
converter = AttnLabelConverter(character)
num_classes = len(converter.character)

imgH = 32
imgW = 100
num_fiducial = 20

# feature extraction params
input_channel = 1
output_channel = 512

# LSTM hidden state size
hidden_size = 256

input_channel = 1

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

        # Spatial Transformation Network
        self.Transformation = TPS_SpatialTransformerNetwork(
                F=num_fiducial, I_size=(imgH, imgW), I_r_size=(imgH, imgW), I_channel_num=input_channel)


        # Feature Extraction
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        # Semantic Vectors
        if config.SEMANTIC_SOURCE == 'zero':
            self.get_semantic_vectors = Zero()
        elif config.SEMANTIC_SOURCE  == 'rand':
            self.get_semantic_vectors = Random()
        elif config.SEMANTIC_EMBEDDING  == 'bert':
            self.get_semantic_vectors = Bert_Embedding()
        elif config.SEMANTIC_EMBEDDING  == 'linear':
            self.get_semantic_vectors = Linear_Embedding()
        else:
            raise Exception("Model.py Semantic vector error: check config.SEMANTIC_SOURCE and config.SEMANTIC_EMBEDDING")


        # Encoder
        if config.ENCODER == 'LSTM':
            self.encoder = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        elif config.ENCODER == 'Transformer':
            self.encoder = TF_Encoder()
        elif config.ENCODER == 'Oscar':
            self.encoder = Oscar_Bert()
        else:
            raise Exception("Model.py Encoder Error: '" + config.ENCODER + "' not recognized")

        # Decoder
        if config.DECODER == "LSTM":
            self.decoder = Attention(256, 256, num_classes)
        elif config.DECODER == "Transformer":
            self.decoder = TF_Decoder(num_classes)
        elif config.DECODER == "Linear":
            self.decoder = Linear_Decoder(num_classes)
        else:
            raise Exception("Model.py Decoder Error: '" + config.DECODER + "' not recognized")


    def forward(self, input, text, overlap, scene, is_train=True):

        # Transformation
        input = self.Transformation(input)
        
        # Feature Extraction
        visual_features = self.FeatureExtraction(input)
        visual_features = self.AdaptiveAvgPool(visual_features.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_features = visual_features.squeeze(3)

        # Semantic Vectors
        semantics = self.get_semantic_vectors(overlap, scene)

        # Encode
        if config.ENCODER == 'LSTM':
            encoded_features = self.encoder(visual_features)
        else:
            encoded_features = self.encoder(col_feats=visual_features, semantics=semantics, is_train=is_train)

        # Decode
        prediction = self.decoder(encoded_features.contiguous(), text=text, semantics=semantics, is_train=is_train)

        return prediction

def get_model(saved_model=None):
    '''
    input: saved_model = the path to pretrained weights to load model from, if None then wont load any weights
    return: model
    '''
    model = Model()
    model = torch.nn.DataParallel(model, device_ids=config.DEVICE_IDS)
    model = model.to(config.PRIMARY_DEVICE)

    if saved_model:
        print('  - Loading model from:', saved_model)
        pretrained_dict = torch.load(saved_model, map_location=lambda storage, loc: storage)

        model.load_state_dict(pretrained_dict, strict=False)

    else:
        print('  - Training from scratch (no pretrained weights provided)')

    return model