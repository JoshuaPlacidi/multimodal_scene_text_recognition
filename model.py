import config
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
torch.backends.cudnn.enabled = False

from utils import AttnLabelConverter
import string, json

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor,RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.init_generation import Random_Initilisation, Zero_Initilisation, Bert_Initilisation, Frequency_Initilisation
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention

# model
transformation = 'TPS'
features = 'ResNet'
sequence = 'BiLSTM'
prediction = 'Attn'

character = string.printable[:-6]
converter = AttnLabelConverter(character)
num_classes = len(converter.character)

batch_max_length = 25
imgH = 32
imgW = 100
num_fiducial = 20

# feature extraction params
input_channel = 1
output_channel = 512

# LSTM hidden state size
hidden_size = 256

rgb = False
input_channel = 1
if rgb: input_channel = 3

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.stages = {'Trans': transformation, 'Feat': features,
                       'Seq': sequence, 'Pred': prediction}

        """ Transformation """
        if transformation == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=num_fiducial, I_size=(imgH, imgW), I_r_size=(imgH, imgW), I_channel_num=input_channel)
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if features == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        elif features == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(input_channel, output_channel)
        elif features == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        if config.ANNOTATION == 'ZERO':
            self.init_generation = Zero_Initilisation()
        elif config.ANNOTATION == 'RAND':
            self.init_generation = Random_Initilisation()
        elif config.ANNOTATION == 'BERT':
            self.init_generation = Bert_Initilisation()
        elif config.ANNOTATION == 'FREQ':
            self.init_generation = Frequency_Initilisation()

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        """ Prediction """
        self.Prediction = Attention(self.SequenceModeling_output, hidden_size, num_classes)

    #def forward(self, input, text, scene_semantic, overlap_semantic, is_train=True):
    def forward(self, input, text, scene, overlap, is_train=True):
        # Transformation
        input = self.Transformation(input)
        
        # Feature Extraction
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        
        visual_feature = visual_feature.squeeze(3)

        features_for_lang = visual_feature

        init_hidd, init_cell = self.init_generation(overlap, scene)

        # Sequence
        contextual_feature, _, _ = self.SequenceModeling((features_for_lang, init_hidd, init_cell))

        # Prediction
        prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=batch_max_length)

        return prediction

def get_model():
    return Model()