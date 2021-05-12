import config
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
torch.backends.cudnn.enabled = False

from utils import AttnLabelConverter
import string

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import ResNet_FeatureExtractor
from modules.semantic_vectors import Random, Zero, Seperate_Bert, Joined_Bert, Frequency
from modules.sequence_modeling import BidirectionalLSTM, TF_Encoder, ImgBert
from modules.prediction import Attention, TF_Decoder, TF_encoder_prediction, Linear_Decoder

character = config.CHARS
converter = AttnLabelConverter(character)
num_classes = len(converter.character)

batch_max_length = config.MAX_TEXT_LENGTH
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

        # Spatial Transformation Network
        self.Transformation = TPS_SpatialTransformerNetwork(
                F=num_fiducial, I_size=(imgH, imgW), I_r_size=(imgH, imgW), I_channel_num=input_channel)


        # Feature Extraction
        self.FeatureExtraction = ResNet_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1


        # Semantic Vectors
        if config.SEMANTIC_FORM == 'ZERO':
            self.get_semantic_vectors = Zero()
        elif config.SEMANTIC_FORM == 'RAND':
            self.get_semantic_vectors = Random()
        elif config.SEMANTIC_FORM == 'BERT':
            self.get_semantic_vectors = Joined_Bert(output_dim=512)#Seperate_Bert_Initilisation()
        elif config.SEMANTIC_FORM == 'FREQ':
            self.get_semantic_vectors = Frequency()
        else:
            raise Exception("Model.py Semantic Vector Form Error: '" + config.SEMANTIC_VECTOR_FORM + "' not recognized")


        # Encoder
        if config.ENCODER == 'LSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
                BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        elif config.ENCODER == 'Transformer':
            self.SequenceModeling = TF_Encoder()
        else:
            raise Exception("Model.py Encoder Error: '" + config.ENCODER + "' not recognized")

        #self.SequenceModeling = ImgBert()

        self.SequenceModeling_output = hidden_size
        

        # Decoder
        if config.DECODER == "LSTM":
            self.Prediction = Attention(256, 256, num_classes)
        elif config.DECODER == "Transformer":
            self.Prediction = TF_Decoder(512, num_classes, embed_dim=config.EMBED_DIM)
        elif config.DECODER == "Linear":
            self.Prediction = Linear_Decoder(num_classes)
        else:
            raise Exception("Model.py Decoder Error: '" + config.DECODER + "' not recognized")

    #def forward(self, input, text, scene_semantic, overlap_semantic, is_train=True):
    def forward(self, input, text, scene, overlap, is_train=True):

        # Transformation
        input = self.Transformation(input)
        
        # Feature Extraction
        visual_features = self.FeatureExtraction(input)
        visual_features = self.AdaptiveAvgPool(visual_features.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_features = visual_features.squeeze(3)

        # Semantic Vectors
        overlap, scene = self.get_semantic_vectors(overlap, scene)

        # Encode
        if config.ENCODER == 'Transformer':
            encoded_features = self.SequenceModeling(col_feats=visual_features, overlap=overlap, scene=scene, is_train=is_train)
        elif config.ENCODER == 'LSTM':
            encoded_features = self.SequenceModeling(visual_features)

        # Decode
        prediction = self.Prediction(encoded_features.contiguous(), text=text, overlap=overlap, scene=scene, is_train=is_train)

        return prediction

def get_model(saved_model=None):

    model = Model()
    model = torch.nn.DataParallel(model, device_ids=config.DEVICE_IDS)
    model = model.to(config.PRIMARY_DEVICE)

    del_keys = ['module.SequenceModeling.0.rnn.weight_ih_l0', 'module.SequenceModeling.0.rnn.weight_hh_l0', 'module.SequenceModeling.0.rnn.bias_ih_l0', 'module.SequenceModeling.0.rnn.bias_hh_l0', 'module.SequenceModeling.0.rnn.weight_ih_l0_reverse', 'module.SequenceModeling.0.rnn.weight_hh_l0_reverse', 'module.SequenceModeling.0.rnn.bias_ih_l0_reverse', 'module.SequenceModeling.0.rnn.bias_hh_l0_reverse', 'module.SequenceModeling.0.linear.weight', 'module.SequenceModeling.0.linear.bias', 'module.SequenceModeling.1.rnn.weight_ih_l0', 'module.SequenceModeling.1.rnn.weight_hh_l0', 'module.SequenceModeling.1.rnn.bias_ih_l0', 'module.SequenceModeling.1.rnn.bias_hh_l0', 'module.SequenceModeling.1.rnn.weight_ih_l0_reverse', 'module.SequenceModeling.1.rnn.weight_hh_l0_reverse', 'module.SequenceModeling.1.rnn.bias_ih_l0_reverse', 'module.SequenceModeling.1.rnn.bias_hh_l0_reverse', 'module.SequenceModeling.1.linear.weight', 'module.SequenceModeling.1.linear.bias', 'module.Prediction.attention_cell.i2h.weight', 'module.Prediction.attention_cell.h2h.weight', 'module.Prediction.attention_cell.h2h.bias', 'module.Prediction.attention_cell.score.weight', 'module.Prediction.attention_cell.rnn.weight_ih', 'module.Prediction.attention_cell.rnn.weight_hh', 'module.Prediction.attention_cell.rnn.bias_ih', 'module.Prediction.attention_cell.rnn.bias_hh', 'module.Prediction.generator.weight', 'module.Prediction.generator.bias']

    if saved_model:
        pretrained_dict = torch.load(saved_model)

        # for k in del_keys:
        #     if k in pretrained_dict.keys():
        #         del pretrained_dict[k]


        model.load_state_dict(pretrained_dict, strict=True)

    return model