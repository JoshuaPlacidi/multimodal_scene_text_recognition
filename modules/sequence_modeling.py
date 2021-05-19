import torch
import torch.nn as nn

import config

from collections import OrderedDict
import math

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, col_feats): #, overlap, scene, is_train)
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        #print('RUN')
        self.rnn.flatten_parameters()

        recurrent, _ = self.rnn(col_feats)

        #recurrent, _ = self.rnn(input, (overlap, scene))  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size

        return output

from transformers import BertConfig, EncoderDecoderConfig, EncoderDecoderModel

class ImgBert(nn.Module):
    def __init__(self):
        super(ImgBert, self).__init__()
        self.encoder_config = self.get_encoder_config()
        self.decoder_config = self.get_decoder_config()
        self.config = EncoderDecoderConfig.from_encoder_decoder_configs(self.encoder_config, self.decoder_config)
        self.model = EncoderDecoderModel(config = self.config).from_pretrained('bert-base-uncased','bert-base-uncased')

    def get_encoder_config(self):
        config = BertConfig()
        return config


    def get_decoder_config(self):
        config = BertConfig()
        config.is_decoder = True
        config.add_cross_attention = True
        return config



    def forward(self, col_feats, overlap, scene, is_train):
        print(col_feats.shape, overlap.shape)
        return None
import time

import pandas as pd
class TF_Encoder(nn.Module):
    def __init__(self):
        super(TF_Encoder, self).__init__()
        self.pos_encoder = PositionalEncoding(config.HIDDEN_DIM)
        self.encoder_layer = TransformerEncoderLayer(d_model=config.HIDDEN_DIM, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.layer_norm = nn.LayerNorm(config.HIDDEN_DIM)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6, norm=self.layer_norm)

        # self.overlap_mlp = MLP(input_size=(config.HIDDEN_DIM + config.EMBED_DIM), hidden_size=(config.HIDDEN_DIM + config.EMBED_DIM), num_classes=1, num_layers=3)
        # self.scene_mlp = MLP(input_size=(config.HIDDEN_DIM + config.EMBED_DIM), hidden_size=(config.HIDDEN_DIM + config.EMBED_DIM), num_classes=1, num_layers=3)

        # self.combine_mlp = MLP(input_size=(config.HIDDEN_DIM + (2*config.EMBED_DIM)), hidden_size= (config.HIDDEN_DIM + (2*config.EMBED_DIM)), num_classes=config.HIDDEN_DIM, num_layers=3)

    # def get_relevant_semantic(self, col_feats, sem_vec, mask=None, is_train=False, source='overlap'):
    #     #print('Start: ', col_feats.shape, sem_vec.shape)
    #     sem_seq_len = sem_vec.shape[1]      # Number of objects
    #     col_seq_len = col_feats.shape[1]    # Number of visual cols

    #     sem_seq = sem_vec.unsqueeze(1).repeat(1, col_seq_len, 1, 1)
    #     col_seq = col_feats.unsqueeze(2).repeat(1, 1, sem_seq_len, 1)
    #     # Reshape both tensors to [batch, col_seq_len, sem_seq_len, feats]

    #     #print('Cat:  ', sem_seq.shape, col_seq.shape)

    #     col_and_sem = torch.cat((col_seq, sem_seq), dim=3)

    #     if source == 'overlap':
    #         scores = self.overlap_mlp(col_and_sem)
    #     elif source == 'scene':
    #         scores = self.scene_mlp(col_and_sem)
    #     scores = nn.functional.softmax(scores, dim=2)
        
    # #     scores = self.mlp(col_and_sem)
    # #     mask = mask.unsqueeze(1).repeat(1,26,1).unsqueeze(-1)
    # #    # print(scores[0,0])
    # #     if str(sem_vec.device)[-1] == config.PRIMARY_DEVICE[-1]:
    # #         print('MLP Scores:', scores[0,0][:5])
    # #         print('Mask:', mask[0,0][:5])
    # #     scores = scores * mask
    # #     #print(mask[0,0])
    # #     #print(scores[0,0])
    # #     if str(sem_vec.device)[-1] == config.PRIMARY_DEVICE[-1]:
    # #         print('Mask + Scores:',scores[0,0][:5])
    # #     scores = nn.functional.softmax(scores, dim=2)
    # #     # if str(sem_vec.device)[-1] == config.PRIMARY_DEVICE[-1]:
    # #     #     print('Softmax:', scores[0,0][:5])
    # #     # time.sleep(15)


    #     relevant_sem = sem_seq * scores
    #     relevant_sem = torch.sum(relevant_sem, dim=2)

    #     # if not is_train and str(sem_vec.device)[-1] == config.PRIMARY_DEVICE[-1]: # if running validation and is on primary device (to prevent multiple print outs if more than 1 gpu is being used)
    #     #     rel_list = scores[0].squeeze(-1).cpu().numpy().tolist()
    #     #     #t_list = scores[0].squeeze(-1).cpu().numpy().tolist()
    #     #     cols = [i for i in range(min(sem_seq_len,25))]
    #     #     df = pd.DataFrame(columns=cols)
    #     #     for i in range(config.MAX_TEXT_LENGTH+1):
    #     #         obj_scores = [(round(j*100,2)) for j in rel_list[i]][:min(sem_seq_len,25)]
    #     #         #t = [j for j in t_list[i]][:15]
    #     #         #obj_scores = [i if i > 1 else 0 for i in obj_scores]
    #     #         df = df.append(pd.Series(obj_scores, index=cols), ignore_index=True)
    #     #         #df = df.append(pd.Series(t, index=cols), ignore_index=True)
    #     #     print(df)

    #     #print('End:  ', relevant_sem.shape)

    #     return relevant_sem


    def forward(self, col_feats, overlap, scene, overlap_mask=None, scene_mask=None, is_train=False):
        
        if False:

            rel_overlap = self.get_relevant_semantic(col_feats, overlap, overlap_mask, is_train=is_train, source='overlap')
            rel_scene = self.get_relevant_semantic(col_feats, scene, scene_mask, is_train=is_train, source='scene')
            #rel_scene = self.get_relevant_semantic(col_feats, scene, scene_mask, is_train)

            #print(col_feats.shape, rel_overlap.shape, rel_scene.shape)

            combined = torch.cat((col_feats, rel_overlap, rel_scene), dim=2)
            #combined = self.lin_reduce(combined)
            # combined = col_feats
            # combined[:,:,:config.EMBED_DIM] = combined[:,:,:config.EMBED_DIM] + rel_overlap
            #combined = col_feats
            input = self.combine_mlp(combined)
        else:
            input = col_feats

        input = input.permute(1,0,2)
        input = self.pos_encoder(input)

        output = self.encoder(input)
        output = output.permute(1,0,2)
        
        return output

# class TF_Encoder(nn.Module):
#     def __init__(self):
#         super(TF_Encoder, self).__init__()
#         self.pos_encoder = PositionalEncoding1D(512)
#         self.pos_encoder = PositionalEncoding(512)
#         self.encoder_layer = TransformerEncoderLayer(d_model=(512), nhead=8, dim_feedforward=2048, dropout=0.1)
#         self.layer_norm = nn.LayerNorm(512)
#         self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6, norm=self.layer_norm)

#         #self.mlp = MLP(input_size=(512), hidden_size=512, num_classes=1, num_layers=3)
#         #self.to_hid = nn.Linear((512+64), 512)

#     def forward(self, col_feats, overlap, scene, is_train):
#         # overlap = overlap.unsqueeze(1).repeat(1,26,1,1)
#         # col_features = col_feats.unsqueeze(2).repeat(1,1,20,1)
#         # col_and_overlap = torch.cat((col_features, overlap), dim=3)

#         # scores = self.mlp(col_and_overlap)
#         # scores = nn.functional.softmax(scores, dim=2)

#         # relevant_overlap = overlap * scores
#         # relevant_overlap = torch.sum(relevant_overlap, dim=2)

#         # if not is_train and str(overlap.device)[-1] == config.PRIMARY_DEVICE[-1]: # if running validation and is on primary device (to prevent multiple print outs if more than 1 gpu is being used)
#         #     rel_list = scores[0].squeeze(-1).cpu().numpy().tolist()
#         #     for i in range(config.MAX_TEXT_LENGTH+1):
#         #         print([round(j,2) for j in rel_list[i]])

#         # relevant_overlap = relevance_scores * overlap
#         # relevant_overlap = torch.sum(relevant_overlap, dim=2)
#         # combined = torch.cat((visual_features, relevant_overlap), dim=2)
#         #print(scene.shape)
#         # col_feats = col_feats.permute(1,0,2)
#         # col_feats = self.pos_encoder(col_feats)
#         # col_feats = col_feats.permute(1,0,2)

#         combined = col_feats#torch.cat((col_feats, relevant_overlap), dim=2)
#         #print('Combined: ', combined.shape)
#         combined = combined.permute(1,0,2)
#         combined = self.pos_encoder(combined)

#         output = self.encoder(combined)
#         output = output.permute(1,0,2)
#         #output = self.to_hid(output)
        
#         return output


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.channels = channels
        inv_freq = 1. / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        #print('IN:',tensor.shape)
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        _, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x,self.channels),device=tensor.device).type(tensor.type())
        emb[:,:self.channels] = emb_x
        #print('OUT:',emb[None,:,:orig_ch].shape)
        return emb[None,:,:orig_ch]

import math
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=26):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = torch.nn.functional.relu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = torch.nn.functional.relu
        super(TransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src, src_mask = None, src_key_padding_mask = None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = self.norm1(src)
        src = src + self.dropout1(src2)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src)
        src = src + self.dropout2(src2)
        return src




import math


class MLP(nn.Module):
    """A simple MLP.
    """

    def __init__(self, input_size, hidden_size, num_classes,
                 num_layers=1, dropout_p=0.0):
        """Constructor for MLP.
        Args:
            input_size: The number of input dimensions.
            hidden_size: The number of hidden dimensions for each layer.
            num_classes: The size of the output.
            num_layers: The number of hidden layers.
            dropout_p: Dropout probability.
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        layers = []
        for i in range(num_layers):
            idim = hidden_size
            odim = hidden_size
            if i == 0:
                idim = input_size
            if i == num_layers-1:
                odim = num_classes
            fc = nn.Linear(idim, odim)
            fc.weight.data.normal_(0.0,  math.sqrt(2. / idim))
            fc.bias.data.fill_(0)
            layers.append(('fc'+str(i), fc))
            if i != num_layers-1:
                layers.append(('relu'+str(i), nn.ReLU()))
                layers.append(('dropout'+str(i), nn.Dropout(p=dropout_p)))
        self.layers = nn.Sequential(OrderedDict(layers))

    def params_to_train(self):
        return self.layers.parameters()

    def forward(self, x):
        """Propagate through all the hidden layers.
        Args:
            x: Input of self.input_size dimensions.
        """
        out = self.layers(x)
        return out