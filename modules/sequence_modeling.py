import torch
import torch.nn as nn
import config

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, overlap, scene):
        input, init_hidd, init_cell = input
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()

        recurrent, _ = self.rnn(input)

        #recurrent, (new_hidd, new_cell) = self.rnn(input, (init_hidd, init_cell))  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        
        output = self.linear(recurrent)  # batch_size x T x output_size
        return (output, init_hidd, init_cell)

class TF_encoder(nn.Module):
    def __init__(self):
        super(TF_encoder, self).__init__()
        #self.pos_encoder = PositionalEncoding1D(512)
        self.pos_encoder = PositionalEncoding(512)
        self.encoder_layer = TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=2048, dropout=0.2)
        self.layer_norm = nn.LayerNorm(512)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6, norm=self.layer_norm)

        self.mlp = MLP(input_size=1024, hidden_size=768, num_classes=512, num_layers=3)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, input, overlap, scene, is_train):
        #input = input + self.pos_encoder(input)

        # input = [batch, seq, 512]
        # overlap = [batch, obj_len, 512] -> [batch, 1, 512]
        overlap = torch.sum(overlap, dim=1, keepdim=True)
        overlap = overlap.repeat(1, config.MAX_TEXT_LENGTH+1, 1)
        # combined = concat(input + overlap) -> [batch, seq, 1024]
        combined = torch.cat((input, overlap), dim=2)
        #combined = torch.matmul(input, overlap.permute(0,2,1)) # batch x seq len x obj len

        # mlp input = matmult (overlap )
        # mlp output = batch, 26, 20

        # output_mlp [batch, seq, 1]
        relevance = self.softmax(self.mlp(combined))
        relevant_overlap = relevance * overlap
        input = input + relevant_overlap
        #print(list(relevance.cpu().numpy()[0]))
        #relevance = softmax(relevance)

        # overlap_features = relevance * visual_features
        # visual + overlap
        # pos()
        # output = encoder(visual_features)



        if is_train == False and str(overlap.device)[-1] == config.PRIMARY_DEVICE[-1]:
            print(relevance[0,0])#.cpu().numpy())
            # seq = list(relevance.cpu().numpy()[0])
            # for i in range(len(seq)):
            #     l = list(seq[i])
            #     l = [round(i, 1) for i in l]
            #     print(l,end='')

        # print(overlap.device,config.PRIMARY_DEVICE,overlap.device==config.PRIMARY_DEVICE)
        # if overlap.device == config.PRIMARY_DEVICE:
        #     print(relevance[0])

        

        # overlap = overlap.unsqueeze(1).repeat(1,26,1,1)

        # # CNN 

        # relevance = relevance.unsqueeze(-1).repeat(1,1,1,512)

        # #print('rel', relevance.shape, 'o', overlap.shape)

        # # multiply relevance scalar to each of the features of each object
        
        # relevant_overlap = relevance * overlap # batch x seq_len x obj_len x 512

        # # if str(overlap.device)[-1] == config.PRIMARY_DEVICE[-1]: print(relevant_overlap[0,0,0][:10])

        # relevant_overlap = torch.sum(relevant_overlap, dim=2) # batch x seq_len x obj_len x 512

        input = input.permute(1,0,2)
        input = self.pos_encoder(input)

        #torch.cat((overlap, input), dim=0)

        output = self.encoder(input)

        #output = output.permute(1,0,2)
        #print(output.shape)

        output = output.permute(1,0,2)
        #output = output * relevant_overlap
        
        return output


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


from collections import OrderedDict
from torch import nn

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