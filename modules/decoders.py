import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import pandas as pd

import config

class Attention(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes):
        super(Attention, self).__init__()
        self.attention_cell = AttentionCell(input_size, hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.generator = nn.Linear(hidden_size, num_classes)

    def _char_to_onehot(self, input_char, onehot_dim=38):
        input_char = input_char.unsqueeze(1)
        batch_size = input_char.size(0)
        one_hot = torch.FloatTensor(batch_size, onehot_dim).zero_().to(input_char.device)
        one_hot = one_hot.scatter_(1, input_char, 1)
        return one_hot

    def forward(self, encoder_output, text, semantics, is_train):
        """
        input:
            batch_H : contextual_feature H = hidden state of encoder. [batch_size x num_steps x contextual_feature_channels]
            text : the text-index of each image. [batch_size x (max_length+1)]. +1 for [GO] token. text[:, 0] = [GO].
        output: probability distribution at each step [batch_size x num_steps x num_classes]
        """
        batch_size = encoder_output.size(0)
        num_steps = config.MAX_TEXT_LENGTH + 1  # +1 for [s] at end of sentence.

        output_hiddens = torch.FloatTensor(batch_size, num_steps, self.hidden_size).fill_(0).to(encoder_output.device)
        hidden = (torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(encoder_output.device), 
        torch.FloatTensor(batch_size, self.hidden_size).fill_(0).to(encoder_output.device))
        #hidden = (overlap, overlap)
        
        if is_train:
            for i in range(num_steps):
                # one-hot vectors for a i-th char. in a batch
                char_onehots = self._char_to_onehot(text[:, i], onehot_dim=self.num_classes)
                # hidden : decoder's hidden s_{t-1}, batch_H : encoder's hidden H, char_onehots : one-hot(y_{t-1})
                #print('hidden[0]',hidden[0].shape,'hidden[1]',hidden[1].shape,' | batch_H', batch_H.shape)
                #hidden[0], hidden[1] = init_hidd, init_hidd
                hidden, alpha = self.attention_cell(hidden, encoder_output, char_onehots, semantics=semantics)
                output_hiddens[:, i, :] = hidden[0]  # LSTM hidden index (0: hidden, 1: Cell)
            probs = self.generator(output_hiddens)

        else:
            targets = torch.LongTensor(batch_size).fill_(0).to(encoder_output.device)  # [GO] token
            probs = torch.FloatTensor(batch_size, num_steps, self.num_classes).fill_(0).to(encoder_output.device)

            for i in range(num_steps):
                char_onehots = self._char_to_onehot(targets, onehot_dim=self.num_classes)
                hidden, alpha = self.attention_cell(hidden, encoder_output, char_onehots, semantics=semantics)
                probs_step = self.generator(hidden[0])
                probs[:, i, :] = probs_step
                _, next_input = probs_step.max(1)
                targets = next_input

        return probs  # batch_size x num_steps x num_classes


class AttentionCell(nn.Module):

    def __init__(self, input_size, hidden_size, num_embeddings):
        super(AttentionCell, self).__init__()
        self.i2h = nn.Linear(input_size, hidden_size, bias=False)
        self.h2h = nn.Linear(hidden_size, hidden_size)  # either i2i or h2h should have bias
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.rnn = nn.LSTMCell(input_size + num_embeddings, hidden_size)
        self.hidden_size = hidden_size

        #self.fc1 = nn.Linear(608, 352)

    def forward(self, prev_hidden, batch_H, char_onehots, semantics):
        # [batch_size x num_encoder_step x num_channel] -> [batch_size x num_encoder_step x hidden_size]
        batch_H_proj = self.i2h(batch_H)
        prev_hidden_proj = self.h2h(prev_hidden[0]).unsqueeze(1)
        #overlap = overlap.unsqueeze(1)
        #print(batch_H_proj.shape, h1.shape)
        e = self.score(torch.tanh(batch_H_proj + prev_hidden_proj))  # batch_size x num_encoder_step * 1

        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.permute(0, 2, 1), batch_H).squeeze(1)  # batch_size x num_channel

        concat_context = torch.cat([context, char_onehots], 1)  # batch_size x (num_channel + num_embedding)
       
       # concat_context = torch.cat([concat_context], 1) 
       # concat_context = self.fc1(concat_context)
       # print(prev_hidden[0].shape)

        rnn_hidd = prev_hidden[0]
        rnn_cell = prev_hidden[1]

        cur_hidden = self.rnn(concat_context, (rnn_hidd, rnn_cell))
        #print(len(cur_hidden), cur_hidden[0].shape)
        return cur_hidden, alpha

class TF_Decoder(nn.Module):
    '''
    Transformer decoder class
    input: encoder output [batch, seq, hid dim], text (targets), overlap and scene [batch, num objs, embed dim], is_train bool
    return: sequence of probability distribution over num_classes [batch, seq, num_classes]
    '''
    def __init__(self, num_classes):
        super(TF_Decoder, self).__init__()
        self.decoder_layer = TransformerDecoderLayer(d_model=config.EMBED_DIM, nhead=8, dim_feedforward=2048, dropout=0.1)
        self.layer_norm = nn.LayerNorm(config.EMBED_DIM)
        self.decoder = TransformerDecoder(self.decoder_layer, num_layers=6, norm=self.layer_norm)

        self.pos_encoder = PositionalEncoding(config.EMBED_DIM)

        self.hid_to_emb = nn.Linear(config.HIDDEN_DIM, config.EMBED_DIM)
        self.emb = nn.Embedding(num_classes, config.EMBED_DIM)
        self.emb_to_classes = nn.Linear(config.EMBED_DIM, num_classes)

        self.num_classes = num_classes

        if config.PRE_DECODER_MLP:
            self.relevant_mlp = MLP(input_size=(config.EMBED_DIM*2), hidden_size=config.EMBED_DIM, num_classes=1, num_layers=3)
            self.combine_mlp = MLP(input_size=(config.EMBED_DIM*2), hidden_size=config.EMBED_DIM, num_classes=config.EMBED_DIM, num_layers=2)
        
        if config.CLS_DECODER_INIT:
            self.sem_cls = True
            self.sem_cls_mlp = MLP(input_size=(config.EMBED_DIM*2), hidden_size=config.EMBED_DIM, num_classes=1, num_layers=3)
        else:
            self.sem_cls = False

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def get_relevant_semantic(self, feats, sem_vec, mlp, is_train):
        sem_seq_len = sem_vec.shape[1]  # Number of objects
        col_seq_len = feats.shape[1]    # Number of visual cols

        sem_seq = sem_vec.unsqueeze(1).repeat(1, col_seq_len, 1, 1)
        col_seq = feats.unsqueeze(2).repeat(1, 1, sem_seq_len, 1)
        # Reshape both tensors to [batch, col_seq_len, sem_seq_len, feats]

        col_and_sem = torch.cat((col_seq, sem_seq), dim=3)

        scores = mlp(col_and_sem)
        scores = nn.functional.softmax(scores, dim=2)
        
        relevant_sem = sem_seq * scores
        relevant_sem = torch.sum(relevant_sem, dim=2)

        # Prints attention matrix
        if config.PRINT_ATTENTION_SCORES and not is_train and str(sem_vec.device)[-1] == config.PRIMARY_DEVICE[-1]:
            self.print_attention_scores(scores=scores, sem_seq_len=sem_seq_len)

        return relevant_sem

    def print_attention_scores(self, scores, sem_seq_len):
        scores = scores[0].squeeze(-1).cpu().numpy().tolist()
        cols = [i for i in range(min(sem_seq_len,25))]
        df = pd.DataFrame(columns=cols)

        for i in range(config.MAX_TEXT_LENGTH+1):
            obj_scores = [(round(j*100,2)) for j in scores[i]][:min(sem_seq_len,25)]
            df = df.append(pd.Series(obj_scores, index=cols), ignore_index=True)
        print(df)

    def get_semantic_cls(self, feats, sem_vec, is_train):
        #print(feats.shape, sem_vec.shape)
        relevant_sem = self.get_relevant_semantic(feats, sem_vec, self.sem_cls_mlp, is_train)
        weighted_sem = nn.functional.softmax(relevant_sem, dim=1)

        semantic_cls = torch.sum(weighted_sem, dim=1)
        return semantic_cls

        

    def forward(self, encoder_output, text, semantics, is_train):
        # Map encoder_ouput dim to embed dim
        memory = self.hid_to_emb(encoder_output)

        if config.PRE_DECODER_MLP:
            relevant_semantic = self.get_relevant_semantic(memory, semantics, self.relevant_mlp, is_train)
            combined = torch.cat((memory, relevant_semantic), dim=2)
            memory = memory + self.combine_mlp(combined)


        memory = memory.permute(1,0,2)

        if is_train: # Training

            # convert targets from [batch, seq, feats] -> [seq, batch, feats] and apply embedding and position encoding
            targets = text[:memory.shape[1],:]
            targets = targets.permute(1,0)
            
            emb_targets = self.emb(targets)

            if self.sem_cls:
                sem_cls = self.get_semantic_cls(memory.permute(1,0,2), semantics, is_train)
                emb_targets[0,:,:] = sem_cls

            emb_targets = self.pos_encoder(emb_targets)

            # generate target mask and pass to decoder
            target_mask = self._generate_square_subsequent_mask(config.MAX_TEXT_LENGTH+1).to(encoder_output.device)
            output = self.decoder(tgt=emb_targets, memory=memory, semantics=semantics, tgt_mask=target_mask, is_train=is_train)

            # map embeding dim to number of classes
            output = self.emb_to_classes(output)

        else: # Inference

            # Declare targets and output as zero tensors of output shape
            targets = torch.zeros(memory.shape[1], config.MAX_TEXT_LENGTH+1).to(encoder_output.device)
            targets = targets.permute(1,0)

            output = torch.zeros(config.MAX_TEXT_LENGTH, memory.shape[1], self.num_classes).to(encoder_output.device)

            for t in range(config.MAX_TEXT_LENGTH):

                target_mask = self._generate_square_subsequent_mask(t+1).to(encoder_output.device)
                
                # convert targets into embeddings and apply positional encoding
                emb_targets = self.emb(targets.long())

                if self.sem_cls:
                    sem_cls = self.get_semantic_cls(memory.permute(1,0,2), semantics, is_train)
                    emb_targets[0,:,:] = sem_cls

                emb_targets = self.pos_encoder(emb_targets)
                
                # pass embed targets and encoder memory to decoder
                t_output = self.decoder(tgt=emb_targets[:t+1], memory=memory, semantics=semantics, tgt_mask=target_mask, is_train=is_train)

                # map embeding dim to number of classes
                t_output = self.emb_to_classes(t_output)

                # take index class with max probability and append to targets and output sequence
                _, char_index = t_output[-1].max(1)
                targets[t+1,:] = char_index
                output[t,:] = t_output[t]

        output = output.permute(1,0,2)

        return output


class Linear_Decoder(nn.Module):
    '''
    Linear layer with no bias
    input: encoder output sequence
    return: probability distribution over num_classes of shape [batch, seq, num_classes]
    '''
    def __init__(self, num_classes):
        super(Linear_Decoder, self).__init__()
        self.linear_decoder = nn.Linear(config.HIDDEN_DIM, num_classes)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.linear_decoder.bias.data.zero_()
        self.linear_decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, encoder_output, text, semantics, is_train):
        output = self.linear_decoder(encoder_output)
        return output



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



class TransformerDecoder(nn.Module):
    '''
    Pytorch implementation of transformer decoder
    '''
    __constants__ = ['norm']
    def __init__(self, decoder_layer, num_layers, norm=None):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.modules.transformer._get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, semantics, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, is_train=False):
            output = tgt

            for mod in self.layers:
                output = mod(output, memory, semantics=semantics, 
                            tgt_mask=tgt_mask, memory_mask=memory_mask,
                            tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=memory_key_padding_mask,
                            is_train=is_train)

            if self.norm is not None:
                output = self.norm(output)

            return output

class TransformerDecoderLayer(nn.Module):
    '''
    Pytorch implementation of transformer decoder layer
    '''
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.semantic_to_emb = nn.Linear(config.HIDDEN_DIM, config.EMBED_DIM)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = F.relu

        if config.MULTIHEAD_PRE_TARGET:
            self.multihead_pre_target = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.norm_pre_target = nn.LayerNorm(d_model)
            self.dropout_pre_target = nn.Dropout(dropout)

            self.relevant_mlp_pre_target = MLP(input_size=(config.EMBED_DIM*2), hidden_size=config.EMBED_DIM, num_classes=1, num_layers=3)
            #self.combine_mlp_pre_target = MLP(input_size=(config.EMBED_DIM*2), hidden_size=config.EMBED_DIM, num_classes=config.EMBED_DIM, num_layers=2)

        if config.MULTIHEAD_PRE_MEMORY:
            self.multihead_pre_memory = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.norm_pre_memory = nn.LayerNorm(d_model)
            self.dropout_pre_memory = nn.Dropout(dropout)

            self.relevant_mlp_pre_memory = MLP(input_size=(config.EMBED_DIM*2), hidden_size=config.EMBED_DIM, num_classes=1, num_layers=3)
            #self.combine_mlp_pre_memory = MLP(input_size=(config.EMBED_DIM*2), hidden_size=config.EMBED_DIM, num_classes=config.EMBED_DIM, num_layers=2)

        if config.MULTIHEAD_POST_MEMORY:
            self.multihead_post_memory = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
            self.norm_post_memory = nn.LayerNorm(d_model)
            self.dropout_post_memory = nn.Dropout(dropout)

            self.relevant_mlp_post_memory = MLP(input_size=(config.EMBED_DIM*2), hidden_size=config.EMBED_DIM, num_classes=1, num_layers=3)
            #self.combine_mlp_post_memory = MLP(input_size=(config.EMBED_DIM*2), hidden_size=config.EMBED_DIM, num_classes=config.EMBED_DIM, num_layers=2)
            

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)

    def get_relevant_semantic(self, feats, sem_vec, relevant_mlp, is_train=True):
        # print(feats.shape, sem_vec.shape)
        sem_seq_len = sem_vec.shape[1]  # Number of objects
        col_seq_len = feats.shape[1]    # Number of visual cols

        sem_seq = sem_vec.unsqueeze(1).repeat(1, col_seq_len, 1, 1)
        col_seq = feats.unsqueeze(2).repeat(1, 1, sem_seq_len, 1)
        # Reshape both tensors to [batch, col_seq_len, sem_seq_len, feats]

        col_and_sem = torch.cat((col_seq, sem_seq), dim=3)

        scores = relevant_mlp(col_and_sem)
        scores = nn.functional.softmax(scores, dim=2)
        
        relevant_sem = sem_seq * scores
        relevant_sem = torch.sum(relevant_sem, dim=2)

        # if config.PRINT_ATTENTION_SCORES and not is_train and str(sem_vec.device)[-1] == config.PRIMARY_DEVICE[-1]:
        #     self.print_attention_scores(scores=scores, sem_seq_len=sem_seq_len)

        return relevant_sem

    # def print_attention_scores(self, scores, sem_seq_len):
    #     scores = scores[0].squeeze(-1).cpu().numpy().tolist()
    #     print(scores)
    #     cols = [i for i in range(min(sem_seq_len,25))]
    #     df = pd.DataFrame(columns=cols)

    #     for i in range(len(scores)):
    #         obj_scores = [(round(j*100,2)) for j in scores[i]][:min(sem_seq_len,25)]
    #         df = df.append(pd.Series(obj_scores, index=cols), ignore_index=True)
    #     print(df)

    def forward(self, tgt, memory, semantics, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, is_train=False):
        
        if config.MULTIHEAD_PRE_TARGET:

            semantics = self.get_relevant_semantic(tgt.permute(1,0,2), semantics, self.relevant_mlp_pre_target, is_train)
            tgt2 = self.multihead_pre_target(tgt, semantics, semantics)[0]
            tgt = tgt + self.dropout_pre_target(tgt2)
            tgt = self.dropout_pre_target(tgt)            
        
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        if config.MULTIHEAD_PRE_MEMORY:
            semantics = self.get_relevant_semantic(tgt.permute(1,0,2), semantics, self.relevant_mlp_pre_memory, is_train)
            tgt2 = self.multihead_pre_memory(tgt, semantics, semantics)[0]
            tgt = tgt + self.dropout_pre_memory(tgt2)
            tgt = self.dropout_pre_memory(tgt) 

        tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        if config.MULTIHEAD_POST_MEMORY:
            semantics = self.get_relevant_semantic(tgt.permute(1,0,2), semantics, self.relevant_mlp_post_memory, is_train)
            tgt2 = self.multihead_post_memory(tgt, semantics, semantics)[0]
            tgt = tgt + self.dropout_post_memory(tgt2)
            tgt = self.dropout_post_memory(tgt) 

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


import math
from collections import OrderedDict

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