import config
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
torch.backends.cudnn.enabled = False


from transformers import BertModel, DistilBertModel, DistilBertConfig

# Return random tensors
class Random(nn.Module):

    def __init__(self):
        super(Random, self).__init__()

    def forward(self, overlap, scene):
        overlap = torch.rand(overlap.shape[0], 256).to(overlap.device)
        scene = torch.rand(scene.shape[0], 256).to(scene.device)
        return overlap, scene


# Retrun zero tensors
class Zero(nn.Module):

    def __init__(self):
        super(Zero, self).__init__()

    def forward(self, overlap, scene):
        overlap = torch.zeros(overlap.shape[0], 256).to(overlap.device)
        scene = torch.zeros(scene.shape[0], 256).to(scene.device)
        return overlap, scene


# Return Bert embedding initilisations
#
#  NEEDS UPDATING (NOT USABLE)
#
class Seperate_Bert(nn.Module):

    def __init__(self):
        raise Exception("Seperate BERT is not implemented, use joined BERT")
        super(Seperate_Bert, self).__init__()

        self.configuration = DistilBertConfig()
        self.bert_model = DistilBertModel(self.configuration).from_pretrained('distilbert-base-uncased')

        self.calculate_overlap = False
        self.calculate_scene = False

        self.calculate_overlap = True
        self.overlap_batch_norm = nn.BatchNorm1d(768)

        self.calculate_scene = True
        self.scene_batch_norm = nn.BatchNorm1d(768)

    def forward(self, overlap, scene):
        raise Exception("Seperate BERT is not implemented, use joined BERT")
        # Process overlap embedding
        if self.calculate_overlap:
            flat_overlap = torch.flatten(overlap, start_dim=0, end_dim=1).to(torch.int64)
            bert_overlap = self.bert_model(flat_overlap, output_hidden_states = True)
            bert_overlap = bert_overlap[2][12][:,0,:]
            bert_overlap = torch.split(bert_overlap, overlap.shape[1], dim=0)
            bert_overlap = torch.stack(bert_overlap)

            overlap = torch.sum(bert_overlap, dim=1)
            overlap = self.overlap_batch_norm(overlap)

            overlap = F.relu(self.fc1(overlap))
            overlap = self.fc3(overlap)

        # Process scene embedding
        if self.calculate_scene:
            flat_scene = torch.flatten(scene, start_dim=0, end_dim=1).to(torch.int64)
            bert_scene = self.bert_model(flat_scene, output_hidden_states = True)
            bert_scene = bert_scene[2][12][:,0,:]
            bert_scene = torch.split(bert_scene, scene.shape[1], dim=0)
            bert_scene = torch.stack(bert_scene)

            scene = torch.sum(bert_scene, dim=1)
            scene = self.scene_batch_norm(scene)

            scene = F.relu(self.fc2(scene))
            scene = self.fc4(scene)

        if config.LSTM_HIDD_INIT == 'OVERLAP':
            hidd_init = overlap
        elif config.LSTM_HIDD_INIT == 'SCENE':
            hidd_init = scene
        elif config.LSTM_HIDD_INIT == 'COMBINED':
            hidd_init = self.comb_fc1(torch.cat([overlap, scene], dim=1))        
            
        if config.LSTM_CELL_INIT == 'OVERLAP':
            cell_init = overlap
        elif config.LSTM_CELL_INIT == 'SCENE':
            cell_init = scene
        elif config.LSTM_CELL_INIT == 'COMBINED':
            cell_init = self.comb_fc1(torch.cat([overlap, scene], dim=1))

        return hidd_init.repeat(2, 1, 1), cell_init.repeat(2, 1, 1)

class Joined_Bert(nn.Module):

    def __init__(self, output_dim=512):
        super(Joined_Bert, self).__init__()

        self.configuration = DistilBertConfig()
        self.bert_model = DistilBertModel(self.configuration).from_pretrained('distilbert-base-uncased')

        self.fc_o = nn.Linear(768, output_dim)
        self.fc_s = nn.Linear(768, output_dim)

    def forward(self, overlap, scene):
        # Process overlap embedding
        overlap_bert = self.bert_model(overlap)[0]
        overlap_emb = self.fc_o(overlap_bert)

        # Process scene embedding
        scene_bert = self.bert_model(scene)[0]
        scene = self.fc_s(scene_bert)

        return overlap_emb, scene


# Return frequency embedding initilisations
class Frequency(nn.Module):
    def __init__(self, output_dim=512):
        super(Frequency, self).__init__()
        self.fc_o = nn.Linear(config.EMBED_DIM, output_dim)
        self.fc_s = nn.Linear(config.EMBED_DIM, output_dim)

        self.embed_ = nn.Embedding(1489, config.EMBED_DIM)

    def forward(self, overlap, scene):
        # Embed object indexs
        overlap = self.embed_(overlap.long())
        scene = self.embed_(scene.long())

        # # Pass through linear layers
        # overlap = self.fc_o(overlap)
        # overlap = F.relu(overlap)

        # scene = self.fc_s(scene)
        # scene = F.relu(scene)

        return overlap, scene

        
