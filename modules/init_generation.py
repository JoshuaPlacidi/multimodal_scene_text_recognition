import config
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
torch.backends.cudnn.enabled = False


from transformers import BertModel

# Return random tensors
class Random_Initilisation(nn.Module):

    def __init__(self):
        super(Random_Initilisation, self).__init__()

    def forward(self, overlap, scene):
        init_hidd = torch.rand(2, overlap.shape[0], 256).to(overlap.device)
        init_cell = torch.rand(2, overlap.shape[0], 256).to(overlap.device)
        return init_hidd, init_cell


# Retrun zero tensors
class Zero_Initilisation(nn.Module):

    def __init__(self):
        super(Zero_Initilisation, self).__init__()

    def forward(self, overlap, scene):
        init_hidd = torch.zeros(2, overlap.shape[0], 256).to(overlap.device)
        init_cell = torch.zeros(2, overlap.shape[0], 256).to(overlap.device)
        return init_hidd, init_cell


# Return Bert embedding initilisations
class Bert_Initilisation(nn.Module):

    def __init__(self):
        super(Bert_Initilisation, self).__init__()

        self.bert_model = BertModel.from_pretrained('bert-base-uncased')

        self.overlap_batch_norm = nn.BatchNorm1d(768)
        self.scene_batch_norm = nn.BatchNorm1d(768)

        self.fc1 = nn.Linear(768, 500)
        self.fc2 = nn.Linear(768, 500)
        self.fc3 = nn.Linear(500, 256)
        self.fc4 = nn.Linear(500, 256)

        if config.LSTM_CELL_INIT == 'COMBINED':
                self.comb_fc1 = nn.Linear(512,256)

        if config.LSTM_HIDD_INIT == 'COMBINED':
                self.comb_fc2 = nn.Linear(512,256)

    def forward(self, overlap, scene):
        # Process overlap embedding
        flat_overlap = torch.flatten(overlap, start_dim=0, end_dim=1).to(torch.int64)
        bert_overlap = self.bert_model(flat_overlap, output_hidden_states = True)
        bert_overlap = bert_overlap[2][12][:,0,:]
        bert_overlap = torch.split(bert_overlap, overlap.shape[1], dim=0)
        bert_overlap = torch.stack(bert_overlap)

        overlap = torch.sum(bert_overlap, dim=1)
        overlap = self.overlap_batch_norm(overlap)

        # Process scene embedding
        flat_scene = torch.flatten(scene, start_dim=0, end_dim=1).to(torch.int64)
        bert_scene = self.bert_model(flat_scene, output_hidden_states = True)
        bert_scene = bert_scene[2][12][:,0,:]
        bert_scene = torch.split(bert_scene, scene.shape[1], dim=0)
        bert_scene = torch.stack(bert_scene)

        scene = torch.sum(bert_scene, dim=1)
        scene = self.scene_batch_norm(scene)

        # Linear layers
        overlap = F.relu(self.fc1(overlap))
        overlap = F.relu(self.fc3(overlap))
        scene = self.fc2(scene)
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


# Return frequency embedding initilisations
class Frequency_Initilisation(nn.Module):

    def __init__(self):
        super(Frequency_Initilisation, self).__init__()
        if config.ANNOTATION_SOURCE == 'VG':
            self.fc1 = nn.Linear(1601, 800)
            self.fc2 = nn.Linear(1601, 800)
            self.fc3 = nn.Linear(800, 256)
            self.fc4 = nn.Linear(800, 256)

        elif config.ANNOTATION_SOURCE == 'COCO':
            self.fc1 = nn.Linear(90, 175)
            self.fc2 = nn.Linear(90, 175)
            self.fc3 = nn.Linear(175, 256)
            self.fc4 = nn.Linear(175, 256)

        if config.LSTM_CELL_INIT == 'COMBINED':
                self.comb_fc1 = nn.Linear(512,256)

        if config.LSTM_HIDD_INIT == 'COMBINED':
                self.comb_fc2 = nn.Linear(512,256)

    def forward(self, overlap, scene):
        # Pass through linear layers
        overlap = F.relu(self.fc1(overlap))
        overlap = self.fc3(overlap)

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

        
