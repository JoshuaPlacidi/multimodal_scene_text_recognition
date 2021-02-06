import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, _input):
        input = _input[0]
        init_cell = _input[1]
        init_hid = _input[2]
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()

        # recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        # new_c, new_h = init_cell, init_hid

        recurrent, (new_h, new_c) = self.rnn(input, (init_hid, init_cell))  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        
        output = self.linear(recurrent)  # batch_size x T x output_size
        return [output, new_c, new_h]