import torch
import torch.nn as nn

class varLSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(varLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

        self.fc =  nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, input_feature_tuple):
        # input_seq: (seq_len, batch, input_size)
    