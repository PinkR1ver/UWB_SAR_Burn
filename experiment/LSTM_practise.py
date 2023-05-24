import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq):
        # input_seq: (seq_len, batch, input_size)
        # lstm_out: (seq_len, batch, hidden_size)

        lstm_out, (hidden_state, cell_state) = self.lstm(input_seq)

        lstm_out = self.fc(lstm_out)

        return lstm_out, hidden_state, cell_state
    

if __name__ == '__main__':
    '''
    seq = np.linspace(0, 3801, 3801)
    h = torch.randn(1, 1, 64)
    c = torch.randn(1, 1, 64)

    lstm = LSTM(1, 1, 64, 1)

    input = torch.Tensor(seq).view(len(seq), 1, -1)

    lstm_out, hidden_state, cell_state = lstm(input)
    lstm_out = torch.squeeze(lstm_out)

    print(lstm_out.shape)
    print(hidden_state.shape)
    print(cell_state.shape)
    '''


    rnn = nn.LSTM(10, 5, 2)
    input = torch.randn(5, 1, 10)
    h0 = torch.randn(2, 1, 5)
    c0 = torch.randn(2, 1, 5)
    output, (hn, cn) = rnn(input, (h0, c0))

    print(output.shape)
    print(hn.shape)
    print(cn.shape)

    print(h0)
    print(hn)