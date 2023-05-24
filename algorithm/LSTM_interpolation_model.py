# This file is very fragile

import torch
import torch.nn as nn

class varLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, corresponding_feature_size, dense_node_size, num_layers):
        super(varLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.corresponding_feature_size = corresponding_feature_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

        self.dense1 = nn.ModuleList([
            nn.Linear(corresponding_feature_size, dense_node_size[0])
        ] + [
            nn.Linear(dense_node_size[i], dense_node_size[i+1]) for i in range(len(dense_node_size) - 1)
        ] + [
            nn.Linear(dense_node_size[-1], hidden_size)
        ])

        self.dense2 = nn.ModuleList([
            nn.Linear(corresponding_feature_size, dense_node_size[0])
        ] + [
            nn.Linear(dense_node_size[i], dense_node_size[i+1]) for i in range(len(dense_node_size) - 1)
        ] + [
            nn.Linear(dense_node_size[-1], hidden_size)
        ])


        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_dim, input_seq, input_feature, num_loops):
        # input_seq: (seq_len, batch, input_size)
        # input_feature: (feature_size, batch, layers)

        ans = torch.zeros(input_dim, input_seq.shape[1], input_seq.shape[2], self.output_size)
        
        for i in range(input_dim):
            h_0 = input_feature[i]
            h_0 = h_0.permute(2, 1, 0) # change to (layers, batch, feature_size)

            for module in self.dense1:
                h_0 = module(h_0)
            
            h_0 = h_0.permute(1, 0, 2) # change to (batch, layers, feature_size)

            c_0 = input_feature[i]
            c_0 = c_0.permute(2, 1, 0) # change to (layers, batch, feature_size)

            for module in self.dense2:
                c_0 = module(c_0)
            
            c_0 = c_0.permute(1, 0, 2) # change to (batch, layers, feature_size)

            for j in range(num_loops):
                
                input = input_seq[i]
                lstm_out, (h_n, c_n) = self.lstm(input, (h_0, c_0))

                h_0 = h_n
                c_0 = c_n

            lstm_out = self.fc(lstm_out)

            ans[i] = lstm_out
        
        return ans
        

if __name__ == '__main__':
    model = varLSTM(1, 128, 1, 2, [32, 64, 64], 4)

    input_seq = torch.randn(4, 3801, 4, 1)
    input_feature = torch.randn(4, 2, 4, 4)

    print(model)

    output = model(4, input_seq, input_feature, 4)

    print(output.shape)