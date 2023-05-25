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
        # self.merge_heads = merge_heads
        # self.merge_hidden_size = merge_hidden_size
        # self.merge_encoder_layers = merge_encoder_layers
        # self.merge_decoder_layers = merge_decoder_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

        layers = []
        for i in range(len(dense_node_size)):
            if i == 0:
                layers.append(nn.Linear(corresponding_feature_size, dense_node_size[i]))
            else:
                layers.append(nn.Linear(dense_node_size[i-1], dense_node_size[i]))
            
            layers.append(nn.LayerNorm(dense_node_size[i]))  # 归一化层
            layers.append(nn.ReLU())  # 激活函数层
            layers.append(nn.Dropout(0.5))  # 丢弃层
        
        layers.append(nn.Linear(dense_node_size[-1], hidden_size))

        self.dense1 = nn.Sequential(*layers)
        self.dense2 = nn.Sequential(*layers)

        self.fc = nn.Linear(hidden_size, output_size)

        self.merge = nn.Linear(4, output_size)

        # self.merge = nn.Transformer(d_model=self.merge_hidden_size, nhead=self.merge_heads, num_encoder_layers=self.merge_encoder_layers, num_decoder_layers=self.merge_decoder_layers)

        if torch.cuda.is_available():
            self.device = 'cuda'
            # print("Using cuda")
        else:
            self.device = 'cpu'
            # print("Using CPU")

    def forward(self, input_dim, input_seq, input_feature):
        # input_seq: (input_dim, seq_len, batch, input_size)
        # input_feature: (input_dim, feature_size, batch, layers)

        ans = torch.zeros(input_dim, input_seq.shape[1], input_seq.shape[2], self.output_size)
        
        for i in range(input_dim):
            h_0 = input_feature[i]
            h_0 = h_0.permute(2, 1, 0) # change to (layers, batch, feature_size)

            for module in self.dense1:
                h_0 = module(h_0) # (batch, layers, feature_size)
            
            # h_0 = h_0.permute(1, 0, 2) # change to (batch, layers, feature_size)

            c_0 = input_feature[i]
            c_0 = c_0.permute(2, 1, 0) # change to (layers, batch, feature_size)

            for module in self.dense2:
                c_0 = module(c_0) # (batch, layers, feature_size)
            
            # c_0 = c_0.permute(1, 0, 2) # change to (batch, layers, feature_size)

                
            input = input_seq[i]

            lstm_out, (h_n, c_n) = self.lstm(input, (h_0, c_0))

            lstm_out = self.fc(lstm_out)

            ans[i] = lstm_out # (input_dim, seq_len, batch, output_size)
        
        ans = ans.to(self.device)

        output = self.merge(ans.permute(1, 2, 3, 0)) # (seq_len, batch, output_size, input_dim)

        output = torch.squeeze(output) # (seq_len, batch)

        return output.permute(1, 0) # (batch, seq_len)
        

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda'
        print("Using cuda")
    else:
        device = 'cpu'
        print("Using CPU")
    
    model = varLSTM(1, 128, 1, 2, [32, 64, 64], 1)
    model = model.to(device)

    input_seq = torch.randn(4, 3801, 16, 1) # (input_dim, seq_len, batch, input_size)
    input_feature = torch.randn(4, 2, 16, 1) # (input_dim, feature_size, batch, layers)

    input_seq, input_feature = input_seq.to(device), input_feature.to(device)

    print(model)

    output = model(4, input_seq, input_feature)

    print(output.shape)