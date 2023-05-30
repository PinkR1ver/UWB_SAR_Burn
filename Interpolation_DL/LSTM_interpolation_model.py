# This file is very fragile

import torch
import torch.nn as nn
import time

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
                layers.append(nn.Linear(input_size * corresponding_feature_size, dense_node_size[i]))
            else:
                layers.append(nn.Linear(dense_node_size[i-1], dense_node_size[i]))
            
            layers.append(nn.LayerNorm(dense_node_size[i]))  # 归一化层
            layers.append(nn.ReLU())  # 激活函数层
            layers.append(nn.Dropout(0.5))  # 丢弃层
        
        layers.append(nn.Linear(dense_node_size[-1], hidden_size))

        self.dense1 = nn.Sequential(*layers)
        self.dense2 = nn.Sequential(*layers)

        self.fc = nn.Linear(hidden_size, output_size)

        # self.merge = nn.Linear(4, output_size)

        # self.merge = nn.Transformer(d_model=self.merge_hidden_size, nhead=self.merge_heads, num_encoder_layers=self.merge_encoder_layers, num_decoder_layers=self.merge_decoder_layers)

        if torch.cuda.is_available():
            self.device = 'cuda'
            # print("Using cuda")
        else:
            self.device = 'cpu'
            # print("Using CPU")

    def forward(self, input_seq, input_feature):
        '''
        # input_seq: (input_dim, seq_len, batch, input_size)
        # input_feature: (input_dim, feature_size, batch, layers)

        # Old structure
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

        '''

        # input_seq: (input_size, seq_len, batch)
        # input_feature: (input_size, feature_size, batch, layers)

        input_size = input_feature.size(0)
        feature_size = input_feature.size(1)
        batch = input_feature.size(2)
        layers = input_feature.size(3)

        reshaped_input_feature = input_feature.view(input_size * feature_size, batch, layers)

        h_0 = reshaped_input_feature.permute(1, 2, 0) # shape to (batch, layers, input_size * feature_size)
        c_0 = reshaped_input_feature.permute(1, 2, 0) # shape to (batch, layers, input_size * feature_size)

        for module in self.dense1:
            h_0 = module(h_0) # (batch, layers, hidden_size)

        h_0 = h_0.permute(1, 0, 2) # change to (layers, batch, feature_size)

        for module in self.dense1:
            c_0 = module(c_0) # (batch, layers, hidden_size)

        c_0 = c_0.permute(1, 0, 2) # change to (layers, batch, feature_size)

        input_seq = input_seq.permute(1, 2, 0) # (seq_len, batch, input_size)

        '''
        print(input_seq.shape)
        print(h_0.shape)
        print(c_0.shape)
        '''

        h_0 = h_0.contiguous()
        c_0 = c_0.contiguous()

        lstm_out, _ = self.lstm(input_seq, (h_0, c_0)) # (seq_len, batch, output_size)

        output = self.fc(lstm_out)

        output = torch.squeeze(output)

        return output.permute(1, 0)


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = 'cuda'
        print("Using cuda")
    else:
        device = 'cpu'
        print("Using CPU")
    
    time_start = time.time()
    model = varLSTM(8, 128, 1, 2, [32, 64, 64], 4)
    model = model.to(device)
    time_end = time.time()

    print('Time cost: ', time_end - time_start)

    input_seq = torch.randn(8, 3801, 16) # (input_size, seq_len, batch)
    input_feature = torch.randn(8, 2, 16, 4) # (input_size, feature_size, batch, layers)

    time_start = time.time()
    input_seq, input_feature = input_seq.to(device), input_feature.to(device)
    time_end = time.time()

    print('Time cost: ', time_end - time_start)

    # print(model)

    time_start = time.time()
    output = model(input_seq, input_feature)
    time_end = time.time()

    print('Time cost: ', time_end - time_start)

    print(output.shape)