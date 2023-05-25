import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import dataset_LSTM_interpolation as dli
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import LSTM_interpolation_model
from tqdm import tqdm
from torch import optim
import gc
from rich.progress import track

input_dim = 4
batch_size = 12
epoch = 50

if torch.cuda.is_available():
    device = 'cuda'
    print("Using cuda")
else:
    device = 'cpu'
    print("Using CPU")

if __name__ == '__main__':
    
    echoDataset = dli.echoDataset()

    train_part = 0.8

    train_size = int(train_part * len(echoDataset))
    validation_size = len(echoDataset) - train_size

    train_dataset, validation_dataset = torch.utils.data.random_split(echoDataset, [train_size, validation_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

    model = LSTM_interpolation_model.varLSTM(input_size=4, hidden_size=64, output_size=1, corresponding_feature_size=2, dense_node_size=[16, 32, 32], num_layers=4)
    model = model.to(device)

    # print(model)

    opt = optim.Adam(model.parameters())  # stochastic gradient descent
    loss_function = nn.L1Loss()

    loss_list = np.zeros(len(train_loader))

    for iter in range(epoch):
        for i, (ts, dis, ts_ans) in track(enumerate(train_loader), description=f"epoch{iter} Processing...", total=len(train_loader)):
            # ts, dis, ts_ans = ts.to(device), dis.to(device), ts_ans.to(device)
            ts_ans = ts_ans.to(device)

            '''
            print(ts.shape) # shape is (batch, input_size, seq_len), the model needs (seq_len, batch, input_size)
            print(dis.shape) # shape is (batch, input_size, feature_len)
            print(ts_ans.shape) # shape is (batch, seq_len)
            pause = input('pause')
            '''

            # ts = torch.unsqueeze(ts, dim=3) # change to (batch, input_dim, seq_len, input_size)
            input_ts = ts.permute(1, 2, 0) # change to (input_size, seq_len, batch)

            dis = dis.unsqueeze(-1).expand(-1, -1, -1, model.num_layers) # change to (batch, input_dim, seq_len, num_layers)
            input_dis = dis.permute(1, 2, 0, 3) # change to (input_size, faeture_size, batch, num_layers)

            '''
            print(input_ts.shape)
            print(input_dis.shape)
            pause = input('pause')
            '''

            input_ts, input_dis = input_ts.to(device), input_dis.to(device)
            
            predict_ts = model(input_ts, input_dis)

            loss = loss_function(predict_ts, ts_ans)
            loss_list[i] = loss

            opt.zero_grad()
            loss.backward()
            opt.step()

            if (i+1) % 100 == 0:
                fig, ax = plt.subplots()

                ax.plot(ts_ans[0].cpu().detach().numpy(), label='ans', color='blue')
                ax.plot(predict_ts[0].cpu().detach().numpy(), label='predict', color='red')

                ax.legend()
                
                # if don't have dir, create it
                if not os.path.isdir('result'):
                    os.makedirs('result')
                
                plt.savefig('result/' + 'epoch' + str(iter) + '_' + str(i) + '.png')
                plt.close()

            if (i+1) % 1000 == 0:

                # if don't have dir, create it
                if not os.path.isdir('result'):
                    os.makedirs('result')
                
                torch.save(model.state_dict(), 'result/' + 'epoch' + str(iter) + '_' + str(i) + '.pth')

            gc.collect()

        fig = plt.figure
        plt.plot(loss_list)
        plt.savefig('result/' + 'epoch' + str(iter) + '_loss' + '.png')

        

            

