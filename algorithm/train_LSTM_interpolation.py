import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
import dataset_LSTM_interpolation as dli
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

batch_size = 16

if __name__ == '__main__':
    
    echoDataset = dli.echoDataset()

    train_part = 0.8

    train_size = int(train_part * len(echoDataset))
    validation_size = len(echoDataset) - train_size

    train_dataset, validation_dataset = torch.utils.data.random_split(echoDataset, [train_size, validation_size])

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=False)

    for i, (ts, dis, ts_ans) in enumerate(train_loader):
        print(ts.shape)
        print(dis.shape)
        print(ts_ans.shape)
        pause = input('pause')