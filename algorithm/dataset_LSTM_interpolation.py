import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import interpolation_data_collect as idc

class echoDataset(Dataset):
    def __init__(self):
        self.df = idc.interpolation_data_collect() # idc is short for interpolation_data_collect, a manully written module

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        ts1 = self.df.loc[index, 'ts1'] #ts means time series
        ts2 = self.df.loc[index, 'ts2']
        ts3 = self.df.loc[index, 'ts3']
        ts4 = self.df.loc[index, 'ts4']

        ts_ans = self.df.loc[index, 'ts_ans']

        ts = np.stack((ts1, ts2, ts3, ts4), axis=0)

        dis1 = self.df.loc[index, 'dis1']
        dis2 = self.df.loc[index, 'dis2']
        dis3 = self.df.loc[index, 'dis3']
        dis4 = self.df.loc[index, 'dis4']

        dis = np.stack((dis1, dis2, dis3, dis4), axis=0)

        return torch.Tensor(ts), torch.Tensor(dis), torch.Tensor(ts_ans)
    

if __name__ == '__main__':

    dataset = echoDataset()

    print(dataset[0][0].shape)