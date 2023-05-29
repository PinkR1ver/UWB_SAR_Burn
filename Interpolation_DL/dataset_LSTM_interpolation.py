import os
import torch
from torch.utils.data import Dataset
import numpy as np
import scipy.io
import itertools
import math

def index_to_position(index, Xbeg=0, Xend=0.16, Ybeg=0, Yend=0.16, scan_points=5):
    
    Xstep = (Xend - Xbeg) / (scan_points - 1)
    Ystep = (Yend - Ybeg) / (scan_points - 1)

    x = math.floor(index / 5) * Xstep + Xbeg;
    y = (index % 5) * Ystep + Ybeg;

    return (x, y)


class echoDataset(Dataset):
    def __init__(self, input_size, data_path):
        self.input_size = input_size
        self.data_path = data_path

        # Load .mat file

        SAR_sample = scipy.io.loadmat(data_path)
        data_name = data_path.split('\\')[-1].split('.')[0]
        SAR_sample = SAR_sample[data_name]

        self.SAR_sample = SAR_sample

        # 从0到24生成所有可能的4个数的组合
        numbers = list(range(25))
        combinations = list(itertools.combinations(numbers, self.input_size))

        comb_list = []
        for comb in combinations:
            for num in range(25):
                if num not in comb:
                    comb_list.append(comb + (num,))

        self.comb_list = comb_list

    def __len__(self):
        return len(self.comb_list)
    
    def __getitem__(self, index):
        '''
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
        '''

        ts = np.empty((self.input_size, len(self.SAR_sample[0])))

        for i in range(self.input_size):
            ts[i] = self.SAR_sample[self.comb_list[index][i]]

        ts_ans = self.SAR_sample[self.comb_list[index][self.input_size]]

        dis = np.empty((self.input_size, 2))

        for i in range(self.input_size):
            (x, y) = index_to_position(self.comb_list[index][i])
            (x_ans, y_ans) = index_to_position(self.comb_list[index][self.input_size])
            dis[i] = (x - x_ans, y - y_ans)

        return torch.Tensor(ts), torch.Tensor(dis), torch.Tensor(ts_ans)
    

if __name__ == '__main__':

    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, '..', 'data', 'data_8080_2_1_25.mat')

    dataset = echoDataset(8, data_path=data_path)

    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)