import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import interpolation_data_collect as idc
import scipy
import itertools
import math

def index_to_position(index, Xbeg=0, Xend=0.16, Ybeg=0, Yend=0.16, scan_points=5):
    
    Xstep = (Xend - Xbeg) / (scan_points - 1)
    Ystep = (Yend - Ybeg) / (scan_points - 1)

    x = math.floor(index / 5) * Xstep + Xbeg;
    y = (index % 5) * Ystep + Ybeg;

    return (x, y)


class echoDataset(Dataset):
    def __init__(self):
        base_path = os.path.dirname(__file__)

        # Load .mat file

        SAR_sample = scipy.io.loadmat(os.path.join(base_path, '..', 'data', 'data_8080_2_1_25.mat'))
        SAR_sample = SAR_sample['data_8080_2_1_25']

        self.SAR_sample = SAR_sample

        # 从0到24生成所有可能的4个数的组合
        numbers = list(range(25))
        combinations = list(itertools.combinations(numbers, 4))

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

        ts1 = self.SAR_sample[self.comb_list[index][0]]
        ts2 = self.SAR_sample[self.comb_list[index][1]]
        ts3 = self.SAR_sample[self.comb_list[index][2]]
        ts4 = self.SAR_sample[self.comb_list[index][3]]
        ts_ans = self.SAR_sample[self.comb_list[index][4]]

        ts = np.stack((ts1, ts2, ts3, ts4), axis=0)

        (x1, y1) = index_to_position(self.comb_list[index][0])
        (x2, y2) = index_to_position(self.comb_list[index][1])
        (x3, y3) = index_to_position(self.comb_list[index][2])
        (x4, y4) = index_to_position(self.comb_list[index][3])
        (x_ans, y_ans) = index_to_position(self.comb_list[index][4])

        dis1 = (x1-x_ans, y1-y_ans)
        dis2 = (x2-x_ans, y2-y_ans)
        dis3 = (x3-x_ans, y3-y_ans)
        dis4 = (x4-x_ans, y4-y_ans)

        dis = np.stack((dis1, dis2, dis3, dis4), axis=0)


        return torch.Tensor(ts), torch.Tensor(dis), torch.Tensor(ts_ans)
    

if __name__ == '__main__':

    dataset = echoDataset()

    print(dataset[0][0].shape)