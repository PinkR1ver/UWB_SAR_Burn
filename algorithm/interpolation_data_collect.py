import scipy.io
import os
import math
import itertools
import pandas as pd
import numpy as np
from rich.progress import track
from matplotlib import pyplot as plt

def index_to_position(index, Xbeg=0, Xend=0.16, Ybeg=0, Yend=0.16, scan_points=5):
    
    Xstep = (Xend - Xbeg) / (scan_points - 1)
    Ystep = (Yend - Ybeg) / (scan_points - 1)

    x = math.floor(index / 5) * Xstep + Xbeg;
    y = (index % 5) * Ystep + Ybeg;

    return (x, y)

def interpolation_data_collect():

    base_path = os.path.dirname(__file__)

    # Load .mat file

    SAR_sample = scipy.io.loadmat(os.path.join(base_path, '..', 'data', 'data_8080_2_1_25.mat'))
    SAR_sample = SAR_sample['data_8080_2_1_25']

    # 从0到24生成所有可能的4个数的组合
    numbers = list(range(25))
    combinations = list(itertools.combinations(numbers, 4))

    # 遍历所有组合并加上剩下的21个数中的一个
    comb_list = []
    for comb in combinations:
        for num in range(25):
            if num not in comb:
                comb_list.append(comb + (num,))

    data_size  = len(comb_list)

    '''
    # 创建特征名称列表
    feature_names = ['ts1', 'dis1', 'ts2', 'dis2', 'ts3', 'dis3', 'ts4', 'dis4', 'ts_ans']

    # 创建全为0的DataFrame
    df = pd.DataFrame(0, columns=feature_names, index=range(data_size))
    '''

    # 创建包含初始值的字典
    data = {
        'ts1': pd.Series([np.array([0])] * data_size),
        'ts2': pd.Series([np.array([0])] * data_size),
        'ts3': pd.Series([np.array([0])] * data_size),
        'ts4': pd.Series([np.array([0])] * data_size),
        'ts_ans': pd.Series([np.array([0])] * data_size),
        'dis1': pd.Series([np.array([0])] * data_size),
        'dis2': pd.Series([np.array([0])] * data_size),
        'dis3': pd.Series([np.array([0])] * data_size),
        'dis4': pd.Series([np.array([0])] * data_size)
    }

    # 创建DataFrame
    df = pd.DataFrame(data)

    for index in track(range(data_size), description="Processing..."):
        ts1 = SAR_sample[comb_list[index][0]]
        ts2 = SAR_sample[comb_list[index][1]]
        ts3 = SAR_sample[comb_list[index][2]]
        ts4 = SAR_sample[comb_list[index][3]]
        ts_ans = SAR_sample[comb_list[index][4]]

        (x1, y1) = index_to_position(comb_list[index][0])
        (x2, y2) = index_to_position(comb_list[index][1])
        (x3, y3) = index_to_position(comb_list[index][2])
        (x4, y4) = index_to_position(comb_list[index][3])
        (x_ans, y_ans) = index_to_position(comb_list[index][4])

        dis1 = (x1-x_ans, y1-y_ans)
        dis2 = (x2-x_ans, y2-y_ans)
        dis3 = (x3-x_ans, y3-y_ans)
        dis4 = (x4-x_ans, y4-y_ans)

        df.loc[index, 'ts1'] = ts1
        df.loc[index, 'ts2'] = ts2
        df.loc[index, 'ts3'] = ts3
        df.loc[index, 'ts4'] = ts4
        df.loc[index, 'ts_ans'] = ts_ans
        
        df.loc[index, 'dis1'] = dis1
        df.loc[index, 'dis2'] = dis2
        df.loc[index, 'dis3'] = dis3
        df.loc[index, 'dis4'] = dis4

    pd.set_option('display.max_columns', None)  # 显示所有列
    pd.set_option('display.max_rows', None)     # 显示所有行

    np.set_printoptions(linewidth=4000)

    if __name__ == '__main__':
        #df to .csv
        df.to_csv(os.path.join(base_path, '..', 'data', 'interploation_data.csv'), index=False)

        print("Data saved to interploation_data.csv")

    return df;

if __name__ == '__main__':
    df = interpolation_data_collect()
    
    '''
    plt.figure()
    plt.plot(df['ts1'][0])
    plt.show()
    '''