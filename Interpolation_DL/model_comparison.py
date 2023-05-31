import dataset_LSTM_interpolation as dli
import numpy as np
import evaluation_interpolation_model as eim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import scipy.io
import os
import math
import torch
from rich.progress import track
from scipy.stats import norm

def index_to_position(index, Xbeg=0, Xend=0.16, Ybeg=0, Yend=0.16, scan_points=5):
    
    Xstep = (Xend - Xbeg) / (scan_points - 1)
    Ystep = (Yend - Ybeg) / (scan_points - 1)

    x = math.floor(index / scan_points) * Xstep + Xbeg;
    y = (index % scan_points) * Ystep + Ybeg;

    return (x, y)


class interpolation_from_4ScanPosition_dataset(Dataset):
    def __init__(self, data_path, size):
        self.data_path = data_path
        self.size = size

        SAR_sample = scipy.io.loadmat(data_path)
        data_name = data_path.split('\\')[-1].split('.')[0]
        SAR_sample = SAR_sample[data_name]

        self.SAR_sample = SAR_sample
        
        comb_list = []
        for i in range(SAR_sample.shape[0]):
            if i % size == 0 or (i + 1) % size == 0 or (i + (size * 2)) > SAR_sample.shape[0]: # board detection
                continue
                
            comb_list.append((i, i + size - 1, i + (size * 2), i + size + 1, i + size))
        
        self.comb_list = comb_list

    def __len__(self):
        return len(self.comb_list)
    
    def __getitem__(self, index):
        ts = np.empty((4, len(self.SAR_sample[0])))

        for i in range(4):
            ts[i] = self.SAR_sample[self.comb_list[index][i]]

        ts_ans = self.SAR_sample[self.comb_list[index][4]]

        dis = np.empty((4, 2))

        for i in range(4):
            (x, y) = index_to_position(self.comb_list[index][i])
            (x_ans, y_ans) = index_to_position(self.comb_list[index][4])
            dis[i] = (x - x_ans, y - y_ans)

        return ts, dis, ts_ans
    
def four4_scanPosition_linearInterpolation(ts, position):
    dis = np.empty(ts.shape[0])
    dis_sum  = 0

    for i in range(position.shape[0]):
        dis[i] = math.sqrt(position[i][0] ** 2 + position[i][1] ** 2)
        dis_sum += dis[i]


    partial = np.empty(ts.shape[0])
    for i in range(partial.shape[0]):
        partial[i] = dis[i] / dis_sum

    for i in range(ts.shape[0]):
        ts[i] = ts[i] * partial[i]
    
    ts_interpolation = np.sum(ts, axis=0)

    return ts_interpolation

def four4_scanPosition_squardInterpolation(ts, position):
    dis = np.empty(ts.shape[0])
    dis_sum  = 0

    for i in range(position.shape[0]):
        dis[i] = math.sqrt(position[i][0] ** 2 + position[i][1] ** 2)
        dis_sum += dis[i] ** 2


    partial = np.empty(ts.shape[0])
    for i in range(partial.shape[0]):
        partial[i] = (dis[i] ** 2) / dis_sum

    for i in range(ts.shape[0]):
        ts[i] = ts[i] * partial[i]
    
    ts_interpolation = np.sum(ts, axis=0)

    return ts_interpolation

def four4_scanPosition_polyInterpolation(ts_input, position, degree=2):
    ts = ts_input.copy() # avoid change input ts, deep copy it
    dis = np.empty(ts.shape[0])
    dis_sum  = 0

    for i in range(position.shape[0]):
        dis[i] = math.sqrt(position[i][0] ** 2 + position[i][1] ** 2)
        dis_sum += (dis[i] ** degree)


    partial = np.empty(ts.shape[0])
    for i in range(partial.shape[0]):
        partial[i] = (dis[i] ** degree) / dis_sum

    for i in range(ts.shape[0]):
        ts[i] = ts[i] * partial[i]
    
    ts_interpolation = np.sum(ts, axis=0)

    return ts_interpolation

def metrics_hist_plot(metircs_list, path):
    for i in range(4):
        name = ['RMSE', 'MAE', 'DTW', 'MFA']
        plt.figure()
        plt.hist(metircs_list[:, i], edgecolor='black')
        plt.title(f'{name[i]} distribution')
        plt.savefig(os.path.join(path[i]))
        plt.close()


def scan_position_interpolation_combine_prediction_show(ts_ture, ts_pred, confidence, scan_position_index, scan_position_shape=(5, 5), save_path=None):

    x = np.arange(0, ts_ture.shape[0], 1)

    # 生成置信区间
    z = norm.ppf(1 - (1 - confidence) / 2)  # 对应置信度的Z分数
    std = np.std(ts_pred)  # 预测值的标准差
    y_pred_upper = ts_pred + z * std  # 置信区间上界
    y_pred_lower = ts_pred - z * std  # 置信区间下界

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    rmse, mae, distance, mfa = eim.evaluation_time_series_txt(ts_ture, ts_pred)


    ax1.plot(x, ts_ture, label='ground_truth')
    ax1.plot(x, ts_pred, label='pred')
    ax1.fill_between(x, y_pred_lower, y_pred_upper, color='gray', alpha=0.3, label='95% Confidence Interval')

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.set_title('Time Series')

    ax1.text(0.02, 0.3, f'RMSE: {rmse:.4f}', transform=ax1.transAxes)
    ax1.text(0.02, 0.25, f'MAE: {mae:.4f}', transform=ax1.transAxes)
    ax1.text(0.02, 0.2, f'DTW: {distance:.4f}', transform=ax1.transAxes)
    ax1.text(0.02, 0.15, f'Mean Forecast Accuracy: {mfa * 100:.2f}%', transform=ax1.transAxes)

    ax1.legend()
    ax1.grid(True)

    ## plot scan position
    scan_position = np.zeros(scan_position_shape)
    for i in range(len(scan_position_index) - 1):
        scan_position[scan_position_index[i] // scan_position_shape[0], scan_position_index[i] % scan_position_shape[1]] = 1
    
    scan_position[scan_position_index[-1] // scan_position_shape[0], scan_position_index[-1] % scan_position_shape[1]] = 2
    
    x = np.arange(0, scan_position.shape[1])
    y = np.arange(0, scan_position.shape[0])
    X, Y = np.meshgrid(x, y)

    alphas = np.where(scan_position == 0, 0.05, 0.7)

    scatter = plt.scatter(X.flatten(), Y.flatten(), c=scan_position.flatten(), cmap='viridis', alpha=alphas.flatten(), s=100)

    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=str(i), markerfacecolor=(*scatter.to_rgba(i)[:3], 0.05 if i == 0 else 0.7), markersize=10) for i in range(int(scan_position.max())+1)]
    ax2.legend(handles=legend_elements, labels=['None Use', 'Use', 'Target'], bbox_to_anchor=(1.4, 1), loc='upper right')

    ax2.set_aspect('equal')
    ax2.set_xticks(np.arange(0, scan_position.shape[1]), np.arange(0, scan_position.shape[1]))
    ax2.set_yticks(np.arange(0, scan_position.shape[0]), np.arange(0, scan_position.shape[0]))
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Interpolation situation')




    fig.tight_layout()

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()




if __name__ == '__main__':
    base_path = os.path.dirname(__file__)
    model_evalution_path = os.path.join(base_path, 'model_evalution')
    # linear_interpolation_path_4 = os.path.join(model_evalution_path, 'linear_interpolation_4')
    # linear_interpolation_path_all = os.path.join(model_evalution_path, 'linear_interpolation_all')
    # squard_interpolation_path_4 = os.path.join(model_evalution_path, 'squard_interpolation_4')

    # if not os.path.exists(linear_interpolation_path_4):
    #     os.makedirs(linear_interpolation_path_4)
    
    # if not os.path.exists(linear_interpolation_path_all):
    #     os.makedirs(linear_interpolation_path_all)

    # if not os.path.exists(squard_interpolation_path_4):
    #     os.makedirs(squard_interpolation_path_4)


    ## create degree path
    degree_path = os.path.join(model_evalution_path, 'comparison_different_degree')
    if not os.path.exists(degree_path):
        os.makedirs(degree_path)

    degree = np.arange(1, 3, 0.5)    

    test_dataset = interpolation_from_4ScanPosition_dataset(os.path.join(base_path, '..', 'data', 'data_8080_2_1_25.mat'), 5)
    metrics_list = np.empty((len(test_dataset), 4)) # [rmse, mae, distance, mfa]


    for i in range(len(test_dataset)):

        ts = test_dataset[i][0]
        dis = test_dataset[i][1]
        ts_ans = test_dataset[i][2]

        scan_index = test_dataset.comb_list[i]

        ts_pred = np.empty((degree.shape[0], ts_ans.shape[0]))

        for j in range(degree.shape[0]):
            ts_pred[j] = four4_scanPosition_polyInterpolation(ts, dis, degree[j])

        scan_position_interpolation_combine_prediction_show(ts_ans, ts_pred[0], 0.95, scan_index)

        # x = np.arange(0, ts_ans.shape[0], 1)
        # plt.figure(figsize=(10, 8))
        # for j in range(degree.shape[0]):
        #     plt.plot(x, ts_pred[j], label=f'degree={degree[j]}')
        # plt.plot(x, ts_ans, label='ans')

        # plt.xlabel('Time')
        # plt.ylabel('Value')
        # plt.title('Time Series')

        # plt.legend()
        # plt.grid(True)
        # plt.show()
        


    '''

    test_dataset = interpolation_from_4ScanPosition_dataset(os.path.join(base_path, '..', 'data', 'data_8080_2_1_25.mat'), 5)

    # print(test_dataset[0][0].shape)

    metrics_list = np.empty((len(test_dataset), 4)) # [rmse, mae, distance, mfa]

    for i in range(len(test_dataset)):

        ts = test_dataset[i][0]
        dis = test_dataset[i][1]
        ts_ans = test_dataset[i][2]

        ts_pred = four4_scanPosition_linearInterpolation(ts, dis)

        eim.visualEvaluation_time_series_single(ts_ans, ts_pred, 0.95, os.path.join(linear_interpolation_path_4, f'{i}.png'))

        metrics_list[i] = eim.evaluation_time_series_txt(ts_ans, ts_pred)

    
    # save metrics distribution image
    name = ['RMSE', 'MAE', 'DTW', 'MFA']
    path = []
    for i in range(4):
        path.append(os.path.join(linear_interpolation_path_4, f'{name[i]} distribution.png'))


    metrics_hist_plot(metrics_list, path)

    '''

    '''
    test_dataset = interpolation_from_4ScanPosition_dataset(os.path.join(base_path, '..', 'data', 'data_8080_2_1_25.mat'), 5)

    # print(test_dataset[0][0].shape)

    metrics_list = np.empty((len(test_dataset), 4)) # [rmse, mae, distance, mfa]

    for i in track(range(len(test_dataset)), description="Squard Interpolation evaluation...", total=len(test_dataset)):

        ts = test_dataset[i][0]
        dis = test_dataset[i][1]
        ts_ans = test_dataset[i][2]

        ts_pred = four4_scanPosition_squardInterpolation(ts, dis)

        eim.visualEvaluation_time_series_single(ts_ans, ts_pred, 0.95, os.path.join(squard_interpolation_path_4, f'{i}.png'))

        metrics_list[i] = eim.evaluation_time_series_txt(ts_ans, ts_pred)

    
    # save metrics distribution image
    name = ['RMSE', 'MAE', 'DTW', 'MFA']
    path = []
    for i in range(4):
        path.append(os.path.join(squard_interpolation_path_4, f'{name[i]} distribution.png'))


    metrics_hist_plot(metrics_list, path)
    '''

    

    '''
    test_dataset = dli.echoDataset(4, os.path.join(base_path, '..', 'data', 'data_8080_2_1_25.mat'))

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=12, shuffle=False)

    metrics_list = np.empty((len(test_dataset), 4)) # [rmse, mae, distance, mfa]

    for i, (ts, dis, ts_ans) in track(enumerate(test_loader), description="Interpolation evaluation...", total=len(test_loader)):
        
        for j in range(ts.shape[0]):
            ts_pred = four4_scanPosition_linearInterpolation(ts[j].detach().numpy(), dis[j].detach().numpy())
            ts_ans_tmp = ts_ans[j].detach().numpy()

            # print(ts_ans.shape)

            eim.visualEvaluation_time_series_single(ts_ans_tmp, ts_pred, 0.95, os.path.join(linear_interpolation_path_all, f'{i * 12 + j}.png'))

            metrics_list[i * 12 + j] = eim.evaluation_time_series_txt(ts_ans_tmp, ts_pred)

    name = ['RMSE', 'MAE', 'DTW', 'MFA']
    path = []
    for i in range(4):
        path.append(os.path.join(linear_interpolation_path_all, f'{name[i]} distribution.png'))


    metrics_hist_plot(metrics_list, path)
    '''



    

