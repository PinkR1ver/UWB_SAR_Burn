import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from fastdtw import fastdtw
from scipy.stats import norm
import dataset_LSTM_interpolation as dli
import LSTM_interpolation_model as lim
import torch

def visualEvaluation_time_series_single(y_true, y_pred, confidence, savePath=None):

    x = np.arange(0, len(y_true), 1)  # 时间点

    # 生成置信区间
    z = norm.ppf(1 - (1 - confidence) / 2)  # 对应置信度的Z分数
    std = np.std(y_pred)  # 预测值的标准差
    y_pred_upper = y_pred + z * std  # 置信区间上界
    y_pred_lower = y_pred - z * std  # 置信区间下界

    # 计算评价指标
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    distance, path = fastdtw(y_true, y_pred)

    # 绘制图形
    plt.figure(figsize=(10, 8))
    plt.plot(x, y_true, label='Ground Truth')
    plt.plot(x, y_pred, label='Prediction')
    plt.fill_between(x, y_pred_lower, y_pred_upper, color='gray', alpha=0.3, label='95% Confidence Interval')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Time Series')

    # 显示评价指标
    plt.text(0.02, 0.9, f'RMSE: {rmse:.4f}', transform=plt.gca().transAxes)
    plt.text(0.02, 0.85, f'MAE: {mae:.4f}', transform=plt.gca().transAxes)
    plt.text(0.02, 0.8, f'DTW: {distance:.4f}', transform=plt.gca().transAxes)
    plt.text(0.02, 0.75, f'Mean Forecast Accuracy: {(1 - distance / (len(x) - 1)) * 100:.2f}%', transform=plt.gca().transAxes)

    plt.legend()
    plt.grid(True)

    if savePath is None:
        plt.show()
    else:
        plt.savefig(savePath)
        plt.close()



if __name__ == '__main__':
    model = lim.varLSTM(input_size=4, hidden_size=64, output_size=1, corresponding_feature_size=2, dense_node_size=[16, 32, 32], num_layers=4)

    model.load_state_dict(torch.load('LSTM_result/epoch9_9999.pth'))

    test_dataset = dli.echoDataset()

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

    for i, (ts, dis, ts_ans) in enumerate(test_loader):
        input_ts = ts.permute(1, 2, 0) # change to (input_size, seq_len, batch)

        dis = dis.unsqueeze(-1).expand(-1, -1, -1, model.num_layers) # change to (batch, input_dim, seq_len, num_layers)
        input_dis = dis.permute(1, 2, 0, 3) # change to (input_size, faeture_size, batch, num_layers)

        y_pred = model(input_ts, input_dis)[0]
        y_true = ts_ans[0]

        y_pred = y_pred.detach().numpy()
        y_true = y_true.detach().numpy()

        visualEvaluation_time_series_single(y_true, y_pred, 0.95)



    


