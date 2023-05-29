import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from fastdtw import fastdtw

# 生成示例数据
time = np.arange(0, 10, 0.1)
ground_truth = np.sin(time)
prediction = np.sin(time)
confidence_interval = 0.2 * np.ones_like(prediction)  # 示例置信区间的宽度

# 计算评价指标
rmse = np.sqrt(mean_squared_error(ground_truth, prediction))
mae = mean_absolute_error(ground_truth, prediction)
dtw_distance, _ = fastdtw(ground_truth, prediction)
# ground_truth = np.where(ground_truth != 0, ground_truth, 1e-16)
# mean_accuracy = np.mean(np.where((1 - (np.abs(ground_truth - prediction) / np.abs(ground_truth))) > 0, 1 - (np.abs(ground_truth - prediction) / np.abs(ground_truth)), 0)) * 100
# print(np.where((1 - (np.abs(ground_truth - prediction) / np.abs(ground_truth))) > 0, 1 - (np.abs(ground_truth - prediction) / np.abs(ground_truth)), 0))

# mean_accuracy = np.mean(np.where(ground_truth != 0, np.maximum(1 - (np.abs(ground_truth - prediction) / np.abs(ground_truth)), 0), np.maximum(1 - (np.abs(ground_truth - prediction + 1e-8) / np.abs(ground_truth + 1e-8)), 0))) * 100
ground_truth
mean_forecast_accuracy = np.mean(1 - np.abs(ground_truth - prediction) / np.abs(ground_truth))

# 绘制预测值和置信区间
plt.plot(time, prediction, label='Prediction', color='blue')
plt.fill_between(time, prediction - confidence_interval, prediction + confidence_interval, color='blue', alpha=0.3)

# 绘制实际值
plt.plot(time, ground_truth, label='Ground Truth', color='red')

# 添加图例和标签
plt.legend(loc='upper right')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Time Series Prediction')

# 显示评价指标
metrics_text = f'RMSE: {rmse:.3f}\nMAE: {mae:.3f}\nDTW: {dtw_distance:.3f}\nMean Accuracy: {mean_forecast_accuracy:.2f}%'
plt.text(3, -2, metrics_text, fontsize=10, verticalalignment='top')
plt.subplots_adjust(bottom=0.35)

# 显示图形
plt.show()
