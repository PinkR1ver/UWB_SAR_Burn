import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from fastdtw import fastdtw
from scipy.stats import norm

# 生成时间序列数据
x = np.linspace(0, 10, 100)  # 时间点
y_true = np.sin(x)  # 真实值
y_pred = np.sin(x) + np.random.randn(100) * 0.2  # 预测值，添加噪声

# 生成置信区间
confidence = 0.95
z = norm.ppf(1 - (1 - confidence) / 2)  # 对应置信度的Z分数
std = np.std(y_pred)  # 预测值的标准差
y_pred_upper = y_pred + z * std  # 置信区间上界
y_pred_lower = y_pred - z * std  # 置信区间下界

# print(z)

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
plt.text(0.02, 0.75, f'Mean Forecast Accuracy: {1 - distance / (len(x) - 1):.4f}', transform=plt.gca().transAxes)

plt.legend()
plt.grid(True)
plt.show()
