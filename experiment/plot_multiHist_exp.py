import matplotlib.pyplot as plt
import numpy as np

# 创建四个分布数据
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(1, 1.5, 1000)
data3 = np.random.normal(-2, 0.5, 1000)
data4 = np.random.normal(3, 2, 1000)

# 设置直方图参数
bins = 30
alpha = 0.5
label = ['Distribution 1', 'Distribution 2', 'Distribution 3', 'Distribution 4']
colors = ['red', 'green', 'blue', 'orange']

# 绘制四个分布的直方图
plt.hist([data1, data2, data3, data4], bins=bins, alpha=alpha, label=label, color=colors, histtype='barstacked')

# 添加图例
plt.legend()

# 显示图表
plt.show()

# 创建四个分布数据
data1 = np.random.normal(0, 1, 1000)
data2 = np.random.normal(1, 1.5, 1000)
data3 = np.random.normal(-2, 0.5, 1000)
data4 = np.random.normal(3, 2, 1000)

# 绘制四个分布的直方图
plt.hist(data1, bins=30, alpha=0.5, label='Distribution 1', color='red')
plt.hist(data2, bins=30, alpha=0.5, label='Distribution 2', color='green')
plt.hist(data3, bins=30, alpha=0.5, label='Distribution 3', color='blue')
plt.hist(data4, bins=30, alpha=0.5, label='Distribution 4', color='orange')

# 添加图例
plt.legend()

# 显示图表
plt.show()

n_bins=30
colors = ['blue', 'orange', 'green']
# Make a multiple-histogram of array of three values with different length.
array = [10000, 5000, 2000]
x_multi = [np.random.randn(n) for n in array ]
plt.hist(x_multi, n_bins, histtype='bar', label=colors)
plt.legend(loc="upper right")
plt.title('Different Sample Sizes')
plt.show()