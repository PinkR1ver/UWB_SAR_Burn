import matplotlib.pyplot as plt
import numpy as np

# 样本数据
data = np.random.randn(1000)
print(data)

# 绘制直方图
plt.hist(data, edgecolor='black')  # 使用5个柱形，黑色边界

# 添加标签和标题
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram')

# 显示图形
plt.show()
