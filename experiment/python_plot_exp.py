import matplotlib.pyplot as plt
import numpy as np

# 生成示例数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建图形对象和坐标轴
fig, ax = plt.subplots()

# 绘制曲线
ax.plot(x, y1, color='blue', label='Sin(x)')
ax.plot(x, y2, color='red', label='Cos(x)')

# 添加图例
ax.legend()

# 保存图片
plt.savefig('plot.png')
