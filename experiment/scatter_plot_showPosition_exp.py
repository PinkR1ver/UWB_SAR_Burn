import numpy as np
import matplotlib.pyplot as plt

# 创建一个 (5, 5) 大小的示例二维数组
data = np.array([[0, 0, 0, 0, 2],
                 [0, 1, 0, 1, 0],
                 [0, 0, 0, 1, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]])

# 创建一个与数组相同大小的网格
x = np.arange(0, data.shape[1])
y = np.arange(0, data.shape[0])
X, Y = np.meshgrid(x, y)

# 设置透明度的数组
alphas = np.where(data == 0, 0.1, 1.0)

# 绘制散点图，调整透明度和大小
scatter = plt.scatter(X.flatten(), Y.flatten(), c=data.flatten(), cmap='viridis', alpha=alphas.flatten(), s=100)

# 获取值为1和2的点的坐标
indices = np.argwhere(np.logical_or(data == 1, data == 2))
points = [tuple(idx) for idx in indices]

# 在选定的点之间绘制箭头和标注距离
for i in range(len(points) - 1):
    p1 = points[i]
    p2 = points[i + 1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    distance = np.sqrt(dx ** 2 + dy ** 2)
    arrow_color = plt.cm.viridis(distance / np.max(data))  # 根据距离获取箭头的颜色
    plt.annotate(f'{distance:.2f}', xy=(p2[0], p2[1]), xytext=(p1[0], p1[1]), arrowprops=dict(arrowstyle='->', lw=1.5, color=arrow_color))

# 创建一个颜色映射表，并显示颜色栏
norm = plt.Normalize(vmin=0, vmax=np.max(data))
cbar = plt.colorbar(scatter, norm=norm, label='Distance')

# 创建一个与散点图颜色对应的图例
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=str(i), markerfacecolor=scatter.to_rgba(i), markersize=10, markeredgecolor='k') for i in range(data.max()+1)]
plt.legend(handles=legend_elements, title='Values', loc='upper right')

# 设置坐标轴和标题
plt.xticks(np.arange(0, data.shape[1]), np.arange(0, data.shape[1]))
plt.yticks(np.arange(0, data.shape[0]), np.arange(0, data.shape[0]))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Data Points')

# 显示图形
plt.show()
