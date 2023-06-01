import numpy as np
import matplotlib.pyplot as plt

# 创建一个 (5, 5) 大小的示例二维数组
data = np.array([[0, 0, 0, 0, 2],
                 [0, 1, 0, 1, 0],
                 [0, 0, 0, 1, 0],
                 [1, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]])

# 创建一个与数组相同大小的网格
x = np.arange(0, data.shape[0])
y = np.arange(0, data.shape[1])
X, Y = np.meshgrid(x, y)

# 设置透明度的数组
alphas = np.where(data == 0, 0.05, 0.7)

# 绘制散点图，调整透明度和大小
scatter = plt.scatter(X.flatten(), Y.flatten(), c=data.flatten(), cmap='viridis', alpha=alphas.flatten(), s=100)

# 获取值为1和2的点的坐标
indices_1 = np.argwhere(data == 1)
indices_2 = np.argwhere(data == 2)
# print(indices_1)
# print('----------------')
# print(indices_2)

indices_1 = indices_1[:,::-1]
indices_2 = indices_2[:,::-1]

distance = np.empty(len(indices_1))

for i in range(len(indices_1)):
    dx = indices_2[0][0] - indices_1[i][0]
    dy = indices_2[0][1] - indices_1[i][1]
    distance[i] = np.sqrt(dx ** 2 + dy ** 2)

# 创建一个标准化器，将距离值归一化到 [0, 1] 范围内
norm = plt.Normalize(distance.min(), distance.max())

# 创建一个颜色映射对象
cmap = plt.cm.get_cmap('jet')

for i in range(len(indices_1)):
    arrow_color = cmap(norm(distance[i]))  # 根据距离获取箭头的颜色
    plt.annotate(f'{distance[i]:.2f}', xy=(indices_2[0][0], indices_2[0][1]), xytext=(indices_1[i][0], indices_1[i][1]), arrowprops=dict(arrowstyle='->', lw=1.5, color=arrow_color))




# points = [tuple(idx) for idx in indices]

# # 在选定的点之间绘制箭头和标注距离
# for i in range(len(points) - 1):
#     p1 = points[i]
#     p2 = points[i + 1]
#     dx = p2[0] - p1[0]
#     dy = p2[1] - p1[1]
#     distance = np.sqrt(dx ** 2 + dy ** 2)
#     arrow_color = plt.cm.viridis(distance / np.max(data))  # 根据距离获取箭头的颜色
#     plt.annotate(f'{distance:.2f}', xy=(p2[0], p2[1]), xytext=(p1[0], p1[1]), arrowprops=dict(arrowstyle='->', lw=1.5, color=arrow_color))

# # 创建一个颜色映射表，并显示颜色栏
# norm = plt.Normalize(vmin=0, vmax=np.max(data))
# cbar = plt.colorbar(scatter, norm=norm, label='Distance')

# 创建一个与散点图颜色对应的图例
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=str(i), markerfacecolor=(*scatter.to_rgba(i)[:3], 0.05 if i == 0 else 0.7), markersize=10) for i in range(data.max()+1)]
plt.legend(handles=legend_elements, labels=['None Use', 'Use', 'Target'] ,title='Values', loc='upper right')

# 设置坐标轴和标题
plt.xticks(np.arange(0, data.shape[1]), np.arange(0, data.shape[1]))
plt.yticks(np.arange(0, data.shape[0]), np.arange(0, data.shape[0]))
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of Data Points')

# 创建 ScalarMappable 对象，并设置相应的属性
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])  # 设置空数组以避免警告

# 绘制颜色条
colorbar = plt.colorbar(sm)

# 设置颜色条标签
colorbar.set_label('Distance')

# 显示图形
plt.show()
