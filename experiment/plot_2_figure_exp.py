import matplotlib.pyplot as plt

# 创建数据
x = [1, 2, 3, 4, 5]
y1 = [1, 4, 9, 16, 25]
y2 = [1, 8, 27, 64, 125]

# 创建一个 figure 对象和两个 subplot 对象
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

# 在第一个 subplot 上绘制第一个 plot
ax1.plot(x, y1, 'r-', label='Plot 1')
ax1.set_ylabel('Plot 1')
ax1.legend()

# 在第二个 subplot 上绘制第二个 plot
ax2.plot(x, y2, 'g-', label='Plot 2')
ax2.set_ylabel('Plot 2')
ax2.legend()

# 调整 subplot 的位置
fig.tight_layout()

# 显示图形
plt.show()
