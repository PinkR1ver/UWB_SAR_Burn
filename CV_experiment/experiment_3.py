# 导入必要的库
import numpy as np
import cv2
import matplotlib.pyplot as plt
from rich.progress import track
import multiprocessing as mp
import time
import math

# 定义图片的大小和中心点
width = 256
height = 256
center_x = width // 2
center_y = height // 2

# 定义中心亮度值和周围亮度值的差异
center_brightness = 128
delta_brightness = 10

radius_param = 0.12


# 创建一个空的8位彩图

def simulate_radarRun(args):
    img = np.zeros((height, width), dtype=np.uint32)
    run_x, run_y = args # unpack the tuple
    new_center_x = center_x + run_x
    new_center_y = center_y + run_y

    for i in range(height):
        for j in range(width):
            # 计算距离中心点的欧几里得距离
            distance = np.sqrt((i - new_center_y) ** 2 + (j - new_center_x) ** 2)
            # 根据距离计算亮度值，保证在0到255之间
            brightness = max(0, min(128, center_brightness - delta_brightness * distance * radius_param))
            # 设置图片的每个通道的亮度值，这里使用相同的值，你可以根据需要改变颜色
            img[i, j] += brightness

    return img

if __name__ == '__main__':

    '''
    # 遍历图片的每个像素，根据距离中心点的距离设置亮度值
    for i in range(height):
        for j in range(width):
            # 计算距离中心点的欧几里得距离
            distance = np.sqrt((i - center_y) ** 2 + (j - center_x) ** 2)
            # 根据距离计算亮度值，保证在0到255之间
            brightness = max(0, min(255, center_brightness - delta_brightness * distance * radius_param))
            # 设置图片的每个通道的亮度值，这里使用相同的值，你可以根据需要改变颜色
            img[i, j] = brightness

    # 显示图片
    plt.imshow(img)
    plt.show()
    '''

    pool = mp.Pool(mp.cpu_count())

    input_list = [(run_x, run_y) for run_x in np.arange(-40, 40, 5) for run_y in np.arange(-40, 40, 5)]

    print("Processing...")

    T1 = time.time()
    img_list = pool.map(simulate_radarRun, input_list)
    T2 = time.time()

    print(f"Processing time: {math.floor((T2- T1) / 60)}min{(T2 - T1) - (math.floor((T2- T1) / 60) * 60)}s")

    # Close the pool and wait for the processes to finish
    pool.close()
    pool.join()


    img_all = np.zeros((height, width), dtype=np.uint32)

    for img in img_list:
        img_all += img

    plt.imshow(img_all)
    plt.show()


    # 计算img的最大值和最小值
    img_max = np.max(img_all)
    img_min = np.min(img_all)

    # 归一化img到0-255的范围
    img_norm = ((img_all - img_min) / (img_max - img_min))
    img_norm = img_norm * 255

    img_norm  = img_norm > 200

    # 显示图片
    plt.imshow(img_norm, cmap="gray")
    # plt.colorbar()
    plt.show()
