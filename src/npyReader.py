import numpy as np
import os

# 0AAQ6BO3_lower_lower_sampled_points.npy, 0AAQ6BO3_upper_upper_sampled_points.npy, 0DNK2I7H_lower_lower_sampled_points.npy
npy_path = "../run/preprocessed_data/0AAQ6BO3_lower_lower_sampled_points.npy"

data = np.load(npy_path)

# v 一共7维，前3维为x，y，z；后三维为法向量；最后一位为标签
# f 面数据，代表绘制3D模型的面的顶点索引，预处理不保存面
print(data[:5])

num_points = data.shape[0]
print(f"点数量: {num_points}")

