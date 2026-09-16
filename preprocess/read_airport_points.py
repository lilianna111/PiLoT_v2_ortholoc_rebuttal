"""读取 airport_final_points.npy 并打印若干点"""
import numpy as np

# 修改为你的 .npy 文件实际路径
file_path = "/home/amax/Documents/datasets/crop/DJI_20251217153719_0002_V/airport_final_points.npy"  # 或在同目录下，或写绝对路径
data = np.load(file_path)

print(f"数据形状: {data.shape}")
print(f"数据类型: {data.dtype}")
print()

# 打印前几个点
n_show = min(5, len(data))  # 最多打印 5 个点
print(f"前 {n_show} 个点:")
for i in range(n_show):
    print(f"  点 {i}: {data[i]}")
