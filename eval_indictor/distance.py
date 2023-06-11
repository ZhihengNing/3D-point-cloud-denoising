import os

import numpy as np
import torch

from denoise.whole_process import noise
from environments import BASE_PATH
from eval_indictor.util import write_file

dataset_base = os.path.join(BASE_PATH, "train_result", "whole_process")
DEVICE = "cpu"

"""
计算两个三维点云之间的 Chamfer 距离
Args:
    point_cloud1 (torch.Tensor): 第一个点云，形状为 (N1, 3)
    point_cloud2 (torch.Tensor): 第二个点云，形状为 (N2, 3)
Returns:
    chamfer_dist (torch.Tensor): Chamfer 距离
"""
def chamfer_distance(point_cloud1, point_cloud2):
    # 将点云转换为批量形式，增加维度
    point_cloud1 = point_cloud1.unsqueeze(0)
    point_cloud2 = point_cloud2.unsqueeze(0)

    # 计算点云之间的欧氏距离
    dist1 = torch.cdist(point_cloud1, point_cloud2)  # (1, N1, N2)
    dist2 = torch.cdist(point_cloud2, point_cloud1)  # (1, N2, N1)

    # 找到每个点云中距离最近的点的索引
    min_dist1, _ = torch.min(dist1, dim=2)  # (1, N1)
    min_dist2, _ = torch.min(dist2, dim=2)  # (1, N2)

    # 计算 Chamfer 距离，取两个方向的距离之和
    chamfer_dist = torch.mean(min_dist1) + torch.mean(min_dist2)

    return chamfer_dist


def result_generator(name):
    point_cloud_path = os.path.join(dataset_base, point_cloud_name, name)
    if not os.path.exists(point_cloud_path):
        noise()
    points1 = torch.tensor(np.loadtxt(point_cloud_path)).to(DEVICE)
    print(f"读取{point_cloud_path}完成...")
    path2 = os.path.join(dataset_base, point_cloud_name, point_cloud_name + "_clean.xyz")
    points2 = torch.tensor(np.loadtxt(path2)).to(DEVICE)
    print(f"读取{path2}完成...")
    res = chamfer_distance(points1, points2)
    print(res)


if __name__ == '__main__':
    point_cloud_name = "netsuke100k_noise_white_2.50e-03_nzh"
    write_file(os.path.join(BASE_PATH, "data", "experiment_noise_removal.txt"), point_cloud_name)

    result_generator("noise_removal_result.xyz")
    result_generator(point_cloud_name + "_sor.xyz")
    result_generator(point_cloud_name + "_shendu.xyz")
