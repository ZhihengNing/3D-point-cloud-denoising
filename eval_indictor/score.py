import os

import numpy as np
import torch

from denoise.whole_process import outliers
from environments import BASE_PATH
from eval_indictor.util import read_file, write_file

dataset_base = os.path.join(BASE_PATH, "train_result", "whole_process")

Experiment_Outliers_Removal_Txt = os.path.join(BASE_PATH, "data", "experiment_outliers_removal.txt")
Res = [[], [], [], [], [], []]


def acc_precision_recall(predictions, targets):
    tp = np.sum((predictions == 0) & (targets == 0))
    fp = np.sum((predictions == 0) & (targets == 1))
    fn = np.sum((predictions == 1) & (targets == 0))
    tn = np.sum((predictions == 1) & (targets == 1))
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    return accuracy, precision, recall


def fa_score(precision, recall, a):
    return (1 + a ** 2) * (precision * recall) / ((a ** 2 * precision) + recall)


def f1_score(precision, recall):
    return fa_score(precision, recall, 1)


def f2_score(precision, recall):
    return fa_score(precision, recall, 2)


def get_points(point_cloud_name, file_name, dtype):
    path = os.path.join(dataset_base, point_cloud_name, file_name)
    points = np.genfromtxt(path, dtype=dtype)
    return points


"""
计算两个张量的交集，并生成交集元素个数个1，其余元素全为0的张量
Args:
    tensor1 (torch.Tensor): 第一个张量
    tensor2 (torch.Tensor): 第二个张量
Returns:
    intersection_tensor (torch.Tensor): 交集张量
"""


def intersection(tensor1, tensor2):
    # 获取两个张量的交集
    intersect = torch.intersect1d(tensor1, tensor2)

    # 创建与 tensor1 大小相同的全零张量
    intersection_tensor = torch.zeros_like(tensor1)

    # 将交集元素设置为1
    intersection_tensor[intersect] = 1

    return intersection_tensor


def get_points_tensor(base_points, base_points_indices, points):
    # tuple_array1 = set(base_points)
    # tuple_array2 = set(points)
    #
    # #先找到交集
    # res=list(tuple_array1&tuple_array2)

    points_indices = np.ones(base_points_indices.shape[0], dtype=np.int32)

    indices = np.unique(np.where(np.isin(base_points, points))[0])

    points_indices[indices] = 0

    # temp=np.where(np.isin(base_points, points))
    # points_indices[temp]=base_points_indices[temp]
    # for index,item in enumerate(res):
    #     base_points_index=np.where(base_points==list(item))[0]
    #     points_indices[base_points_index]=0

    return points_indices


def outliers_eval_indictor(point_cloud_name, file_name):
    base_points_indices = get_points(point_cloud_name, point_cloud_name + ".outliers", np.int32)
    base_points = get_points(point_cloud_name, point_cloud_name + ".xyz", np.float32)
    points = get_points(point_cloud_name, file_name, dtype=np.float32)

    point_tensor = get_points_tensor(base_points, base_points_indices, points)

    acc, pre, recall = acc_precision_recall(point_tensor, base_points_indices)
    return acc, pre, recall, f1_score(pre, recall), f2_score(pre, recall)


def calculate_outliers_score(point_cloud_name, file_name):
    print(f"getting information from {point_cloud_name}")

    #
    if not os.path.exists(os.path.join(dataset_base, point_cloud_name, file_name)):
        outliers()
    res = outliers_eval_indictor(point_cloud_name, file_name)

    if "gauss" in point_cloud_name:
        if "5.00e-03" in point_cloud_name:
            Res[0].append(np.array(res))
        elif "1.00e-02" in point_cloud_name:
            Res[1].append(np.array(res))
        elif "2.50e-02" in point_cloud_name:
            Res[2].append(np.array(res))
    else:
        if "5.00e-03" in point_cloud_name:
            Res[3].append(np.array(res))
        elif "1.00e-02" in point_cloud_name:
            Res[4].append(np.array(res))
        elif "2.50e-02" in point_cloud_name:
            Res[5].append(np.array(res))
    return res


def auto_calculate_outliers_score():
    test_set = read_file(os.path.join(BASE_PATH, "data", "outliers_removal_testset.txt"))

    for item in test_set:
        write_file(os.path.join(BASE_PATH, "data", "experiment_outliers_removal.txt"), item)
        calculate_outliers_score(item, "outliers_removal_result")


if __name__ == '__main__':
    # 自动查看结果
    # auto_calculate_outliers_score()
    #
    # for item in Res:
    #     print(item)
    #     print(np.mean(item,axis=0))

    point_cloud_name = "dragon100k_noise_white_2.50e-02_outliers_gaussian"
    res = calculate_outliers_score(point_cloud_name, point_cloud_name + "_sor.xyz")
    print(res)
    # for point_cloud_name in list:
    #     write_file(os.path.join(BASE_PATH, "data", "experiment_outliers_removal.txt"), point_cloud_name)
    #     calculate_outliers_score(point_cloud_name,"outliers_removal_result")
