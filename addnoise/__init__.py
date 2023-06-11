import os.path
import shutil

import numpy as np

from addnoise.util import search_file_in_dir
from environments import BASE_PATH


def generator_noise_point_cloud(path, new_path, func):
    points = np.loadtxt(path)
    noise_points = func(points)
    np.savetxt(new_path, noise_points)
    return new_path


def add_anisotropic_gaussian_noise(points, mean, cov):
    # 生成与点云数据形状相同的高斯噪声
    noise = np.random.multivariate_normal(mean, cov, size=points.shape[0])

    # 将噪声添加到点云数据中
    noisy_points = points + noise

    return noisy_points


def add_noise(points, mean=0, cov=0.1):
    noise = np.random.multivariate_normal(mean, cov, points.shape)
    noise_points = noise + points
    return noise_points


def add_normal_outliers(points, mean, conv):
    points_num = points.shape[0]
    index = np.random.choice(points_num, size=int(points_num * 0.75), replace=False)
    points = points[index]
    outliers = np.random.multivariate_normal(mean, conv, size=int(points_num * 0.25))
    points = np.concatenate((points, outliers))
    return points


def add_outliers_extra(points, mean, conv, ratio):
    points_num = points.shape[0]
    n_outliers = int(points_num * ratio)

    outliers_id = np.random.choice(points_num, size=n_outliers, replace=False)
    offsets = np.random.normal(mean, conv, size=[n_outliers, 3])
    new_points = points[outliers_id] + offsets
    new_points = np.concatenate((points, new_points))
    return new_points


def add_outliers(points, new_points, mean, conv, ratio):
    points_num = points.shape[0]
    # n_outliers=np.random.randint(0,points_num)
    n_outliers = int(points_num * ratio)
    # max_offset=float(np.linalg.norm(points.max(0) - points.min(0), 2))
    #
    outliers_idx = np.random.choice(points_num, size=n_outliers, replace=False)

    offsets = np.random.normal(mean, conv, size=[n_outliers, 3])

    final_points = points[outliers_idx]
    need_indices = np.logical_not(np.isin(np.arange(len(points)), outliers_idx))

    final_points = final_points + offsets
    new_points = new_points[need_indices]
    new_points = np.concatenate((new_points, final_points))
    return new_points, points[need_indices]


if __name__ == '__main__':
    point_cloud_name = "netsuke100k_noise_white_2.50e-03"
    in_dir = search_file_in_dir(os.path.join(BASE_PATH, "data"), point_cloud_name + ".xyz")
    point_cloud_result_dir = os.path.join(BASE_PATH, "data", "pointCleanNetComplete")
    original_path = os.path.join(in_dir, f"{point_cloud_name}.xyz")
    if not os.path.exists(point_cloud_result_dir):
        os.makedirs(point_cloud_result_dir)
    new_path = os.path.join(point_cloud_result_dir, f"{point_cloud_name}.xyz")
    if os.path.exists(original_path):
        shutil.copyfile(original_path, new_path)

    clean_original_path = os.path.join(in_dir, f"{point_cloud_name}.clean_xyz")
    clean_new_path = os.path.join(point_cloud_result_dir, f"{point_cloud_name}_clean.xyz")
    if os.path.exists(clean_original_path):
        shutil.copyfile(clean_original_path, clean_new_path)

    final_path = os.path.join(point_cloud_result_dir, point_cloud_name + "_nzh.xyz")

    mean = [0, 0, 0]  # 均值
    cov = [[0.06, 0, 0],
           [0, 0.13, 0],
           [0, 0, 0.11]]  # 协方差矩阵

    mean2 = [0.52, 0.23, 0.56]
    conv2 = [[0.52, 0, 0],
             [0, 0.23, 0],
             [0, 0, 0.56]]

    points = np.loadtxt(clean_new_path)
    noise_points = points
    clean_points = None
    noise_points = add_anisotropic_gaussian_noise(points, mean, cov)
    # noise_points=add_normal_outliers(noise_points,mean2,conv2)
    noise_points = add_outliers_extra(noise_points, 0, 7, 0.5)
    # 0.7贼稀疏 0.2还好
    # noise_points,clean_points=add_outliers(points,noise_points,0,10,ratio=0.25)
    # 这里是为了把稀疏点云作为干净的点
    if clean_points is not None:
        np.savetxt(clean_new_path.replace("_clean.xyz", "_nzh.clean_xyz"), clean_points)
    np.savetxt(final_path, noise_points)

    # point_cloud_name="dragon100k_nzh"
    # path = os.path.join(dataset_base, point_cloud_name + ".clean_xyz")
    # path=generator_noise_point_cloud(path,add_noise)
    # generator_noise_point_cloud(point_cloud_name,add_outliers)
