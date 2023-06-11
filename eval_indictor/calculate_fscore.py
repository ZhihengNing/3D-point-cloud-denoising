import os

import numpy as np

from eval_indictor.util import dataset_base


def remove_outliers(point_cloud_name):
    outliers_path = os.path.join(dataset_base, point_cloud_name, point_cloud_name + ".outliers")
    outliers_indices = np.loadtxt(outliers_path)
    point_cloud_path = os.path.join(dataset_base, point_cloud_name, point_cloud_name + ".xyz")
    point_cloud = np.loadtxt(point_cloud_path)
    clean_point_cloud = point_cloud[outliers_indices == 0]
    np.savetxt(os.path.join(dataset_base, point_cloud_name, point_cloud_name + "_clean.xyz"), clean_point_cloud)


if __name__ == '__main__':
    remove_outliers("star_smooth100k_noise_white_5.00e-03_outliers_gaussian")
