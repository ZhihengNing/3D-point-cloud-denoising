import os
import sys
from collections import OrderedDict
from typing import List

import numpy as np
import torch.nn
from scipy import spatial

from denoise.noise_removal.loss import noise_loss2
from denoise.outliers_removal.loss import outliers_loss_recall_plus


class PointCloudBase:
    # radius是相对于点云尺寸的比例作为球形领域的半径
    def __init__(self,
                 indir: str,
                 point_cloud_id: int,
                 point_cloud_name: str,
                 patch_count: int,
                 points,
                 radius: List[float]):
        self.indir = indir
        self.id: int = point_cloud_id
        self.name: str = point_cloud_name
        # 点云块的数量
        self.patch_count: int = patch_count
        # 点云里面点的数量
        self.points_count: int = points.shape[0]
        # 点云对角线长度
        self.diagonal_length: float = float(np.linalg.norm(points.max(0) - points.min(0), 2))
        # 点云块的半径, 希望获取的是
        self.patch_radii: List[float] = [self.diagonal_length * x for x in radius]


class PointCloudFeature:
    # Feature_Map = ["normal", "max_curvature", "min_curvature", "clean_points", "original", "outliers"]
    # Dim_Map = [3, 1, 1, 3, 1, 1]
    # Weight_Map = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # 点云特征（包括法线曲率什么的） /是否包含关键点，关键点要作为点云中心的（否则每一个点都有可能作为点云中心）/是否使用pca
    def __init__(self, patch_features: List[str], use_pca: bool):
        self.feature_Map = {
            "normal": {
                "dim": 3,
                "weight": 1.0,
                "index": -1,
                "begin_position": -1,
                "end_position": -1,
                "criterion": torch.nn.L1Loss()
            },
            "max_curvature": {
                "dim": 1,
                "weight": 1.0,
                "index": -1,
                "begin_position": -1,
                "end_position": -1,
                "criterion": torch.nn.L1Loss()
            },
            "min_curvature": {
                "dim": 1,
                "weight": 1.0,
                "index": -1,
                "begin_position": -1,
                "end_position": -1,
                "criterion": torch.nn.L1Loss()
            },
            "clean_points": {
                "dim": 3,
                "weight": 1.0,
                "index": -1,
                "begin_position": -1,
                "end_position": -1,
                "criterion": noise_loss2
            },
            "original": {
                "dim": 3,
                "weight": 1.0,
                "index": -1,
                "begin_position": -1,
                "end_position": -1,
                "criterion": torch.nn.L1Loss()
            },
            "outliers": {
                "dim": 1,
                "weight": 1.0,
                "index": -1,
                "begin_position": -1,
                "end_position": -1,
                "criterion": outliers_loss_recall_plus
            }
        }

        # 特征占的总列数
        self.dim = 0

        self.include_normal = False
        self.include_curvatures = False
        self.include_max_curvatures = False
        self.include_min_curvatures = False
        self.include_clean_points = False
        self.include_original = False
        self.include_outliers = False

        for index, feature in enumerate(patch_features):
            if feature == "normal":
                self.include_normal = True
                self.generator_feature_record("normal", index)
            elif feature == "max_curvature" or feature == "min_curvature":
                self.include_curvatures = True
                # 因为最大曲率和最小曲率是存在一个文件里面的，有曲率就说明最大曲率和最小曲率都存在
                if feature == "max_curvature":
                    self.include_max_curvatures = True
                    self.generator_feature_record("max_curvature", index)
                else:
                    self.include_min_curvatures = True
                    self.generator_feature_record("min_curvature", index)
            elif feature == "clean_points":
                self.include_clean_points = True
                self.generator_feature_record("clean_points", index)
            elif feature == "original":
                self.include_original = True
                # 疑惑
                self.generator_feature_record("original", index)
            elif feature == "outliers":
                self.include_outliers = True
                self.generator_feature_record("outliers", index)
            else:
                raise ValueError(f"error feature map {feature}")
        self.include_key_points = False
        self.use_pca = use_pca

    def generator_feature_record(self, feature_name, feature_index):
        self.feature_Map[feature_name]["index"] = feature_index
        self.feature_Map[feature_name]["begin_position"] = self.dim
        self.dim += self.feature_Map[feature_name]["dim"]
        self.feature_Map[feature_name]["end_position"] = self.dim


class PointCloud:
    def __init__(self, base_info: PointCloudBase):
        self.base_info: PointCloudBase = base_info
        self.points = self.load_point_cloud_data_xyz()
        self.clean_points = self.load_point_cloud_data_clean_xyz()
        self.outliers = self.load_point_cloud_data_outliers()
        self.curvature = self.load_point_cloud_data_curvature()
        self.normals = self.load_point_cloud_data_normals()
        # shape [n,]
        self.key_points = self.load_point_cloud_data_key_points()

        sys.setrecursionlimit(int(max(1000, round(self.points.shape[0] / 10))))
        # otherwise KDTree construction may run out of recursions
        # 找到某个集合的任何一个点距离最近的10个点
        self.kdtree = spatial.cKDTree(self.points, 10)
        # 保证有数据才能进行下面的操作
        if self.clean_points is not None:
            self.clean_kdtree = spatial.cKDTree(self.clean_points, 10)

    def load_point_cloud_data(self, file_suffix):
        point_cloud_data_path = os.path.join(self.base_info.indir, self.base_info.name + file_suffix + ".npy")
        if os.path.exists(point_cloud_data_path):
            point_cloud_data = np.load(point_cloud_data_path)
        else:
            point_cloud_data = None
            # raise ValueError(f"this shape({self.point_cloud_name} has no attribute like {file_suffix}")
        return point_cloud_data

    def load_point_cloud_data_xyz(self):
        points_xyz = self.load_point_cloud_data(".xyz")
        if points_xyz is None:
            raise ValueError(f"this point_cloud({self.base_info.name}) has no attribute like .xyz")
        return points_xyz

    def load_point_cloud_data_clean_xyz(self):
        return self.load_point_cloud_data(".clean_xyz")

    def load_point_cloud_data_curvature(self):
        return self.load_point_cloud_data(".curv")

    def load_point_cloud_data_normals(self):
        return self.load_point_cloud_data(".normals")

    def load_point_cloud_data_outliers(self):
        return self.load_point_cloud_data(".outliers")

    def load_point_cloud_data_key_points(self):
        return self.load_point_cloud_data(".pidx")


# 只有在取的时候才加载
class Cache:
    def __init__(self, capacity: int):
        self.point_cloud_set: OrderedDict = OrderedDict()
        self.capacity: int = capacity

    def get(self, point_cloud_base_info: PointCloudBase) -> PointCloud:
        point_cloud_index = point_cloud_base_info.id
        if point_cloud_index not in self.point_cloud_set.keys():
            if len(self.point_cloud_set) >= self.capacity:
                k, v = self.point_cloud_set.popitem(last=False)
                del k, v
            # 不在缓存里面才会创建对象，把文件里面的数据读入内存中
            point_cloud = PointCloud(point_cloud_base_info)
            self.point_cloud_set[point_cloud_index] = point_cloud
        return self.point_cloud_set[point_cloud_index]


class PointCloudPatch:
    def __init__(self, radius: float, points, points_count, center_point):
        self.radius = radius
        self.points = points
        self.points_count = points_count
        self.center_point = center_point

