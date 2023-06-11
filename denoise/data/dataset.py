import os
import platform
import random
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from denoise.data.pointcloud import PointCloudBase, Cache, PointCloudPatch, PointCloudFeature


class PointCloudPatchDataset(Dataset):
    def __init__(self,
                 point_cloud_name_path: str,
                 indir: str,
                 target_features: List[str],
                 points_per_patch: int,
                 use_pca: bool,
                 radius: List[float],
                 center: str,
                 point_tuple: int,
                 cache_capacity: int,
                 seed: int,
                 patch_sampling: str):
        self.point_tuple: int = point_tuple
        self.point_cloud_name_path: str = point_cloud_name_path
        self.indir = indir
        self.point_cloud_names: List[str] = self.get_point_cloud_list()
        # 点云的基本信息，常驻内存。但是数据就通过cache进行读取
        self.point_cloud_base_info: List[PointCloudBase] = []
        # 点云所需特征
        self.point_cloud_features: PointCloudFeature = PointCloudFeature(target_features, use_pca)
        self.radius: List[float] = radius
        # 把数据保存成为npy的形式,并且存储一些点云的基本信息
        self.save()
        self.pred_dim: int = self.point_cloud_features.dim

        self.cache: Cache = Cache(cache_capacity)
        self.points_per_patch: int = points_per_patch

        self.center: str = center
        self.rng: np.random.RandomState = np.random.RandomState(seed)
        self.patch_sampling = patch_sampling

    def get_point_cloud_list(self):
        # 因为文件上传到操作系统之后会存在命名问题
        strange_str = ""
        if "Linux" == platform.system():
            # strange_str="2%"
            strange_str = ""
        elif "Windows" == platform.system():
            strange_str = ""
        else:
            raise Exception("错误的操作系统")

        with open(self.point_cloud_name_path) as f:
            lines = f.readlines()
            point_cloud_names = [line.strip() + strange_str for line in lines]

            def need(line):
                return line is not None and line != strange_str

            point_cloud_names = list(filter(need, point_cloud_names))
        return point_cloud_names

    def outliers_generator_key_points(self, points, outliers, point_cloud_name):
        point_indices = np.array(range(len(points)))
        original_points = point_indices[outliers == 0]
        outliers_points = point_indices[outliers == 1]

        points_num = min(len(original_points), len(outliers_points))
        ratio = 1.0
        # 生成数量一致的原点云和离群点
        selected_original_points = random.sample(list(original_points), int(points_num * ratio))
        selected_outliers_points = random.sample(list(outliers_points), int(points_num * ratio))

        key_points = selected_original_points + selected_outliers_points
        random.shuffle(key_points)
        np.savetxt(os.path.join(self.indir, point_cloud_name + ".pidx"), key_points)

    def save_to_npy(self, point_cloud_name, file_suffix, file_type):
        file_name = os.path.join(self.indir, point_cloud_name + file_suffix)
        if not os.path.exists(file_name):
            return None

        # 切记 如果修改了读取点云的配置
        # 就要把这里改掉
        file_suffix_list = ['pidx']
        file_suffix_list = []
        # 这里第三个条件是我们全流程都要重新生成不然用的肯定是以前的结果
        if os.path.exists(file_name + ".npy") \
                and (file_suffix not in file_suffix_list) \
                and self.indir.find("whole_process") == -1:
            return np.load(file_name + ".npy")

        point_cloud_data_npy = np.loadtxt(file_name).astype(file_type)
        np.save(file_name + '.npy', point_cloud_data_npy)
        return point_cloud_data_npy

    def save(self):
        for index, point_cloud_name in enumerate(self.point_cloud_names):
            print(f"getting information from {point_cloud_name}...")
            points = self.save_to_npy(point_cloud_name, ".xyz", "float32")
            if points is None:
                raise ValueError(f"{self.indir}")
            outliers = None
            if self.point_cloud_features.include_normal:
                self.save_to_npy(point_cloud_name, ".normals", "float32")
            if self.point_cloud_features.include_outliers:
                outliers = self.save_to_npy(point_cloud_name, ".outliers", "float32")
            if self.point_cloud_features.include_curvatures:
                self.save_to_npy(point_cloud_name, ".curv", "float32")
            if self.point_cloud_features.include_clean_points:
                self.save_to_npy(point_cloud_name, ".clean_xyz", "float32")

            if self.point_cloud_features.include_key_points:
                patch_count = len(self.save_to_npy(point_cloud_name, ".pidx", "int32"))
            # # 要有pidx这个文件大家都会有，要是没有大家就都没有
            # if key_points_indices is not None:
            #     self.point_cloud_features.include_key_points=True
            #     patch_count = len(key_points_indices)
            # 如果存在离群点，但是本身又没有pidx的文件信息
            elif self.point_cloud_features.include_outliers and not self.point_cloud_features.include_key_points:
                self.outliers_generator_key_points(points, outliers, point_cloud_name)
                key_points_indices = self.save_to_npy(point_cloud_name, ".pidx", "int32")
                patch_count = len(key_points_indices)
            else:
                patch_count = points.shape[0]

            base_info = PointCloudBase(self.indir, index, point_cloud_name, patch_count, points, self.radius)
            self.point_cloud_base_info.append(base_info)

        if self.point_cloud_features.include_outliers:
            self.point_cloud_features.include_key_points = True

        print(f"data load finish... all point cloud are {len(self.point_cloud_names)}")

    def __len__(self):
        return sum([point_cloud_base.patch_count for point_cloud_base in self.point_cloud_base_info])

    def get_point_cloud_and_patch_by_dataset_index(self, index):
        begin = 0
        for i, item in enumerate(self.point_cloud_base_info):
            patch_count = item.patch_count
            if begin <= index < patch_count + begin:
                return i, index - begin

            begin += patch_count

        raise ValueError(f"不存在这样的一个index坐标！{index}")

    def __getitem__(self, index):
        # 拿到点云id和点云片id
        id, patch_id = self.get_point_cloud_and_patch_by_dataset_index(index)

        point_cloud = self.cache.get(self.point_cloud_base_info[id])
        if self.point_cloud_features.include_key_points:
            # 如果存在关键点，那么取到的是关键点的索引
            center = point_cloud.key_points[patch_id]
        else:
            center = patch_id

        patch_radii = point_cloud.base_info.patch_radii

        final_point_cloud_patch_list = []
        for k, patch_radius in enumerate(patch_radii):
            center_point = point_cloud.points[center]
            point_cloud_patch = self.get_point_cloud_patch_by_center(point_cloud.points,
                                                                     point_cloud.kdtree,
                                                                     center_point,
                                                                     patch_radius)
            final_point_cloud_patch_list.append(point_cloud_patch)

        final_point_cloud_patch_features_list = []

        if self.point_cloud_features.include_normal:
            patch_normal = point_cloud.normals[center]
            final_point_cloud_patch_features_list.append(patch_normal)

        if self.point_cloud_features.include_curvatures:
            patch_curvature = point_cloud.curvature[center]
            # 这里存在疑惑为什么要取第0个半径
            patch_curvature = patch_curvature * patch_radii[0]
            if self.point_cloud_features.include_max_curvatures:
                final_point_cloud_patch_features_list.append(patch_curvature[0:1])
            if self.point_cloud_features.include_min_curvatures:
                final_point_cloud_patch_features_list.append(patch_curvature[1:2])

        if self.point_cloud_features.include_original:
            original = point_cloud.points[center]
            final_point_cloud_patch_features_list.append(original)

        # 注意 这里是500行的数据不能和前面的整合在一起
        if self.point_cloud_features.include_clean_points:
            center_point = point_cloud.points[center]
            clean_patch_radius = max(patch_radii)
            point_cloud_clean_patch = self.get_point_cloud_patch_by_center(point_cloud.clean_points,
                                                                           point_cloud.clean_kdtree,
                                                                           center_point,
                                                                           clean_patch_radius)
            final_point_cloud_patch_features_list.append(point_cloud_clean_patch.points)
        if self.point_cloud_features.include_outliers:
            outlier = point_cloud.outliers[center]
            # 这里一个数字要转化成一个数组才行
            outlier = np.asarray([outlier])
            final_point_cloud_patch_features_list.append(outlier)

        # 一个点为中心的领域块集合成一个[n,3]列的矩阵，其中n={patch_count1+patch_count2]
        # patch_count1是第一个半径下的点的数量依次类推,那么自然原论文是n=500+500+500,因为要保证整齐划一
        # shape[n,3]

        final_patch_points = torch.from_numpy(
            np.concatenate([patch.points for patch in final_point_cloud_patch_list], axis=0))

        if self.point_cloud_features.use_pca:
            # 找到有效的数据下标索引
            valid_indices = []
            start = 0
            for patch in final_point_cloud_patch_list:
                points_count = patch.points_count
                valid_indices = np.append(valid_indices, np.arange(start, start + points_count))
                start = start + self.points_per_patch

            points_mean = final_patch_points[valid_indices].mean(0)
            final_patch_points[valid_indices] = final_patch_points[valid_indices] - points_mean
            trans, _, _ = torch.svd(torch.t(final_patch_points[valid_indices]))
            final_patch_points[valid_indices] = torch.mm(final_patch_points[valid_indices], trans)

            cp_new = -points_mean
            cp_new = torch.matmul(cp_new, trans)

            final_patch_points[valid_indices] = final_patch_points[valid_indices] - cp_new

            if self.point_cloud_features.include_normal:
                patch_normal = torch.matmul(point_cloud.normals, trans)
                # 之所以要放在这里是因为pca之后法向量会发生改变
                final_point_cloud_patch_features_list[0] = patch_normal
        else:
            trans = torch.eye(3).float()

        # self.point_tuple 是 PointNet 网络中的一个超参数，表示每个点要和多少个邻居点一起作为一个元组(tuple)进行处理。
        # 这里将邻居点放在一起组成一个元组，是为了在后续的操作中，将点的位置信息和邻居点的位置信息一起作为输入送入神经网络中。

        # 从已经存在的点云块里面继续操作point_tuple,原本是一组也就是一行一个点，现在是一组point_tuple个点
        if self.point_tuple > 1:
            new_final_point_cloud_patch_list = []
            for m in range(len(patch_radii)):
                # 转换成ndarray类型
                final_point_cloud_patch_points = np.array([patch.points for patch in final_point_cloud_patch_list])
                temp_patch_points_list = []

                for t in range(self.point_tuple):
                    # np.random.shuffle(base_points_coordinate_indices)
                    np.random.shuffle(final_point_cloud_patch_points)
                    # 在列的维度拼接
                    temp_patch_points_list.append(final_point_cloud_patch_points)

                # if point_tuple=3 把[500,3] [500,3] [500,3]的向量合并为[500,9]
                temp_patch_points = np.concatenate(temp_patch_points_list, axis=1)
                new_final_point_cloud_patch_list.append(temp_patch_points)
            # if len(patch_radii)=2 把[500,9] [500,9]的向量合并成 [1000,9]
            final_patch_points = torch.from_numpy(np.concatenate(new_final_point_cloud_patch_list, axis=0))

        patch_radii = torch.tensor(patch_radii)
        # 把[1,1] [1,3] [1,3]的矩阵合成[1,7]
        return (final_patch_points,) + (final_point_cloud_patch_features_list,) + (trans,) + (patch_radii,)

    def get_points_patch_indices(self, kdtree, center_point, patch_radius):
        if self.patch_sampling == "point":
            # 查找最近的500个点
            dists, points_patch_indices = kdtree.query(center_point, self.points_per_patch)
            points_patch_indices = np.array(points_patch_indices)
            # 后面要归一化，所以必须这样子操作
            patch_radius = np.max(dists)
            return points_patch_indices, patch_radius
        elif self.patch_sampling == "radius":
            # 使用算法选出按照某个半径的点，形成领域
            points_patch_indices = np.array(kdtree.query_ball_point(center_point, patch_radius))
            return points_patch_indices, patch_radius
        else:
            raise ValueError("没有这种采样方法")

    def get_point_cloud_patch_by_center(self, points, kdtree, center_point, patch_radius: float):

        points_patch_indices, patch_radius = self.get_points_patch_indices(kdtree, center_point, patch_radius)

        final_points_per_patch = min(self.points_per_patch, len(points_patch_indices))

        # 设置点数标准差,无所谓就是增加一点随机性
        point_count_std = 0
        if point_count_std > 0:
            final_points_per_patch = max(5, round(final_points_per_patch * self.rng.uniform(1.0 - point_count_std * 2)))
            final_points_per_patch = min(final_points_per_patch, len(points_patch_indices))

        if final_points_per_patch < len(points_patch_indices):
            points_patch_indices = points_patch_indices[
                self.rng.choice(len(points_patch_indices), final_points_per_patch, replace=False)]

        # 周围可能没有干净的点
        if points_patch_indices is None or len(points_patch_indices) == 0:
            points_patch = np.empty(shape=[0, 3])
        else:
            points_patch = points[points_patch_indices, :]

        points_count = points_patch.shape[0]
        # 这里只对有效点取平均，所以要把补充点的操作放在这个的后面
        if self.center == 'mean':
            points_patch = points_patch - points_patch.mean(0)
        elif self.center == 'point':
            points_patch = points_patch - center_point
        elif self.center == 'none':
            pass
        else:
            raise ValueError('Unknown patch centering option: %s' % self.center)

        points_patch_len = points_patch.shape[0]
        points_patch_dim = points_patch.shape[1]
        # 只有半径的算法会出现这种情况
        if points_patch_len < self.points_per_patch:
            center_arr = np.zeros([self.points_per_patch - points_patch_len, points_patch_dim], dtype=np.float32)
            points_patch = np.concatenate((points_patch, center_arr))

        # 因为之前已经是取了一定长度作为领域半径（每个点云的固定比例，比如100*0.1，200*0.2）
        # 但是需要最后面都化领域半径为1，所以需要除以这个领域半径
        points_patch = points_patch / patch_radius

        return PointCloudPatch(radius=patch_radius, points=points_patch, points_count=points_count,
                               center_point=center_point)
