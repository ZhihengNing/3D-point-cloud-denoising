import numpy as np
import torch.utils.data as data

from denoise.data.dataset import PointCloudPatchDataset


# 顺序读取点云片
class SequentialPatchSampler(data.sampler.Sampler):
    def __init__(self, data_source: PointCloudPatchDataset):
        super().__init__(data_source)
        self.data_source = data_source
        self.total_patch_count = self.data_source.__len__()

    def __iter__(self):
        return iter(range(self.total_patch_count))

    def __len__(self):
        return self.total_patch_count


# 在点云位置按顺序的情况下，点云里面的点云片位置随机，并且不会完全取完
class SequentialPointCloudRandomPatchSampler(data.sampler.Sampler):

    def __init__(self, data_source: PointCloudPatchDataset, seed: int, patches_per_point_cloud: int):
        super().__init__(data_source)
        self.data_source = data_source
        self.patches_per_point_cloud = patches_per_point_cloud
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.total_patch_count = 0

        for item in self.data_source.point_cloud_base_info:
            self.total_patch_count = self.total_patch_count + min(patches_per_point_cloud, item.patch_count)

    def __iter__(self):

        all_point_cloud_patch_indices = []
        begin = 0
        for item in self.data_source.point_cloud_base_info:
            patch_count = item.patch_count
            per_point_cloud_patches_indices = self.rng.choice \
                (patch_count, size=min(self.patches_per_point_cloud, patch_count), replace=False)
            all_point_cloud_patch_indices.extend(begin + per_point_cloud_patches_indices)
            begin = begin + patch_count

        return iter(all_point_cloud_patch_indices)

    def __len__(self):
        return self.total_patch_count


# 把所有点云片的顺序全部打乱，并且一般不会全部取完
class RandomPatchSampler(data.sampler.Sampler):
    def __init__(self, data_source: PointCloudPatchDataset, seed: int, patches_per_point_cloud: int):
        super().__init__(data_source)
        self.data_source = data_source
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)
        self.total_patch_count = 0
        for item in self.data_source.point_cloud_base_info:
            self.total_patch_count = self.total_patch_count + min(patches_per_point_cloud, item.patch_count)

    def __iter__(self):
        return iter(self.rng.choice(self.data_source.__len__(), size=self.total_patch_count, replace=False))

    def __len__(self):
        return self.total_patch_count
