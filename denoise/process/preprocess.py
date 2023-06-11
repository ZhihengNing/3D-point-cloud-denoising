from abc import abstractmethod

from torch.utils import data
from torch.utils.data import DataLoader

from denoise.data.datasampler import RandomPatchSampler, SequentialPointCloudRandomPatchSampler, SequentialPatchSampler
from denoise.data.dataset import PointCloudPatchDataset


class Origin:
    def __init__(self,opt):
        self.opt = opt
        self._dataset = self.set_dataset()
        self._dataset_sampler = self.set_dataset_sampler()
        self._data_loader = self.set_data_loader()

    @abstractmethod
    def set_dataset(self) -> PointCloudPatchDataset:
        raise NotImplementedError

    @abstractmethod
    def set_dataset_sampler(self) -> data.sampler.Sampler:
        raise NotImplementedError

    @abstractmethod
    def set_data_loader(self) -> DataLoader:
        raise NotImplementedError

    @property
    def dataset(self) -> PointCloudPatchDataset:
        return self._dataset


    @property
    def dataset_sampler(self) -> data.sampler.Sampler:
        return self._dataset_sampler

    @property
    def data_loader(self) -> DataLoader:
        return self._data_loader


class Train(Origin):

    def __init__(self, opt):
        super().__init__(opt)

    def set_dataset(self):
        return PointCloudPatchDataset(
            point_cloud_name_path=self.opt.trainset,
            indir=self.opt.indir,
            target_features=self.opt.outputs,
            points_per_patch=self.opt.points_per_patch,
            use_pca=self.opt.use_pca,
            radius=self.opt.patch_radius,
            center=self.opt.patch_center,
            point_tuple=self.opt.point_tuple,
            cache_capacity=self.opt.cache_capacity,
            seed=self.opt.seed,
            patch_sampling=self.opt.patch_sampling)

    def set_dataset_sampler(self) -> data.sampler.Sampler:
        if self.opt.training_order=='sequential':
            return SequentialPatchSampler(data_source=self.dataset)
        elif self.opt.training_order == 'random':
            return RandomPatchSampler(
                data_source=self.dataset,
                seed=self.opt.seed,
                patches_per_point_cloud=self.opt.patches_per_point_cloud)
        elif self.opt.training_order == 'sequential_point_cloud_random_patches':
            return SequentialPointCloudRandomPatchSampler(
                data_source=self.dataset,
                seed=self.opt.seed,
                patches_per_point_cloud=self.opt.patches_per_point_cloud)
        else:
            raise ValueError('Unknown training order: %s' % self.opt.training_order)

    def set_data_loader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset,
                          sampler=self.dataset_sampler,
                          batch_size=self.opt.batchSize,
                          num_workers=int(self.opt.workers))


class Valid(Origin):

    def __init__(self, opt):
        super().__init__(opt)

    def set_dataset(self) -> PointCloudPatchDataset:
        return PointCloudPatchDataset(
            point_cloud_name_path=self.opt.validateset,
            indir=self.opt.indir,
            target_features=self.opt.outputs,
            points_per_patch=self.opt.points_per_patch,
            use_pca=self.opt.use_pca,
            radius=self.opt.patch_radius,
            center=self.opt.patch_center,
            point_tuple=self.opt.point_tuple,
            cache_capacity=self.opt.cache_capacity,
            seed=self.opt.seed,
            patch_sampling=self.opt.patch_sampling)

    def set_dataset_sampler(self) -> data.sampler.Sampler:
        if self.opt.training_order=='sequential':
            return SequentialPatchSampler(data_source=self.dataset)
        elif self.opt.training_order == 'random':
            return RandomPatchSampler(
                data_source=self.dataset,
                seed=self.opt.seed,
                patches_per_point_cloud=self.opt.patches_per_point_cloud)
        elif self.opt.training_order == 'sequential_point_cloud_random_patches':
            return SequentialPointCloudRandomPatchSampler(
                data_source=self.dataset,
                seed=self.opt.seed,
                patches_per_point_cloud=self.opt.patches_per_point_cloud)
        else:
            raise ValueError('Unknown training order: %s' % self.opt.training_order)

    def set_data_loader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset,
                          sampler=self.dataset_sampler,
                          batch_size=self.opt.batchSize,
                          num_workers=int(self.opt.workers))


class Test(Origin):
    def __init__(self, eval_opt,train_opt):
        self.eval_opt=eval_opt
        super().__init__(train_opt)

    def set_dataset(self) -> PointCloudPatchDataset:
        return PointCloudPatchDataset(
            point_cloud_name_path=self.eval_opt.testset,
            indir=self.eval_opt.indir,
            target_features=self.opt.outputs,
            points_per_patch=self.opt.points_per_patch,
            use_pca=self.opt.use_pca,
            radius=self.opt.patch_radius,
            center=self.opt.patch_center,
            point_tuple=self.opt.point_tuple,
            cache_capacity=self.opt.cache_capacity,
            seed=self.eval_opt.seed,
            patch_sampling=self.opt.patch_sampling)

    def set_dataset_sampler(self) -> data.sampler.Sampler:
        if self.eval_opt.sampling=='full':
            return SequentialPatchSampler(data_source=self.dataset)
        elif self.eval_opt.sampling == 'random':
            return RandomPatchSampler(
                data_source=self.dataset,
                seed=self.eval_opt.seed,
                patches_per_point_cloud=self.eval_opt.patches_per_point_cloud)
        elif self.eval_opt.sampling == 'sequential_point_cloud_random_patches':
            return SequentialPointCloudRandomPatchSampler(
                data_source=self.dataset,
                seed=self.eval_opt.seed,
                patches_per_point_cloud=self.eval_opt.patches_per_point_cloud)
        else:
            raise ValueError('Unknown training order: %s' % self.eval_opt.sampling)

    def set_data_loader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset,
                          sampler=self.dataset_sampler,
                          batch_size=self.eval_opt.batchSize,
                          num_workers=int(self.eval_opt.workers))


class WholeProcess(Origin):

    def __init__(self,eval_opt,train_opt):
        self.eval_opt=eval_opt
        super().__init__(train_opt)

    def set_dataset(self) -> PointCloudPatchDataset:
        return PointCloudPatchDataset(
            point_cloud_name_path=self.eval_opt.testset,
            indir=self.eval_opt.indir,
            target_features=self.eval_opt.outputs, #注意因为这里是全流程了
            points_per_patch=self.opt.points_per_patch,
            use_pca=self.opt.use_pca,
            radius=self.opt.patch_radius,
            center=self.opt.patch_center,
            point_tuple=self.opt.point_tuple,
            cache_capacity=self.opt.cache_capacity,
            seed=self.eval_opt.seed,
            patch_sampling=self.opt.patch_sampling)

    def set_dataset_sampler(self) -> data.sampler.Sampler:
        return SequentialPatchSampler(data_source=self.dataset)

    def set_data_loader(self) -> DataLoader:
        return DataLoader(dataset=self.dataset,
                          sampler=self.dataset_sampler,
                          batch_size=self.eval_opt.batchSize,
                          num_workers=int(self.eval_opt.workers))