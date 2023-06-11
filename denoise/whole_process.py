import os
import shutil
from abc import abstractmethod

import torch

from denoise.common.core import PointCleanNet, DEVICE
from denoise.noise_removal.eval_config import NoiseRemovalEvalConfig
from denoise.noise_removal.model.respcpnet import NoiseRemovalResPCPNet
from denoise.outliers_removal.eval_config import OutliersRemovalEvalConfig
from denoise.outliers_removal.model.respcpnet import OutliersRemovalResPCPNet
from denoise.process.preprocess import WholeProcess
from denoise.process.process import DataProcess
from denoise.utils.device_util import state_dict_process
from denoise.utils.file_util import write_file, read_file, search_file_in_dir
from denoise.utils.point_cloud_util import get_mean_displacement
from denoise.utils.seed_util import set_seed
from environments import BASE_PATH

RESULT_DIR = os.path.join(BASE_PATH, "train_result", "whole_process")
Experiment_Noise_Removal_Txt = os.path.join(BASE_PATH, "data", "experiment_noise_removal.txt")
Experiment_Outliers_Removal_Txt = os.path.join(BASE_PATH, "data", "experiment_outliers_removal.txt")


def whole_process(train_opt, eval_opt, data_process_type):
    process = WholeProcess(eval_opt, train_opt)
    if data_process_type == PointCleanNet.OUTLIERS_REMOVAL:
        net = OutliersRemovalResPCPNet(
            num_points=train_opt.points_per_patch,
            output_dim=1,
            use_point_stn=train_opt.use_point_stn,
            use_feat_stn=train_opt.use_feat_stn,
            sym_op=train_opt.sym_op,
            point_tuple=train_opt.point_tuple)
    elif data_process_type == PointCleanNet.NOISE_REMOVAL:
        net = NoiseRemovalResPCPNet(
            num_points=train_opt.points_per_patch,
            output_dim=3,
            use_point_stn=train_opt.use_point_stn,
            use_feat_stn=train_opt.use_feat_stn,
            sym_op=train_opt.sym_op,
            point_tuple=train_opt.point_tuple)
    else:
        raise ValueError("eval期间错误的net类型")

    state_dict = torch.load(eval_opt.model_filename)
    new_state_dict = state_dict_process(state_dict)
    net.load_state_dict(new_state_dict)
    net.to(DEVICE)

    data_process = DataProcess(net, process)

    result = data_process.whole_process_epoch_process()
    predict, transpose, target, trans, patch_radii = result

    predict = torch.cat(predict, dim=0).cpu()
    transpose = torch.cat(transpose, dim=0).cpu()
    trans = torch.cat(trans, dim=0).cpu()
    target = [torch.cat([target[i][j] for i in range(len(target))], dim=0).cpu() for j in range(len(target[0]))]
    patch_radii = torch.cat(patch_radii, dim=0).cpu()

    return predict, transpose, target, trans, patch_radii


class NoiseRemoval:
    def __init__(self, point_cloud_result_dir: str):
        self.point_cloud_result_dir: str = point_cloud_result_dir

    @abstractmethod
    def eval_opt_init(self):
        raise NotImplementedError

    @abstractmethod
    def train_opt_init(self):
        raise NotImplementedError

    @abstractmethod
    def process(self):
        raise NotImplementedError


class OutlierPointRemoval(NoiseRemoval):
    # indir，记录数据来自于训练集，测试集，还是验证集
    def __init__(self, point_cloud_result_dir: str, in_dir):
        super().__init__(point_cloud_result_dir)
        self.in_dir: str = in_dir
        self.config, self.eval_opt = self.eval_opt_init()
        self.train_opt = self.train_opt_init()

    def eval_opt_init(self):
        config = OutliersRemovalEvalConfig()
        eval_opt = self.config.get_config()
        eval_opt.testset = Experiment_Outliers_Removal_Txt
        eval_opt.indir = self.point_cloud_result_dir
        eval_opt.outputs = ['original']
        set_seed(eval_opt.seed)
        return config, eval_opt

    def train_opt_init(self):
        train_opt = torch.load(self.eval_opt.params_filename)
        # 离群点去除的取样方式
        train_opt.patch_sampling = 'point'
        train_opt.nepoch = 1
        return train_opt

    def process(self):
        predict, _, target, trans, _ = \
            whole_process(self.train_opt, self.eval_opt, PointCleanNet.OUTLIERS_REMOVAL)
        original = target[0]
        outliers_removal_point_cloud = []

        for index, item in enumerate(predict):
            if item[0] < 0.5:
                outliers_removal_point_cloud.append(original[index].numpy())

        path = os.path.join(self.point_cloud_result_dir, "outliers_removal_result.xyz")
        write_file(path, outliers_removal_point_cloud)

        info = f"original {len(original)}: predict {len(outliers_removal_point_cloud)}"
        print(info)

        basic_Info_file = os.path.join(self.point_cloud_result_dir, "info.txt")
        with open(basic_Info_file, "w") as f:
            f.write(self.in_dir + "\n")
            f.write(info)
        print("outliers remove finish...")
        print("==========================")


class NoisePointRemoval(NoiseRemoval):

    def __init__(self, point_cloud_result_dir: str, in_dir: str):
        super().__init__(point_cloud_result_dir)
        self.in_dir = in_dir
        self.config, self.eval_opt = self.eval_opt_init()
        self.train_opt = self.train_opt_init()

    def eval_opt_init(self):
        config = NoiseRemovalEvalConfig()
        eval_opt = config.get_config()
        # 如果是全流程，直接操作需要的内容即可
        eval_opt.testset = Experiment_Noise_Removal_Txt
        eval_opt.indir = self.point_cloud_result_dir
        eval_opt.outputs = ['original']

        set_seed(eval_opt.seed)
        return config, eval_opt

    def train_opt_init(self):
        train_opt = torch.load(self.eval_opt.params_filename)
        # 这里真的懒得改了 涉及类的设计
        train_opt.nepoch = 1
        train_opt.patch_sampling = 'radius'
        return train_opt

    def process(self):
        predict, transpose, target, trans, patch_radii = \
            whole_process(self.train_opt, self.eval_opt, PointCleanNet.NOISE_REMOVAL)
        original = target[0]
        patch_radii = patch_radii[0]
        o_pred = predict.clone()

        # 这两步是为了回归原始点云
        if self.train_opt.use_point_stn:
            # 使用了qstn进行旋转，我们需要进行逆运算
            o_pred[:, :] = torch.bmm(o_pred.unsqueeze(1), transpose.transpose(2, 1)).squeeze(1)
        if self.train_opt.use_pca:
            # 使用了pca进行操作我们需要使用逆运算
            o_pred[:, :] = torch.bmm(o_pred.unsqueeze(1), trans.transpose(2, 1)).squeeze(1)

        n_points = patch_radii.shape[0]
        # new coordinates are : old coordiantes + displacement vector
        o_pred = torch.mul(o_pred, torch.t(patch_radii.expand(3, n_points)).float()) + original
        predict = o_pred
        predict = get_mean_displacement(original, predict).cpu().numpy()

        path = os.path.join(self.point_cloud_result_dir, "noise_removal_result.xyz")
        write_file(path, predict)
        print(f"original {len(original)}: predict {len(predict)}")
        print("noise remove finish...")


def generator_base_info(point_cloud_data_file_path):
    point_cloud_name = read_file(point_cloud_data_file_path)[0]
    point_cloud_result_dir = os.path.join(RESULT_DIR, point_cloud_name)
    if not os.path.exists(point_cloud_result_dir):
        os.makedirs(point_cloud_result_dir)
    in_dir = search_file_in_dir(os.path.join(BASE_PATH, "data"), point_cloud_name + ".xyz")

    if in_dir is None:
        raise ValueError('找不到文件')
    if not os.path.exists(point_cloud_result_dir):
        os.makedirs(point_cloud_result_dir)

    return in_dir, point_cloud_result_dir, point_cloud_name


def copy_original_point_cloud(in_dir, point_cloud_result_dir, point_cloud_name):
    original_path = os.path.join(in_dir, f"{point_cloud_name}.xyz")
    new_path = os.path.join(point_cloud_result_dir, f"{point_cloud_name}.xyz")
    shutil.copyfile(original_path, new_path)


def copy_clean_point_cloud(in_dir, point_cloud_result_dir, point_cloud_name):
    clean_original_path = os.path.join(in_dir, f"{point_cloud_name}.clean_xyz")
    clean_new_path = os.path.join(point_cloud_result_dir, f"{point_cloud_name}_clean.xyz")
    if os.path.exists(clean_original_path):
        shutil.copyfile(clean_original_path, clean_new_path)


def copy_outliers_point_cloud(in_dir, point_cloud_result_dir, point_cloud_name):
    outliers_original_path = os.path.join(in_dir, f"{point_cloud_name}.outliers")
    outliers_new_path = os.path.join(point_cloud_result_dir, f"{point_cloud_name}.outliers")
    if os.path.exists(outliers_original_path):
        shutil.copyfile(outliers_original_path, outliers_new_path)


def outliers():
    in_dir, point_cloud_result_dir, point_cloud_name = \
        generator_base_info(Experiment_Outliers_Removal_Txt)

    copy_original_point_cloud(in_dir, point_cloud_result_dir, point_cloud_name)
    copy_outliers_point_cloud(in_dir, point_cloud_result_dir, point_cloud_name)

    opr = OutlierPointRemoval(point_cloud_result_dir, in_dir)
    opr.process()


def noise():
    in_dir, point_cloud_result_dir, point_cloud_name = generator_base_info(Experiment_Noise_Removal_Txt)

    copy_original_point_cloud(in_dir, point_cloud_result_dir, point_cloud_name)
    copy_clean_point_cloud(in_dir, point_cloud_result_dir, point_cloud_name)

    npr = NoisePointRemoval(point_cloud_result_dir, in_dir)
    npr.process()


# 进行全过程的时候，需要在experiment_noise_removal.txt文件内写入outliers_removal_result
def whole():
    in_dir, point_cloud_result_dir, point_cloud_name = \
        generator_base_info(Experiment_Outliers_Removal_Txt)

    copy_original_point_cloud(in_dir, point_cloud_result_dir, point_cloud_name)
    copy_clean_point_cloud(in_dir, point_cloud_result_dir, point_cloud_name)
    copy_outliers_point_cloud(in_dir, point_cloud_result_dir, point_cloud_name)

    opr = OutlierPointRemoval(point_cloud_result_dir, in_dir)
    opr.process()
    write_file(Experiment_Noise_Removal_Txt, "outliers_removal_result")
    npr = NoisePointRemoval(point_cloud_result_dir, in_dir)
    npr.process()


if __name__ == '__main__':
    whole()
