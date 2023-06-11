import argparse
import os
import sys

import math
import torch

from denoise.common.core import PointCleanNet, DEVICE
from denoise.noise_removal.model.respcpnet import NoiseRemovalResPCPNet
from denoise.noise_removal.train_config import NoiseRemovalTrainConfig
from denoise.outliers_removal.model.respcpnet import OutliersRemovalResPCPNet
from denoise.outliers_removal.train_config import OutliersRemovalTrainConfig
from denoise.process.preprocess import Train, Valid
from denoise.process.process import DataProcess
from denoise.utils.seed_util import set_seed


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", default="noise_removal", type=str)
    return parser.parse_args()


def train_model(data_process_type):
    if data_process_type == PointCleanNet.OUTLIERS_REMOVAL:
        config = OutliersRemovalTrainConfig()
    elif data_process_type == PointCleanNet.NOISE_REMOVAL:
        config = NoiseRemovalTrainConfig()
    else:
        raise ValueError("训练期间错误的opt类型")

    opt = config.get_config()
    set_seed(opt.seed)
    train = Train(opt)
    valid = Valid(opt)
    output_dim = train.dataset.pred_dim

    if data_process_type == PointCleanNet.OUTLIERS_REMOVAL:
        net = OutliersRemovalResPCPNet(
            num_points=opt.points_per_patch,
            output_dim=output_dim,
            use_point_stn=opt.use_point_stn,
            use_feat_stn=opt.use_feat_stn,
            sym_op=opt.sym_op,
            point_tuple=opt.point_tuple)
    elif data_process_type == PointCleanNet.NOISE_REMOVAL:
        net = NoiseRemovalResPCPNet(
            num_points=opt.points_per_patch,
            output_dim=output_dim,
            use_point_stn=opt.use_point_stn,
            use_feat_stn=opt.use_feat_stn,
            sym_op=opt.sym_op,
            point_tuple=opt.point_tuple)
    else:
        raise ValueError("错误的net类型")
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.to(DEVICE)

    train_data_process = DataProcess(net, train)
    valid_data_process = DataProcess(net, valid)

    if sys.gettrace() is None:
        print("正在保存训练opt参数...")
        torch.save(opt, opt.params_filename)
        print("保存训练opt参数成功...")

        print("正在保存描述...")
        # np.savetxt(opt.desc)
        print("保存描述成功...")

    for epoch in range(opt.nepoch):
        train_data_process.train_epoch_process(epoch)

        valid_data_process.valid_epoch_process(epoch)

        # save model, overwriting the old model
        if epoch % opt.saveinterval == 0 or epoch == opt.nepoch - 1:
            torch.save(net.state_dict(), opt.model_filename)

        # save model in a separate file in epochs 0,5,10,50,100,500,1000, ...
        if epoch % (5 * 10 ** math.floor(
                math.log10(max(2, epoch - 1)))) == 0 or epoch % 100 == 0 or epoch == opt.nepoch - 1:
            torch.save(net.state_dict(), os.path.join(opt.outdir, "models", '%s_model_%d.pth' % (opt.name, epoch)))

    train_data_process.writer.close()
    valid_data_process.writer.close()


if __name__ == "__main__":
    train_model(PointCleanNet.OUTLIERS_REMOVAL)
