import torch

from denoise.common.core import DEVICE, PointCleanNet
from denoise.noise_removal.eval_config import NoiseRemovalEvalConfig
from denoise.noise_removal.model.respcpnet import NoiseRemovalResPCPNet
from denoise.outliers_removal.eval_config import OutliersRemovalEvalConfig
from denoise.outliers_removal.model.respcpnet import OutliersRemovalResPCPNet
from denoise.process.preprocess import Test
from denoise.process.process import DataProcess
from denoise.utils.device_util import state_dict_process
from denoise.utils.seed_util import set_seed


def eval_model(data_process_type):
    if data_process_type == PointCleanNet.OUTLIERS_REMOVAL:
        config = OutliersRemovalEvalConfig()
    elif data_process_type == PointCleanNet.NOISE_REMOVAL:
        config = NoiseRemovalEvalConfig()
    else:
        raise ValueError("eval期间错误的opt类型")

    eval_opt = config.get_config()
    set_seed(eval_opt.seed)
    train_opt=None
    try:
        train_opt = torch.load(eval_opt.params_filename)
    except Exception as e:
        print(e)

    train_opt.nepoch = 1
    if data_process_type == PointCleanNet.OUTLIERS_REMOVAL:
        train_opt.patch_sampling = 'point'
    elif data_process_type == PointCleanNet.NOISE_REMOVAL:
        train_opt.patch_sampling = 'radius'
    else:
        raise ValueError
    test = Test(eval_opt, train_opt)
    output_dim = test.dataset.pred_dim
    if data_process_type == PointCleanNet.OUTLIERS_REMOVAL:
        net = OutliersRemovalResPCPNet(
            num_points=train_opt.points_per_patch,
            output_dim=output_dim,
            use_point_stn=train_opt.use_point_stn,
            use_feat_stn=train_opt.use_feat_stn,
            sym_op=train_opt.sym_op,
            point_tuple=train_opt.point_tuple)
    elif data_process_type == PointCleanNet.NOISE_REMOVAL:
        net = NoiseRemovalResPCPNet(
            num_points=train_opt.points_per_patch,
            output_dim=output_dim,
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

    test_data_process = DataProcess(net, test)

    for epoch in range(int(eval_opt.nepoch)):
        test_data_process.test_epoch_process(epoch)


if __name__ == "__main__":
    eval_model(PointCleanNet.OUTLIERS_REMOVAL)
