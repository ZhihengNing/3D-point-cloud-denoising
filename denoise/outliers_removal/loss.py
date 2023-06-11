import numpy as np
import torch

from denoise.common.core import DEVICE
from denoise.utils.point_cloud_util import acc_precision_recall, fa_score


def outliers_loss(one_predict, one_target):
    loss_func = torch.nn.L1Loss(reduction='mean')
    loss = loss_func(one_predict, one_target)
    return loss


def outliers_loss2(one_predict, one_target):
    loss_func = torch.nn.MSELoss(reduction='mean')
    loss = loss_func(one_predict, one_target)
    return loss


def outliers_loss3(one_predict, one_target):
    loss_func = torch.nn.SmoothL1Loss(reduction='mean')
    loss = loss_func(one_predict, one_target)
    return loss


def outlier_loss_balance(one_predict, one_target, p=10):
    predict_zero_nums = torch.sum(one_predict == 0).item()
    target_zero_nums = torch.sum(one_target == 0).item()
    x = predict_zero_nums / target_zero_nums

    par = 1 / (np.exp(p * (1.05 - 1)) - 1)
    if x < 1:
        return 3.0
    else:
        return par * (np.exp(p * (x - 1)) - 1)


def outliers_loss_acc(one_predict, one_target):
    loss_func = torch.nn.SmoothL1Loss(reduction='mean')
    loss1 = loss_func(one_predict, one_target)
    one_predict = torch.where(one_predict < 0.5, torch.tensor(0).to(DEVICE), torch.tensor(1).to(DEVICE))
    # 准确率 acc
    loss2 = 1.0 - (one_target == one_predict).sum().item() / one_predict.shape[0]
    a = 0.9
    loss = a * loss1 + (1 - a) * loss2
    return loss


def outliers_loss_recall(one_predict, one_target):
    loss_func = torch.nn.SmoothL1Loss(reduction='mean')
    loss1 = loss_func(one_predict, one_target)
    one_predict = torch.where(one_predict < 0.5, torch.tensor(0).to(DEVICE), torch.tensor(1).to(DEVICE))
    # 召回率与精确率 然后构造f2score
    _, precision, recall = acc_precision_recall(one_predict, one_target)
    loss2 = 1.0 - fa_score(precision, recall, 2)
    a = 0.9
    loss = a * loss1 + (1 - a) * loss2
    return loss


def outliers_loss_recall_plus(one_predict, one_target):
    loss_func = torch.nn.L1Loss(reduction='mean')
    loss1 = loss_func(one_predict, one_target)
    one_predict = torch.where(one_predict < 0.5, torch.tensor(0).to(DEVICE), torch.tensor(1).to(DEVICE))

    # 此处设定2 说明召回率比精确率重要
    _, precision, recall = acc_precision_recall(one_predict, one_target)
    loss2 = 1.0 - recall
    loss3 = 1.0 - precision
    loss = 0.05 * loss1 + 0.9 * loss2 + 0.05 * loss3
    # print(f"loss1:{loss1},loss2:{loss2},loss3:{loss3},loss:{loss}")
    return loss


def outliers_loss_precision(one_predict, one_target):
    one_predict = torch.where(one_predict < 0.5, torch.tensor(0).to(DEVICE), torch.tensor(1).to(DEVICE))
    _, precision, _ = acc_precision_recall(one_predict, one_target)
    loss3 = 1.0 - precision
    loss = loss3
    return loss
