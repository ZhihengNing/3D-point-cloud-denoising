import os

import numpy as np
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from denoise.common.core import DEVICE
from denoise.process.preprocess import Train, Valid, Origin, Test, WholeProcess
from denoise.utils.point_cloud_util import acc_precision_recall


class Logger:
    def __init__(self, train_name: str, total_epoch: int, total_iterations: int, data_loader_type: str):
        self.train_name: str = train_name
        self.total_epoch: int = total_epoch
        self.total_iterations: int = total_iterations
        self.data_loader_type: str = data_loader_type

    def log(self, epoch_times: int, iterator_times: int, loss=None):
        if self.data_loader_type == Train.__name__:
            return self.log_train(loss, epoch_times, iterator_times)
        elif self.data_loader_type == Valid.__name__:
            return self.log_valid(loss, epoch_times, iterator_times)
        elif self.data_loader_type == Test.__name__:
            return self.log_test(loss, epoch_times, iterator_times)
        elif self.data_loader_type == WholeProcess.__name__:
            return self.log_whole_process(epoch_times, iterator_times)
        else:
            raise ValueError(f"no type!!!{self.data_loader_type}")

    def log_train(self, loss, epoch_times: int, iterator_times: int):
        def green(x):
            return '\033[92m' + x + '\033[0m'

        print(
            f'{self.train_name} '
            f'[{epoch_times}/{self.total_epoch}: '
            f'{iterator_times}/{self.total_iterations}] '
            f'{green(self.data_loader_type)} loss:{loss.item()}')

    def log_valid(self, loss, epoch_times: int, iterator_times: int):
        def blue(x):
            return '\033[94m' + x + '\033[0m'

        print(
            f'{self.train_name} '
            f'[{epoch_times}/{self.total_epoch}: '
            f'{iterator_times}/{self.total_iterations}] '
            f'{blue(self.data_loader_type)} loss:{loss.item()}')

    def log_test(self, loss, epoch_times: int, iterator_times: int):
        def color2(x):
            return '\033[96m' + x + '\033[0m'

        print(
            f'{self.train_name} '
            f'[{epoch_times}/{self.total_epoch}: '
            f'{iterator_times}/{self.total_iterations}] '
            f'{color2(self.data_loader_type)} loss:{loss.item()}')

    def log_whole_process(self, epoch_times: int, iterator_times: int):
        def color3(x):
            return '\033[96m' + x + '\033[0m'

        print(
            f'{self.train_name} '
            f'[{epoch_times}/{self.total_epoch}: '
            f'{iterator_times}/{self.total_iterations}] '
            f'{color3(self.data_loader_type)}')


class DataProcess:
    def __init__(self, net: nn.Module, origin: Origin):
        self.opt = origin.opt
        self.net = net
        self.origin = origin
        self.log_dirname = self.opt.logdir

        self.criterion = self.compute_loss
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.opt.lr, momentum=self.opt.momentum)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=[], gamma=0.1)

        self.data_loader = origin.data_loader
        self.dataset = origin.dataset
        self.dataset_sampler = origin.dataset_sampler
        self.total_iterations = len(self.data_loader)

        self.data_loader_type = type(origin).__name__

        self.logger = Logger(train_name=self.opt.name,
                             total_epoch=int(self.opt.nepoch),
                             total_iterations=self.total_iterations,
                             data_loader_type=self.data_loader_type)

        self.epoch_loss = []
        self.acc_all = []
        self.precision_all = []
        self.recall_all = []
        self.writer = SummaryWriter(os.path.join(self.log_dirname, self.data_loader_type))

    def compute_loss(self, predict, target, patch_rot):
        point_cloud_feature = self.dataset.point_cloud_features
        loss = 0.
        for feature, parameters in point_cloud_feature.feature_Map.items():
            feature_index = parameters["index"]
            if feature_index != -1:
                begin_position = parameters["begin_position"]
                end_position = parameters["end_position"]
                one_predict = predict[:, begin_position:end_position]
                one_target = target[feature_index]
                if patch_rot is not None and feature == "clean_points":
                    one_predict = torch.bmm(one_predict.unsqueeze(1), patch_rot.transpose(2, 1)).squeeze(1)
                loss += parameters["weight"] * parameters['criterion'](one_predict, one_target)
        return loss

    def record_info(self, epoch_times):
        if not len(self.acc_all) == 0:
            self.writer.add_scalar("acc", np.mean(self.acc_all), epoch_times)
        if not len(self.precision_all) == 0:
            self.writer.add_scalar("precision", np.mean(self.precision_all), epoch_times)
        if not len(self.recall_all) == 0:
            self.writer.add_scalar("recall", np.mean(self.recall_all), epoch_times)

    def iterator_process(self, iterator_times: int, epoch_times, batch_data):
        points = batch_data[0]
        target = batch_data[1]
        trans = batch_data[2]
        patch_radii = batch_data[3]

        points = points.transpose(1, 2)
        points = points.to(DEVICE)
        target = [item.to(DEVICE) for item in target]

        # 比如一个样本就是某个点和周围点的结果，输出的就是某个点的去噪结果，充分利用了领域信息
        predict, transpose, _, _ = self.net(points)

        if self.data_loader_type == Train.__name__:
            self.optimizer.zero_grad()
            loss = self.criterion(predict, target, transpose if self.opt.use_point_stn else None)
            if target[0].shape[1] == 1:
                new_predict = torch.where(predict < 0.5, torch.tensor(0).to(DEVICE), torch.tensor(1).to(DEVICE))
                acc, precision, recall = acc_precision_recall(new_predict.cpu(), target[0].cpu())
                self.acc_all.append(acc)
                self.precision_all.append(precision)
                self.recall_all.append(recall)
            loss.backward()
            self.optimizer.step()
        elif self.data_loader_type == Valid.__name__:
            loss = self.criterion(predict, target, transpose if self.opt.use_point_stn else None)
            if target[0].shape[1] == 1:
                new_predict = torch.where(predict < 0.5, torch.tensor(0).to(DEVICE), torch.tensor(1).to(DEVICE))
                acc, precision, recall = acc_precision_recall(new_predict.cpu(), target[0].cpu())
                self.acc_all.append(acc)
                self.precision_all.append(precision)
                self.recall_all.append(recall)
        elif self.data_loader_type == Test.__name__:
            loss = self.criterion(predict, target, transpose if self.opt.use_point_stn else None)
        elif self.data_loader_type == WholeProcess.__name__:
            if iterator_times % self.opt.loginterval == 0:
                self.logger.log(epoch_times, iterator_times)

            # 如果是全流程直接返回预测值就足够了
            return predict, transpose, target, trans, patch_radii
        else:
            raise ValueError("不合适的类型")

        self.epoch_loss.append(loss.item())
        if iterator_times % self.opt.loginterval == 0:
            self.logger.log(epoch_times, iterator_times, loss)
        return predict, target

    def train_epoch_process(self, epoch_times: int):
        self.epoch_loss.clear()
        self.acc_all.clear()
        self.precision_all.clear()
        self.recall_all.clear()
        self.net.train()
        for index, batch_data in enumerate(self.data_loader):
            self.iterator_process(index, epoch_times, batch_data)
        self.writer.add_scalar("Loss", np.mean(self.epoch_loss), epoch_times)
        self.record_info(epoch_times)
        self.scheduler.step()

    def valid_epoch_process(self, epoch_times: int):
        self.epoch_loss.clear()
        self.acc_all.clear()
        self.precision_all.clear()
        self.recall_all.clear()
        self.net.eval()
        with torch.no_grad():
            for index, batch_data in enumerate(self.data_loader):
                self.iterator_process(index, epoch_times, batch_data)
            self.writer.add_scalar("Loss", np.mean(self.epoch_loss), epoch_times)
            self.record_info(epoch_times)

    def test_epoch_process(self, epoch_times: int):
        self.epoch_loss.clear()
        self.precision_all.clear()
        self.recall_all.clear()
        self.net.eval()
        with torch.no_grad():
            for index, batch_data in enumerate(self.data_loader):
                predict, target = self.iterator_process(index, epoch_times, batch_data)
                count = 0
                len = target[0].shape[1]
                target = target[0]

                if len == 1:
                    acc, precision, recall = acc_precision_recall(predict, target)
                    for index, item in enumerate(predict):
                        if item < 0.5 and target[index] == 0.:
                            count += 1
                        elif item > 0.5 and target[index] == 1.:
                            count += 1
                    if not recall == 1.0:
                        print(f"acc:{count / predict.shape[0]} precision:{precision},recall:{recall}")

            self.writer.add_scalar("Loss", np.mean(self.epoch_loss), epoch_times)

    def whole_process_epoch_process(self):
        epoch_times = 0
        self.net.eval()
        whole_result = [[], [], [], [], []]

        with torch.no_grad():
            for index, batch_data in enumerate(self.data_loader):
                result = self.iterator_process(index, epoch_times, batch_data)
                for i in range(len(result)):
                    whole_result[i].append(result[i])

        return whole_result
