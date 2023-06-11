import argparse
import json
import os
from abc import abstractmethod

from environments import BASE_PATH
from denoise.utils.train_util import NetTrainBasicInfo

# 这个配置类存放着数据文件夹位置等基本信息
file_path = os.path.dirname(os.path.abspath(__file__))


class BaseConfig:
    def __init__(self, data_process_type: str, train: bool):
        self.data_process_type = data_process_type
        self.train = train

        with open(os.path.join(file_path, f"{data_process_type}_config.json")) as f:
            result = json.load(f)
            self.name = result.get("name")
            self.module_dir = result.get("module_dir")
            base_data_dir = result.get("base_data_dir")
            train_data_dir = result.get("train_data_dir")
            eval_data_dir = result.get("eval_data_dir")
            result_dir = result.get("result_dir")
            eval_dir = result.get("eval_dir")

        if self.train:
            self.indir = os.path.join(BASE_PATH, base_data_dir, train_data_dir)
        else:
            self.indir = os.path.join(BASE_PATH, base_data_dir, eval_data_dir)

        self.dataset_dir = os.path.join(BASE_PATH, base_data_dir)
        self.result_dir = os.path.join(BASE_PATH, result_dir)

        net_train_basic_info = NetTrainBasicInfo(train_name=self.name,
                                                 module_name=self.module_dir,
                                                 dataset_dir=self.dataset_dir,
                                                 result_dir=self.result_dir,
                                                 train=self.train)
        self.out_dir = net_train_basic_info.out_dir
        self.log_dir = net_train_basic_info.out_log_dir

        self.train_set = net_train_basic_info.train_set
        self.validate_set = net_train_basic_info.validate_set
        self.test_set = net_train_basic_info.test_set

        # 这里的意思就是使用第一次训练的结果来评估模型
        self.eval_dir = net_train_basic_info.choose_to_eval(eval_dir)
        if not train:
            print(f"use {self.eval_dir} to {self.module_dir} eval")
        self.parser = argparse.ArgumentParser()

    @abstractmethod
    def get_config(self):
        raise NotImplementedError


class BaseTrainConfig(BaseConfig):

    def __init__(self, data_process_type):
        super().__init__(data_process_type, True)

    @abstractmethod
    def base_parser_arguments(self):
        raise NotImplementedError

    @abstractmethod
    def training_parser_arguments(self):
        raise NotImplementedError

    @abstractmethod
    def model_hyperparameter_parser_arguments(self):
        raise NotImplementedError

    def get_config(self):
        self.base_parser_arguments()
        self.training_parser_arguments()
        self.model_hyperparameter_parser_arguments()
        return self.parser.parse_args()


class BaseEvalConfig(BaseConfig):
    def __init__(self, data_process_type):
        super().__init__(data_process_type, False)

    @abstractmethod
    def base_parser_arguments(self):
        raise NotImplementedError

    @abstractmethod
    def eval_parser_arguments(self):
        raise NotImplementedError

    @abstractmethod
    def dataset_parser_arguments(self):
        raise NotImplementedError

    def get_config(self):
        self.base_parser_arguments()
        self.eval_parser_arguments()
        self.dataset_parser_arguments()
        return self.parser.parse_args()
