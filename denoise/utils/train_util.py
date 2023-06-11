import os
import sys


# train_name 此次训练的标识（比如nzh 1951121 2023.5.7)
# module_name 训练任务or模块的名称
# dataset_dir 数据集文件的目录 比如一个txt文件里面存着需要哪些数据集，到时候再根据文件内容读取需要的数据集
# result_dir 训练结果保存在某个文件下
class NetTrainBasicInfo:
    def __init__(self,train_name:str,
                 module_name:str,
                 dataset_dir:str,
                 result_dir:str,
                 train:bool):
        self.train_name=train_name
        self.module_name=module_name
        self.dataset_dir=dataset_dir

        self.train_set = os.path.join(dataset_dir, f"{module_name}_trainingset.txt")
        self.validate_set = os.path.join(dataset_dir, f"{module_name}_validationset.txt")

        self.test_set = os.path.join(dataset_dir, f"{module_name}_testset.txt")

        self.result_dir=result_dir
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        self.out_dir=None
        self.out_log_dir=None
        self.params_dir=None
        self.params_filename=None
        self.model_dir=None
        self.model_filename=None
        self.desc_dir=None
        self.desc_filename=None

        if train:
            self.out_dir=self.generator_out_dir()
            self.out_log_dir=os.path.join(self.out_dir, "logs")

            self.params_dir = os.path.join(self.out_dir, "params")
            if not os.path.exists(self.params_dir):
                os.mkdir(self.params_dir)
            self.params_filename = os.path.join(self.params_dir, f'{self.train_name}_params.pth')

            self.model_dir = os.path.join(self.out_dir, "models")
            if not os.path.exists(self.model_dir):
                os.mkdir(self.model_dir)
            self.model_filename = os.path.join(self.model_dir, f'{self.train_name}_model.pth')

            self.desc_dir = os.path.join(self.out_dir, "description")
            if not os.path.exists(self.desc_dir):
                os.mkdir(self.desc_dir)
            self.desc_filename = os.path.join(self.desc_dir, f'{self.train_name}_description.txt')


    # 锁定输出文件夹
    def generator_out_dir(self):
        times = 1
        per_save_dir = os.path.join(self.result_dir, self.module_name, str(times))
        while True:
            if os.path.exists(per_save_dir):
                times += 1
                per_save_dir = os.path.join(self.result_dir, self.module_name, str(times))
                continue
            else:
                os.makedirs(per_save_dir)
                break
        return per_save_dir

    def choose_to_eval(self,use_special=None):
        if use_special is not None:
            return os.path.join(self.result_dir, self.module_name, use_special)
        # 默认使用最新的模型来评估
        times = 1
        while True:
            if not os.path.exists(os.path.join(self.result_dir, self.module_name, str(times))):
                per_save_dir = os.path.join(self.result_dir,self.module_name, str(times - 1))
                break
            times = times + 1
        return per_save_dir

