import os
from collections import OrderedDict
from typing import List

import torch


def device_init(use_gpu:bool=True,gpu_list: List[int]=None):
    if use_gpu:
        if gpu_list is not None and len(gpu_list)>0:
            gpu_list_str = ','.join(map(str, gpu_list))
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)
            print(f"use gpu{gpu_list} to run")
            print(f"CUDA_VISIBLE_DEVICES :{os.environ['CUDA_VISIBLE_DEVICES']}")
            print(f"device_count :{torch.cuda.device_count()}")
        # time 19:54 begin iteration 400/354375
        # 这里默认第一张就是主卡
        device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
        return device
    else:
        print("use cpu to train")
        return torch.device("cpu")

# 因为有可能出现多gpu训练的情况，需要对其进行一些操作
def state_dict_process(state_dict):
    # 因为有可能出现多gpu训练的情况，需要对其进行一些操作
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        key = k[7:] if k.startswith('module.') else k
        new_state_dict[key] = v
    return new_state_dict