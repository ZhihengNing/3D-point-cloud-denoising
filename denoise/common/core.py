import json
import os

from denoise.utils.device_util import device_init

file_path = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(file_path, "config.json")) as f:
    config = json.load(f)
    USE_GPU = config.get("use_gpu")
    GPU_LIST = config.get("gpu_list")

DEVICE = device_init(use_gpu=USE_GPU, gpu_list=GPU_LIST)


#####################################
class PointCleanNet:
    def __init__(self):
        pass

    OUTLIERS_REMOVAL = "outliers_removal"
    NOISE_REMOVAL = "noise_removal"

#####################################
