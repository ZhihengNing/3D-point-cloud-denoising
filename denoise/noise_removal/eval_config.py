from denoise.common.config import BaseEvalConfig
from denoise.common.core import *


class NoiseRemovalEvalConfig(BaseEvalConfig):
    def __init__(self):
        super().__init__(PointCleanNet.NOISE_REMOVAL)
        self.epoch = 1
        self.batch_size = 1000
        self.patches_per_point_cloud = 500

        if USE_GPU:
            self.num_workers = 4
        else:
            self.num_workers = 0

        self.seed=40938661

        self.sampling='full'

    def base_parser_arguments(self):
        params_filename = os.path.join(self.eval_dir,"params", f'{self.name}_params.pth')
        model_filename = os.path.join(self.eval_dir,"models",f'{self.name}_model.pth')

        # naming / file handling
        self.parser.add_argument('--indir', type=str, default=self.indir,help='input folder (point clouds)')
        self.parser.add_argument('--testset', type=str, default=self.test_set, help='shape set file name')

        self.parser.add_argument("--params_filename",type=str,default=params_filename,help='params')
        self.parser.add_argument("--model_filename",type=str,default=model_filename,help="models_params")

        self.parser.add_argument('--sparse_patches', type=int, default=False,
                            help='evaluate on a sparse set of patches, given by a .pidx file containing the patch center point indices.')

    def eval_parser_arguments(self):
        self.parser.add_argument('--nepoch', type=int, default=self.epoch,
                                 help='number of epochs to train for')
        self.parser.add_argument('--batchSize', type=int,
                                 default=self.batch_size, help='input batch size')
        # 傻逼多线程
        self.parser.add_argument('--workers', type=int, default=self.num_workers,
                                 help='number of data loading workers - 0 means same thread as main execution')
        self.parser.add_argument('--seed', type=int, default=self.seed, help='manual seed')

        self.parser.add_argument('--sampling', type=str, default=self.sampling, help='sampling strategy, any of:\n'
                                                                              'full: evaluate all points in the dataset\n'
                                                                              'sequential_point_cloud_random_patches: pick n random points from each shape as patch centers, shape order is not randomized')

    def dataset_parser_arguments(self):
        self.parser.add_argument('--patches_per_point_cloud', type=int, default=self.patches_per_point_cloud,
                                 help='number of patches evaluated in each shape (only for sequential_shapes_random_patches)')



