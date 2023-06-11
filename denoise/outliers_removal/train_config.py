from denoise.common.config import BaseTrainConfig


from denoise.common.core import *


class OutliersRemovalTrainConfig(BaseTrainConfig):
    def __init__(self):
        super().__init__(PointCleanNet.OUTLIERS_REMOVAL)


        self.epoch = 200
        self.batch_size = 512 * len(GPU_LIST)
        self.patches_per_point_cloud = 500
        self.points_per_patch=500

        if USE_GPU:
            self.num_workers = 4
        else:
            self.num_workers = 0

        self.seed=3627473
        self.training_order = 'random'
        self.log_interval = 10
        self.point_tuple=1
        self.patch_sampling = 'point'

    def base_parser_arguments(self):
        params_filename = os.path.join(self.out_dir,"params", f'{self.name}_params.pth')
        model_filename = os.path.join(self.out_dir,"models",f'{self.name}_model.pth')

        self.parser.add_argument('--name', type=str, default=self.name, help='training run name')
        self.parser.add_argument('--desc', type=str, default='My training run for single-scale normal estimation.',
                                 help='description')

        self.parser.add_argument('--indir', type=str, default=self.indir, help='input folder (point clouds)')

        self.parser.add_argument('--outdir', type=str, default=self.out_dir, help='output folder (trained models)')
        self.parser.add_argument('--logdir', type=str, default=self.log_dir, help='training log folder')

        self.parser.add_argument("--params_filename",type=str,default=params_filename,help='params')
        self.parser.add_argument("--model_filename",type=str,default=model_filename,help="models_params")

        self.parser.add_argument('--trainset', type=str, default=self.train_set, help='training set file name')

        self.parser.add_argument('--validateset', type=str, default=self.validate_set, help='test set file name')

        self.parser.add_argument('--loginterval',type=int,default=self.log_interval,help='log each n iteration')

        self.parser.add_argument('--saveinterval', type=int, default='10', help='save models each n epochs')

        self.parser.add_argument('--refine', type=str, default='', help='refine models at this path')

    def training_parser_arguments(self):
        self.parser.add_argument('--nepoch', type=int, default=self.epoch,
                                 help='number of epochs to train for')
        self.parser.add_argument('--batchSize', type=int,
                                 default=self.batch_size, help='input batch size')
        # 傻逼多线程
        self.parser.add_argument('--workers', type=int, default=self.num_workers,
                                 help='number of data loading workers - 0 means same thread as main execution')
        self.parser.add_argument('--seed', type=int,
                                 default=self.seed, help='manual seed')
        self.parser.add_argument('--training_order', type=str, default=self.training_order,
                                 help='order in which the training patches are presented:\n'
                                      'random: fully random over the entire dataset (the set of all patches is permuted)\n'
                                      'sequential_point_cloud_random_patches: random over the entire dataset, but patches of a shape remain consecutive (shapes and patches inside a shape are permuted)')
        self.parser.add_argument('--identical_epochs', type=int, default=False,
                                 help='use same patches in each epoch, mainly for debugging')
        self.parser.add_argument('--lr', type=float, default=0.0001,
                                 help='learning rate')
        self.parser.add_argument('--momentum', type=float, default=0.9,
                                 help='gradient descent momentum')

    def model_hyperparameter_parser_arguments(self):
        self.parser.add_argument('--outputs', type=str, nargs='+', default=['outliers'], help='outputs of the network')
        self.parser.add_argument('--use_point_stn', type=int,
                                 default=True, help='use point spatial transformer')
        self.parser.add_argument('--use_feat_stn', type=int,
                                 default=True, help='use feature spatial transformer')
        self.parser.add_argument('--sym_op', type=str, default='max',
                                 help='symmetry operation')
        self.parser.add_argument('--point_tuple', type=int, default=self.point_tuple,
                                 help='use n-tuples of points as input instead of single points')
        # 原本是500
        self.parser.add_argument('--points_per_patch', type=int,
                                 default=self.points_per_patch, help='max. number of points per patch')
        self.parser.add_argument('--use_pca', type=int, default=False,
                                 help='Give both inputs and ground truth in local PCA coordinate frame')

        self.parser.add_argument('--patch_radius', type=float, default=[0.05], nargs='+',
                                 help='patch radius in multiples of the point_cloud\'s bounding box diagonal, multiple values for multi-scale.')
        self.parser.add_argument('--patch_center', type=str, default='point', help='center patch at...\n'
                                                                                   'point: center point\n'
                                                                                   'mean: patch mean')
        self.parser.add_argument('--patch_point_count_std', type=float, default=0,
                                 help='standard deviation of the number of points in a patch')

        self.parser.add_argument('--patches_per_point_cloud', type=int, default=self.patches_per_point_cloud,
                                 help='number of patches sampled from each shape in an epoch')

        self.parser.add_argument('--cache_capacity', type=int, default=400,
                                 help='Max. number of dataset elements (usually shapes) to hold in the cache at the same time.')

        self.parser.add_argument('--patch_sampling', type=str, default=self.patch_sampling)

