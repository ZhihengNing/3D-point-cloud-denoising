import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

from denoise.noise_removal.model.pcpnet import BasicBlock
from denoise.utils.net_util import batch_quat_to_rotmat


class ResSTN(nn.Module):
    def __init__(self, num_scales=1, num_points=500, dim=3, sym_op='max', quaternion=False):
        super(ResSTN, self).__init__()
        self.quaternion = quaternion
        self.dim = dim
        self.sym_op = sym_op
        self.num_scales = num_scales
        self.num_points = num_points

        self.b1 = BasicBlock(self.dim, 64, conv=True)
        self.b2 = BasicBlock(64, 128, conv=True)
        self.b3 = BasicBlock(128, 1024, conv=True)
        self.mp1 = torch.nn.MaxPool1d(num_points)

        self.bfc1 = BasicBlock(1024, 512)
        self.bfc2 = BasicBlock(512, 256)
        if not quaternion:
            self.bfc3 = BasicBlock(256, self.dim * self.dim)
        else:
            self.bfc3 = BasicBlock(256, 4)

        if self.num_scales > 1:
            self.bfc0 = BasicBlock(1024 * self.num_scales, 1024)

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        # symmetric operation over all points
        if self.num_scales == 1:
            x = self.mp1(x)
        else:
            if x.is_cuda:
                x_scales = Variable(torch.cuda.FloatTensor(x.size(0), 1024 * self.num_scales, 1))
            else:
                x_scales = Variable(torch.FloatTensor(x.size(0), 1024 * self.num_scales, 1))
            for s in range(self.num_scales):
                x_scales[:, s * 1024:(s + 1) * 1024, :] = self.mp1(
                    x[:, :, s * self.num_points:(s + 1) * self.num_points])
            x = x_scales

        x = x.view(-1, 1024 * self.num_scales)

        if self.num_scales > 1:
            x = self.bfc0(x)

        x = self.bfc1(x)
        x = self.bfc2(x)
        x = self.bfc3(x)

        if not self.quaternion:
            iden = Variable(torch.from_numpy(np.identity(self.dim, 'float32')).clone()) \
                .view(1, self.dim * self.dim) \
                .repeat(batchsize, 1)

            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden
            x = x.view(-1, self.dim, self.dim)
        else:
            # add identity quaternion (so the network can output 0 to leave the point cloud identical)
            iden = Variable(torch.FloatTensor([1, 0, 0, 0]))
            if x.is_cuda:
                iden = iden.cuda()
            x = x + iden

            # convert quaternion to rotation matrix
            if x.is_cuda:
                trans = Variable(torch.cuda.FloatTensor(batchsize, 3, 3))
            else:
                trans = Variable(torch.FloatTensor(batchsize, 3, 3))
            x = batch_quat_to_rotmat(x, trans)
        return x


class ResPointNetFeat(nn.Module):
    def __init__(self, num_scales=1, num_points=500, use_point_stn=True, use_feat_stn=True, sym_op='max',
                 get_pointfvals=False, point_tuple=1):
        super(ResPointNetFeat, self).__init__()
        self.num_points = num_points
        self.num_scales = num_scales
        self.use_point_stn = use_point_stn
        self.use_feat_stn = use_feat_stn
        self.sym_op = sym_op
        self.get_pointfvals = get_pointfvals
        self.point_tuple = point_tuple

        if self.use_point_stn:
            self.stn1 = ResSTN(num_scales=self.num_scales, num_points=num_points * self.point_tuple, dim=3,
                               sym_op=self.sym_op, quaternion=True)

        if self.use_feat_stn:
            self.stn2 = ResSTN(num_scales=self.num_scales, num_points=num_points, dim=64, sym_op=self.sym_op)

        self.b0a = BasicBlock(3 * self.point_tuple, 64, conv=True)
        self.b0b = BasicBlock(64, 64, conv=True)

        self.b1 = BasicBlock(64, 64, conv=True)
        self.b2 = BasicBlock(64, 128, conv=True)
        self.b3 = BasicBlock(128, 1024, conv=True)

        if self.num_scales > 1:
            self.b4 = BasicBlock(1024, 1024 * self.num_scales, conv=True)

        if self.sym_op == 'max':
            self.mp1 = torch.nn.MaxPool1d(num_points)
        elif self.sym_op == 'sum':
            self.mp1 = None
        else:
            raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

    def forward(self, x):

        # input transform
        if self.use_point_stn:
            # from tuples to list of single points
            x = x.view(x.size(0), 3, -1)
            trans = self.stn1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans)
            x = x.transpose(2, 1)
            x = x.contiguous().view(x.size(0), 3 * self.point_tuple, -1)
        else:
            trans = None

        # mlp (64,64)
        x = self.b0a(x)
        x = self.b0b(x)

        # feature transform
        if self.use_feat_stn:
            trans2 = self.stn2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans2)
            x = x.transpose(2, 1)
        else:
            trans2 = None

        # mlp (64,128,1024)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)

        # mlp (1024,1024*num_scales)
        if self.num_scales > 1:
            x = self.b4(x)

        if self.get_pointfvals:
            pointfvals = x
        else:
            pointfvals = None  # so the intermediate result can be forgotten if it is not needed

        # symmetric max operation over all points
        if self.num_scales == 1:
            if self.sym_op == 'max':
                x = self.mp1(x)
            elif self.sym_op == 'sum':
                x = torch.sum(x, 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))

        else:
            if x.is_cuda:
                x_scales = Variable(torch.cuda.FloatTensor(x.size(0), 1024 * self.num_scales ** 2, 1))
            else:
                x_scales = Variable(torch.FloatTensor(x.size(0), 1024 * self.num_scales ** 2, 1))
            if self.sym_op == 'max':
                for s in range(self.num_scales):
                    x_scales[:, s * self.num_scales * 1024:(s + 1) * self.num_scales * 1024, :] = self.mp1(
                        x[:, :, s * self.num_points:(s + 1) * self.num_points])
            elif self.sym_op == 'sum':
                for s in range(self.num_scales):
                    x_scales[:, s * self.num_scales * 1024:(s + 1) * self.num_scales * 1024, :] = torch.sum(
                        x[:, :, s * self.num_points:(s + 1) * self.num_points], 2, keepdim=True)
            else:
                raise ValueError('Unsupported symmetric operation: %s' % (self.sym_op))
            x = x_scales

        x = x.view(-1, 1024 * self.num_scales ** 2)

        return x, trans, trans2, pointfvals


class NoiseRemovalResPCPNet(nn.Module):

    def __init__(self, num_points=500,
                 output_dim=3,
                 use_point_stn=True,
                 use_feat_stn=True,
                 sym_op='max',
                 get_pointfvals=False,
                 point_tuple=1):
        super(NoiseRemovalResPCPNet, self).__init__()
        self.num_points = num_points

        self.feat = ResPointNetFeat(
            num_points=num_points,
            num_scales=1,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            get_pointfvals=get_pointfvals,
            point_tuple=point_tuple)

        self.b1 = BasicBlock(1024, 512)

        self.b2 = BasicBlock(512, 256)
        self.b3 = BasicBlock(256, output_dim)

    def forward(self, x):
        x, trans, trans2, pointfvals = self.feat(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return x, trans, trans2, pointfvals


class ResMSPCPNet(nn.Module):
    def __init__(self, num_scales=2, num_points=500, output_dim=3, use_point_stn=True, use_feat_stn=True, sym_op='max',
                 get_pointfvals=False, point_tuple=1):
        super(ResMSPCPNet, self).__init__()
        self.num_points = num_points

        self.feat = ResPointNetFeat(
            num_points=num_points,
            num_scales=num_scales,
            use_point_stn=use_point_stn,
            use_feat_stn=use_feat_stn,
            sym_op=sym_op,
            get_pointfvals=get_pointfvals,
            point_tuple=point_tuple)
        self.b0 = BasicBlock(1024 * num_scales ** 2, 1024)
        self.b1 = BasicBlock(1024, 512)
        self.b2 = BasicBlock(512, 256)
        self.b3 = BasicBlock(256, output_dim)

    def forward(self, x):
        x, trans, trans2, pointfvals = self.feat(x)
        x = self.b0(x)
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        return x, trans, trans2, pointfvals
