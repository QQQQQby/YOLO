# coding: utf-8
import time

import torch
from torch import nn
import torch.nn.functional as F

from util.functions import get_same_paddings


class YOLOv3Backbone(nn.Module):
    def __init__(self):
        super(YOLOv3Backbone, self).__init__()
        self.set_dbl_layer(1, 1, 3, 32, 3, stride=1, bias=False)
        self.set_dbl_layer(2, 2, 32, 64, 3, stride=2, bias=False)

        """ResBlock * 1"""
        for i in range(3, 5, 2):
            self.set_dbl_layer(i, i, 64, 32, 1, stride=1, bias=False)
            self.set_dbl_layer(i + 1, i + 1, 32, 64, 3, stride=1, bias=False)

        self.set_dbl_layer(5, 5, 64, 128, 3, stride=2, bias=False)
        """ResBlock * 2"""
        channel_a, channel_b = 128, 64
        for i in range(6, 10, 2):
            self.set_dbl_layer(i, i, channel_a, channel_b, 1, stride=1, bias=False)
            self.set_dbl_layer(i + 1, i + 1, channel_b, channel_a, 3, stride=1, bias=False)

        self.set_dbl_layer(10, 10, 128, 256, 3, stride=2, bias=False)
        """ResBlock * 8"""
        channel_a, channel_b = 256, 128
        for i in range(11, 27, 2):
            self.set_dbl_layer(i, i, channel_a, channel_b, 1, stride=1, bias=False)
            self.set_dbl_layer(i + 1, i + 1, channel_b, channel_a, 3, stride=1, bias=False)

        self.set_dbl_layer(27, 27, 256, 512, 3, stride=2, bias=False)
        """ResBlock * 8"""
        channel_a, channel_b = 512, 256
        for i in range(28, 44, 2):
            self.set_dbl_layer(i, i, channel_a, channel_b, 1, stride=1, bias=False)
            self.set_dbl_layer(i + 1, i + 1, channel_b, channel_a, 3, stride=1, bias=False)

        self.set_dbl_layer(44, 44, 512, 1024, 3, stride=2, bias=False)
        """ResBlock * 4"""
        channel_a, channel_b = 1024, 512
        for i in range(45, 53, 2):
            self.set_dbl_layer(i, i, channel_a, channel_b, 1, stride=1, bias=False)
            self.set_dbl_layer(i + 1, i + 1, channel_b, channel_a, 3, stride=1, bias=False)

        channel_a, channel_b = 1024, 512
        for i in range(53, 59, 2):
            self.set_dbl_layer(i, i, channel_a, channel_b, 1, stride=1, bias=False)
            self.set_dbl_layer(i + 1, i + 1, channel_b, channel_a, 3, stride=1, bias=False)

        """Output 1"""
        self.conv59 = nn.Conv2d(1024, 255, 1, stride=1, bias=True)

        self.set_dbl_layer(60, 59, 512, 256, 1, stride=1, bias=False)

        self.upsample1 = nn.Upsample(scale_factor=2)

        self.set_dbl_layer(61, 60, 768, 256, 1, stride=1, bias=False)
        self.set_dbl_layer(62, 61, 256, 512, 3, stride=1, bias=False)

        channel_a, channel_b = 512, 256
        for i in range(63, 67, 2):
            self.set_dbl_layer(i, i - 1, channel_a, channel_b, 1, stride=1, bias=False)
            self.set_dbl_layer(i + 1, i, channel_b, channel_a, 3, stride=1, bias=False)

        """Output 2"""
        self.conv67 = nn.Conv2d(512, 255, 1, stride=1, bias=True)

        self.set_dbl_layer(68, 66, 256, 128, 1, stride=1, bias=False)

        self.upsample2 = nn.Upsample(scale_factor=2)

        self.set_dbl_layer(69, 67, 384, 128, 1, stride=1, bias=False)
        self.set_dbl_layer(70, 68, 128, 256, 3, stride=1, bias=False)

        channel_a, channel_b = 256, 128
        for i in range(71, 75, 2):
            self.set_dbl_layer(i, i - 2, channel_a, channel_b, 1, stride=1, bias=False)
            self.set_dbl_layer(i + 1, i - 1, channel_b, channel_a, 3, stride=1, bias=False)

        """Output 3"""
        self.conv75 = nn.Conv2d(256, 255, 1, stride=1, bias=True)

    def set_dbl_layer(self, conv_id, bn_id, in_channels, out_channels, kernel_size, stride, bias):
        setattr(
            self, 'conv%d' % conv_id,
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0, bias=bias)
        )
        setattr(self, 'batch_norm%d' % bn_id, nn.BatchNorm2d(out_channels))

    def forward(self, x):
        """W * H * 3"""
        x = x.permute(0, 3, 1, 2).float()
        """3 * W * H"""
        x = self.dbl_forward(x, 1, 1)
        x = self.dbl_forward(x, 2, 2)

        """ResBlock * 1"""
        for i in range(3, 5, 2):
            x = self.res_forward(x, i, i)

        x = self.dbl_forward(x, 5, 5)
        """ResBlock * 2"""
        for i in range(6, 10, 2):
            x = self.res_forward(x, i, i)

        x = self.dbl_forward(x, 10, 10)
        """ResBlock * 8"""
        for i in range(11, 27, 2):
            x = self.res_forward(x, i, i)

        y = self.dbl_forward(x, 27, 27)
        """ResBlock * 8"""
        for i in range(28, 44, 2):
            y = self.res_forward(y, i, i)

        z = self.dbl_forward(y, 44, 44)
        """ResBlock * 4"""
        for i in range(45, 53, 2):
            z = self.res_forward(z, i, i)

        for i in range(53, 58):
            z = self.dbl_forward(z, i, i)
        o1 = self.dbl_forward(z, 58, 58)
        o1 = self.conv59(o1)

        z = self.dbl_forward(z, 60, 59)
        z = self.upsample1(z)

        y = torch.cat([z, y], 1)

        for i in range(61, 66):
            y = self.dbl_forward(y, i, i - 1)

        o2 = self.dbl_forward(y, 66, 65)
        o2 = self.conv67(o2)

        y = self.dbl_forward(y, 68, 66)
        y = self.upsample2(y)

        x = torch.cat([y, x], 1)

        for i in range(69, 75):
            x = self.dbl_forward(x, i, i - 2)

        o3 = self.conv75(x)

        B = 3
        num_classes = 80

        o1 = o1.permute(0, 2, 3, 1)
        o2 = o2.permute(0, 2, 3, 1)
        o3 = o3.permute(0, 2, 3, 1)
        o1 = o1.view(*(o1.size()[:3]), B, num_classes + 5)
        o2 = o2.view(*(o2.size()[:3]), B, num_classes + 5)
        o3 = o3.view(*(o3.size()[:3]), B, num_classes + 5)
        coords = [o1[..., :4], o2[..., :4], o3[..., :4]]
        confs = [o1[..., 4], o2[..., 4], o3[..., 4]]
        probs = [o1[..., 5:], o2[..., 5:], o3[..., 5:]]
        return probs, confs, coords

    def dbl_forward(self, x, conv_id, bn_id):
        conv_layer = getattr(self, 'conv%d' % conv_id)
        x = F.pad(x, get_same_paddings(x.shape[-2:], conv_layer.kernel_size, conv_layer.stride))
        x = conv_layer(x)
        x = getattr(self, 'batch_norm%d' % bn_id)(x)
        x = F.leaky_relu(x, 0.1, inplace=False)
        return x

    def res_forward(self, x, conv_id, bn_id):
        y = x
        x = self.dbl_forward(x, conv_id, bn_id)
        x = self.dbl_forward(x, conv_id + 1, bn_id + 1)
        x = x + y
        return x


# from torch.utils import tensorboard
#
# if __name__ == '__main__':
#     m = YOLOv3Backbone()
#     with tensorboard.SummaryWriter("log") as writer:
#         writer.add_graph(m, [torch.rand(1, 608, 608, 3)])
