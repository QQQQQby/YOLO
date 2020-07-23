# coding: utf-8

import torch
from torch import nn
from torch.nn.modules.utils import _pair
import torch.nn.functional as F


class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, input_size, stride=1,
                 padding=0, dilation=1, bias=True):
        super(LocallyConnected2d, self).__init__()
        self.kernel_size = kernel_size
        self.input_size = input_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.output_size = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1

        self.weight = nn.Parameter(
            torch.randn(
                out_channels,
                in_channels,
                self.output_size,
                self.output_size,
                self.kernel_size,
                self.kernel_size
            ),
            True
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(
                    out_channels,
                    self.output_size,
                    self.output_size
                ),
                True
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        x = F.pad(x, [self.padding] * 4)
        x = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)
        x = x.contiguous().view(*x.size()[:-2], -1).float()
        x = x.unsqueeze(1)
        # print("x:", x.shape)
        # Sum in in_channel and kernel_size dims
        w = self.weight.view(*self.weight.size()[:-2], -1).unsqueeze(0)
        # print("w:", w.shape)
        out = x * w
        out = out.sum([2, -1])
        if self.bias is not None:
            b = self.bias.unsqueeze(0)
            # print("b:", b.shape)
            out += b
        # print("out:", x.shape)
        return out


if __name__ == '__main__':
    torch.random.manual_seed(520)

    batch_size = 4
    in_channels = 3
    out_channels = 2
    image_size = 7
    kernel_size = 1

    a = torch.arange(batch_size * in_channels * image_size * image_size) \
        .view(batch_size, in_channels, image_size, image_size).float()
    # a = F.pad(a, [1, 1, 1, 1])

    l = LocallyConnected2d(in_channels, out_channels, kernel_size, image_size)
    b = l(a)
    print(b)
    print(b.shape)

    print(a)
    b = torch.nn.functional.unfold(a, kernel_size=3)
    b = b.permute(0, 2, 1)
    print(b)
    print(b.shape)
