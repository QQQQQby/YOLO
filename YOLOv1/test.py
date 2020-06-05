# coding: utf-8

import torch
from torch import nn
from YOLOv1.modules import YOLOv1

if __name__ == '__main__':
    torch.manual_seed(1)
    m = YOLOv1()
    o = m(torch.rand(1, 3, 448, 448))
    print(o)
    print(o.shape)

