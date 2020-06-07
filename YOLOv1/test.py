# coding: utf-8

import torch
from torch import nn
from YOLOv1.modules import YOLOv1

if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.manual_seed(1)
    m = YOLOv1()
    o = m(torch.rand(5, 3, 448, 448))
    print(o.shape)

