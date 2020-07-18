# coding: utf-8

import torch
from torch import nn
from xml.dom import minidom
import cv2
from tqdm import tqdm
import numpy as np
import argparse
import os

from YOLOv1.modules import YOLOv1Backbone
from YOLOv1.models import YOLOv1
from data.loaders import VOC2012Loader
from util.functions import show_objects
from util.metrics import get_AP

if __name__ == '__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # a = torch.zeros(3)
    # b = torch.tensor([2, 3, 5], dtype=torch.float)
    # loss = nn.MSELoss(reduction="sum")
    # print(a, b, loss(a, b))
    # exit(-1)

    # torch.manual_seed(1)
    # m = YOLOv1Backbone()
    # d = torch.rand(8, 3, 448, 448)
    # o = m(d)
    # print(o.shape)

    p = [1, 0.6, 0.45, 0.4, 0.4, 0, 0, 0, 0, 0]
    r = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(get_AP(p, r))

    p = [0.85, 0.82, 0.75, 0.73, 0.55]
    r = [0, 0.2, 0.4, 0.6, 0.8]
    print(get_AP(p, r))

    # color_dict = {
    #     "person": (255, 0, 0),
    #     "bird": (112, 128, 105),
    #     "cat": (56, 94, 15),
    #     "cow": (8, 46, 84),
    #     "dog": (210, 105, 30),
    #     "horse": (128, 42, 42),
    #     "sheep": (255, 250, 250),
    #     "aeroplane": (0, 255, 255),
    #     "bicycle": (255, 235, 205),
    #     "boat": (210, 180, 140),
    #     "bus": (220, 220, 220),
    #     "car": (0, 0, 255),
    #     "motorbike": (250, 255, 240),
    #     "train": (127, 255, 212),
    #     "bottle": (51, 161, 201),
    #     "chair": (139, 69, 19),
    #     "diningtable": (115, 74, 18),
    #     "pottedplant": (46, 139, 87),
    #     "sofa": (160, 32, 240),
    #     "tvmonitor": (65, 105, 225)
    # }
    #
    # loader = VOC2012Loader("G:/DataSets")
    #
    # a = loader.get_data_dev()
    #
    # for i in range(200):
    #     show_objects(a[i][0], a[i][1], color_dict)
    #
    # m = YOLOv1(["person",
    #             "bird", "cat", "cow", "dog", "horse", "sheep",
    #             "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
    #             "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"], 5, 5)
    #
    # for i in tqdm(range(5)):
    #     m.train([a[i]])
