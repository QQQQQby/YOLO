# coding: utf-8

import torch
from torch import nn
from xml.dom import minidom
import cv2
import numpy as np
import argparse

from YOLOv1.modules import YOLOv1Backbone
# from YOLOv1.models import YOLOv1
from data.loaders import VOC2012Loader


def show_objects(image_array, objects, color_dict):
    image = image_array[:, :, ::-1]
    for o in objects:
        upper_left = (int(o['x'] - o['w'] / 2), int(o['y'] - o['h'] / 2))
        lower_right = (int(o['x'] + o['w'] / 2), int(o['y'] + o['h'] / 2))
        cv2.rectangle(image, upper_left, lower_right, color_dict[o['name']][::-1], 2)
        # cv2.rectangle(image, upper_left, lower_right, (255, 0, 0)[::-1], 1)

        cv2.rectangle(
            image,
            upper_left,
            (upper_left[0] + len(o['name'] * 8), max(upper_left[1] - 12, 16)),
            (0, 0, 0),
            -1
        )

        upper_left = (upper_left[0] + 2, max(upper_left[1] - 4, 12))
        cv2.putText(image, o['name'], upper_left, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255)[::-1], 1)
    cv2.imshow('Object Detection', image)
    cv2.waitKey(1000)


def parse_args():
    parser = argparse.ArgumentParser(description="Run MNIST Classifier.")
    parser.add_argument('--dataset_path', type=str, default='./data',
                        help='Dataset path.')
    parser.add_argument('--output_path', type=str, default='./output/1000_0.01_dropout0.7',
                        help='Output path.')

    parser.add_argument('--not_train', action='store_true', default=False,
                        help="Whether not to train the model.")
    parser.add_argument('--save', action='store_true', default=False,
                        help="Whether to save the model after training.")
    parser.add_argument('--train_batch_size', type=int, default=1000,
                        help='Batch size of train set.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate.')

    parser.add_argument('--not_eval', action='store_true', default=False,
                        help="Whether not to evaluate the model.")
    parser.add_argument('--dev_batch_size', type=int, default=2000,
                        help='Batch size of dev set.')

    parser.add_argument('--not_test', action='store_true', default=False,
                        help="Whether not to test the model.")
    parser.add_argument('--test_batch_size', type=int, default=2000,
                        help='Batch size of test set.')
    args = parser.parse_args().__dict__
    return args


if __name__ == '__main__':
    # if torch.cuda.is_available():
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    # torch.manual_seed(1)
    # m = YOLOv1Backbone()
    # d = torch.rand(8, 3, 448, 448)
    # o = m(d)
    # print(o.shape)

    color_dict = {
        "person": (255, 0, 0),
        "bird": (112, 128, 105),
        "cat": (56, 94, 15),
        "cow": (8, 46, 84),
        "dog": (210, 105, 30),
        "horse": (128, 42, 42),
        "sheep": (255, 250, 250),
        "aeroplane": (0, 255, 255),
        "bicycle": (255, 235, 205),
        "boat": (210, 180, 140),
        "bus": (220, 220, 220),
        "car": (0, 0, 255),
        "motorbike": (250, 255, 240),
        "train": (127, 255, 212),
        "bottle": (51, 161, 201),
        "chair": (139, 69, 19),
        "diningtable": (115, 74, 18),
        "pottedplant": (46, 139, 87),
        "sofa": (160, 32, 240),
        "tvmonitor": (65, 105, 225)
    }

    loader = VOC2012Loader("G:/DataSets")

    a = loader.get_data_dev()

    for i in range(200):
        show_objects(a[i][0], a[i][1], color_dict)

    m = YOLOv1(["person",
                "bird", "cat", "cow", "dog", "horse", "sheep",
                "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train",
                "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"], 5, 5)

    # m.train(a[:5])
