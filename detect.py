# coding: utf-8

import argparse
import torch

from util.loaders import read_classes, get_color_dict
from util.models import YOLO
from YOLOv1.modules import YOLOv1Backbone, TinyYOLOv1Backbone
from YOLOv3.modules import YOLOv3Backbone


def parse_args():
    parser = argparse.ArgumentParser(description="Object detection.")
    parser.add_argument('--model_load_path', type=str, default='',
                        help='Input path for models.')
    parser.add_argument('--class_path', type=str, default='',
                        help='Path for a file to store names and colors of the classes.')
    parser.add_argument('--color_path', type=str, default='data/colors',
                        help='Path to a file which stores colors.')

    parser.add_argument('--type', type=str, default='image',
                        help='Select detect type.',
                        choices=['image', 'video', 'camera'])
    parser.add_argument('--file_path', type=str, default='',
                        help='Path to the file used for detection.')
    parser.add_argument('--device_ids', type=str, default='-1',
                        help="Device ids. "
                             "Should be seperated by commas. "
                             "-1 means cpu.")
    parser.add_argument('--score_threshold', type=float, default=0.1,
                        help='Threshold of score(IOU * P(Object)).')
    parser.add_argument('--iou_threshold', type=float, default=0.4,
                        help='Threshold of IOU used for calculation of NMS.')
    parser.add_argument('--num_processes', type=int, default=0,
                        help='number of processes.')
    return parser.parse_args().__dict__


if __name__ == '__main__':
    args = parse_args()
    classes = read_classes(args["class_path"])
    color_dict = get_color_dict(classes, args["color_path"])
    model = YOLO(classes, model_load_path=args["model_load_path"], device_ids=args["device_ids"])
    if args["type"] == 'image':
        model.detect_image_and_show(
            args["file_path"],
            args["score_threshold"],
            args["iou_threshold"],
            color_dict,
            0
        )
    elif args["type"] == 'video':
        model.detect_video_and_show(
            args["file_path"],
            args["score_threshold"],
            args["iou_threshold"],
            color_dict,
            1
        )
    elif args["type"] == 'camera':
        model.detect_video_and_show(
            0,
            args["score_threshold"],
            args["iou_threshold"],
            color_dict,
            1
        )
