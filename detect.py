# coding: utf-8

import argparse

from util.loaders import read_classes, get_color_dict, read_anchors
from models import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Object detection.")
    parser.add_argument('--model_load_path', type=str, default='',
                        help='Input path to models.')
    parser.add_argument('--class_path', type=str, default='data/coco.names',
                        help='Path to a file to store names of the classes.')
    parser.add_argument('--color_path', type=str, default='data/colors',
                        help='Path to a file which stores colors.')
    parser.add_argument('--anchor_path', type=str, default='data/anchors',
                        help='Input path to anchors.')

    parser.add_argument('--input_path', type=str, default='',
                        help='Path to the file used for detection. '
                             'If zero, camera on your computer will be used.')
    parser.add_argument('--output_path', type=str, default='',
                        help='Path to the output image or video. '
                             'If Empty, the predicted image will not be saved.')
    parser.add_argument('--do_show', action='store_true', default=False,
                        help="Whether to show predictions.")

    parser.add_argument('--score_threshold', type=float, default=0.5,
                        help='Threshold of score(IOU * P(Object)).')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                        help='Threshold of IOU used for calculation of NMS.')
    parser.add_argument('--device_ids', type=str, default='-1',
                        help="Device ids. "
                             "Should be seperated by commas. "
                             "-1 means cpu.")
    parser.add_argument('--num_processes', type=int, default=0,
                        help='number of processes.')
    return parser.parse_args().__dict__


if __name__ == '__main__':
    args = parse_args()
    classes = read_classes(args["class_path"])
    color_dict = get_color_dict(classes, args["color_path"])
    anchors = read_anchors(args["anchor_path"]) if args["anchor_path"] else None
    model = YOLO(classes,
                 model_load_path=args["model_load_path"],
                 anchors=anchors,
                 device_ids=args["device_ids"])
    if args["input_path"].endswith(".jpg") or \
            args["input_path"].endswith(".jpeg") or \
            args["input_path"].endswith(".png"):
        model.detect_image(
            args["input_path"],
            args["score_threshold"],
            args["iou_threshold"],
            color_dict,
            do_show=args["do_show"],
            delay=0,
            output_path=args["output_path"] if args["output_path"] else None
        )
    elif args["input_path"].endswith(".mp4"):
        model.detect_video(
            args["input_path"],
            args["score_threshold"],
            args["iou_threshold"],
            color_dict,
            do_show=args["do_show"],
            delay=1,
            output_path=args["output_path"] if args["output_path"] else None
        )
    elif args["input_path"] == '0':
        model.detect_video(
            0,
            args["score_threshold"],
            args["iou_threshold"],
            color_dict,
            do_show=args["do_show"],
            delay=1,
            output_path=args["output_path"] if args["output_path"] else None
        )
    else:
        print("Wrong type!")
        exit(-1)
