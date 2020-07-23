# coding: utf-8

import argparse

color_dict = {
    "person": (255, 0, 0),
    "bird": (112, 128, 105),
    "cat": (56, 94, 15),
    "cow": (8, 46, 84),
    "dog": (210, 105, 30),
    "horse": (128, 42, 42),
    "sheep": (255, 0, 250),
    "aeroplane": (0, 255, 255),
    "bicycle": (255, 0, 100),
    "boat": (210, 180, 140),
    "bus": (220, 220, 220),
    "car": (0, 0, 255),
    "motorbike": (200, 0, 200),
    "train": (127, 255, 212),
    "bottle": (51, 161, 201),
    "chair": (139, 69, 19),
    "diningtable": (115, 74, 18),
    "pottedplant": (46, 139, 87),
    "sofa": (160, 32, 240),
    "tvmonitor": (65, 105, 225)
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run MNIST Classifier.")
    parser.add_argument('--image_detect_path', type=str, default='',
                        help='Image path for detection. '
                             'If empty, the detection will not perform.')
    parser.add_argument('--video_detect_path', type=str, default='',
                        help='Image path for detection. '
                             'If zero, OpenCV will predict through camera. '
                             'If empty, the detection will not perform.')

    parser.add_argument('--dataset_path', type=str, default='G:/DataSets',
                        help='Dataset path.')
    parser.add_argument('--model_load_path', type=str, default='',
                        help='Input path for models.')
    parser.add_argument('--model_name', type=str, default='yolov1',
                        help='Model type. optional models: yolov1(default), yolov1-tiny. '
                             'Not required when the loading path of the model is specified.',
                        choices=["yolov1", "yolov1-tiny"])

    parser.add_argument('--graph_save_dir', type=str, default='',
                        help='Output directory for the graph of the model. '
                             'If empty, graph will not be saved.')
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help="Whether to use cuda to run the model.")
    """Arguments for training"""
    parser.add_argument('--do_train', action='store_true', default=False,
                        help="Whether to train the model on dataset.")
    parser.add_argument('--train_batch_size', type=int, default=4,
                        help='Batch size of train set.')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum of optimizer.')
    parser.add_argument('--lambda_coord', type=float, default=5,
                        help='Lambda of coordinates.')
    parser.add_argument('--lambda_noobj', type=float, default=0.5,
                        help='Lambda with no objects.')
    parser.add_argument('--clip_max_norm', type=float, default=0,
                        help='Max norm of the gradients. '
                             'If zero, the gradients will not be clipped.')
    parser.add_argument('--model_save_dir', type=str, default='',
                        help='Output directory for the model. '
                             'When empty, the model will not be saved')
    """Arguments for evaluation"""
    parser.add_argument('--do_eval', action='store_true', default=False,
                        help="Whether to evaluate the model on dataset.")
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='Batch size of evaluation set.')
    parser.add_argument('--score_threshold', type=float, default=0.1,
                        help='Threshold of score(IOU * P(Object)).')
    parser.add_argument('--iou_threshold', type=float, default=0.4,
                        help='Threshold of IOU used for calculation of NMS.')

    parser.add_argument('--iou_thresholds_mmAP', type=list,
                        default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                        help='Thresholds of IOU used for calculation of mmAP.')
    """Arguments for test"""
    parser.add_argument('--do_test', action='store_true', default=False,
                        help="Whether to test the model.")
    parser.add_argument('--test_batch_size', type=int, default=16,
                        help='Batch size of test set.')
    return parser.parse_args().__dict__


def get_args():
    arg_dict = parse_args()
    arg_dict["colors"] = color_dict
    return arg_dict
