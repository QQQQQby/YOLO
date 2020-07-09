# coding: utf-8

import torch
import argparse
import os
import time
import random
from tqdm import tqdm

from YOLOv1.modules import YOLOv1Backbone, TinyYOLOv1Backbone
from data.loaders import VOC2012Loader
from util import metrics
from util.functions import show_objects
from YOLOv1.models import YOLOv1

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


class Classifier:
    def __init__(self, args):
        print("args = {")
        for k in args:
            print("\t{} = {}".format(k, args[k]))
        print("}")
        self.args = args.copy()

        print('-' * 20 + 'Reading data' + '-' * 20, flush=True)
        data_loader = VOC2012Loader(self.args["dataset_path"])
        self.data_train = data_loader.get_data_train()
        self.data_dev = data_loader.get_data_dev()
        self.data_test = data_loader.get_data_test()
        self.labels = data_loader.get_labels()

        if self.args["use_cuda"] and torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device("cpu")
        if self.args["load_model"]:
            self.backbone = torch.load(self.args["model_path"])
        elif self.args["model_type"] == "YOLOv1":
            self.backbone = YOLOv1Backbone()
        elif self.args["model_type"] == "tiny-YOLOv1":
            self.backbone = TinyYOLOv1Backbone()
        self.backbone = self.backbone.to(self.device)

        if not os.path.exists(self.args["model_save_dir"]):
            os.makedirs(self.args["model_save_dir"])
        if not os.path.exists(self.args["log_save_dir"]):
            os.makedirs(self.args["log_save_dir"])

        self.model = YOLOv1(
            self.backbone,
            self.device,
            self.labels,
            self.args["lr"],
            self.args["momentum"],
            self.args["lambda_coord"],
            self.args["lambda_noobj"],
            self.args["clip_max_norm"] if self.args["clip_grad"] else None,
            self.args["score_threshold"],
            self.args["iou_threshold_pred"],
            self.args["iou_thresholds_mmAP"]
        )

    def run(self):
        self.model.save_log(self.args["log_save_dir"])
        for epoch in range(self.args["num_epochs"]):
            if not self.args["not_train"]:
                """Train"""
                print('-' * 20 + 'Training epoch %d' % epoch + '-' * 20, flush=True)
                time.sleep(0.5)
                random.shuffle(self.data_train)  # 打乱训练数据
                self.backbone.train()
                for start in tqdm(
                        range(0, len(self.data_train), self.args["train_batch_size"]),
                        desc='Training batch: '
                ):
                    self.model.get_mmAP(self.data_train[start:start + self.args["train_batch_size"]])
                    self.model.train(self.data_train[start:start + self.args["train_batch_size"]])
                """Save current model"""
                if self.args["save_model"]:
                    self.model.save(os.path.join(self.args["model_save_dir"], str(epoch) + ".pd"))

            if not self.args["not_eval"]:
                """Evaluate"""
                print('-' * 20 + 'Evaluating epoch %d' % epoch + '-' * 20, flush=True)
                time.sleep(0.5)
                self.backbone.eval()
                for start in tqdm(
                        range(0, len(self.data_dev), self.args["dev_batch_size"]),
                        desc='Evaluating batch: '
                ):
                    pass
                    # """forward and show image"""
                    # for image in [data[0] for data in self.data_dev[start:start + self.args["dev_batch_size"]]]:
                    #     pred_objects = self.model.predict([image])[0]
                    #     # print(pred_objects)
                    #     if len(pred_objects) != 0:
                    #         show_objects(image, pred_objects, color_dict)
                    """Calculate mmAP"""

            if not self.args["not_test"]:
                pass
                # """Test"""
                # print('-' * 20 + 'Testing epoch %d' % epoch + '-' * 20, flush=True)
                # time.sleep(0.1)
                # m = metrics.Metrics(self.labels)
                # for start in tqdm(range(0, len(self.data_test), self.args.test_batch_size),
                #                   desc='Testing batch: '):
                #     images = [d[0] for d in self.data_test[start:start + self.args.test_batch_size]]
                #     actual_labels = [d[1] for d in self.data_test[start:start + self.args.test_batch_size]]
                #     """forward"""
                #     batch_images = torch.tensor(images, dtype=torch.float32)
                #     outputs = self.model(batch_images)
                #     """update confusion matrix"""
                #     pred_labels = outputs.softmax(1).argmax(1).tolist()
                #     m.update(actual_labels, pred_labels)
                # """testing"""
                # print(m.get_accuracy())
            print()


def parse_args():
    parser = argparse.ArgumentParser(description="Run MNIST Classifier.")
    parser.add_argument('--dataset_path', type=str, default='G:/DataSets',
                        help='Dataset path.')
    parser.add_argument('--model_type', type=str, default='tiny-YOLOv1',
                        help='Model type. optional models: YOLOv1, tiny-YOLOv1.')
    parser.add_argument('--load_model', action='store_true', default=False,
                        help="Whether to load the model from specific directory.")
    parser.add_argument('--model_load_path', type=str, default='../models/0.pd',
                        help='Input path for models.')

    parser.add_argument('--log_save_dir', type=str, default='../log',
                        help='Output directory for logs.')
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help="Whether to use cuda to run the model.")
    """Arguments for training"""
    parser.add_argument('--not_train', action='store_true', default=False,
                        help="Whether not to train the model.")
    parser.add_argument('--train_batch_size', type=int, default=32,
                        help='Batch size of train set.')
    parser.add_argument('--num_epochs', type=int, default=200,
                        help='Number of epochs.')
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='Learning rate.')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum of optimizer.')
    parser.add_argument('--lambda_coord', type=float, default=5,
                        help='Lambda of coordinates.')
    parser.add_argument('--lambda_noobj', type=float, default=5,
                        help='Lambda with no objects.')
    parser.add_argument('--clip_grad', action='store_true', default=False,
                        help="Whether to clip gradients.")
    parser.add_argument('--clip_max_norm', type=float, default=1000,
                        help='Max norm of the gradients.')
    parser.add_argument('--save_model', action='store_true', default=False,
                        help="Whether to save the model after training.")
    parser.add_argument('--model_save_dir', type=str, default='../output/test/',
                        help='Output directory for the model.')
    """Arguments for evaluation"""
    parser.add_argument('--not_eval', action='store_true', default=False,
                        help="Whether not to evaluate the model.")
    parser.add_argument('--dev_batch_size', type=int, default=2,
                        help='Batch size of dev set.')
    parser.add_argument('--score_threshold', type=float, default=0.2,
                        help='Threshold of score(IOU * P(Object)).')
    parser.add_argument('--iou_threshold_pred', type=float, default=0.5,
                        help='Threshold of IOU used for calculation of NMS.')
    parser.add_argument('--iou_thresholds_mmAP', type=list,
                        default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
                        help='Thresholds of IOU used for calculation of mmAP.')
    """Arguments for test"""
    parser.add_argument('--not_test', action='store_true', default=False,
                        help="Whether not to test the model.")
    parser.add_argument('--test_batch_size', type=int, default=2,
                        help='Batch size of test set.')
    return parser.parse_args().__dict__


if __name__ == '__main__':
    classifier = Classifier(parse_args())
    classifier.run()
