# coding: utf-8

import torch
import os
import random
from tqdm import tqdm
import time

from util.loaders import VOCDataLoader, read_classes, get_color_dict
from models import YOLO
import argparse


class Classifier:
    def __init__(self, args):
        print("args = {")
        for k in args:
            print("\t{} = {}".format(k, args[k]))
        print("}")
        self.args = args.copy()

        if self.args["dataset_path"] != "":
            self.data_loader = VOCDataLoader(self.args["dataset_path"], num_processes=4, preload=self.args["preload"])

        self.classes = read_classes(self.args["class_path"])
        self.color_dict = get_color_dict(self.classes, self.args["color_path"])

        if self.args["model_save_dir"] != "" and not os.path.exists(self.args["model_save_dir"]):
            os.makedirs(self.args["model_save_dir"])
        if self.args["graph_save_dir"] != "" and not os.path.exists(self.args["graph_save_dir"]):
            os.makedirs(self.args["graph_save_dir"])

        self.yolo = YOLO()

    def run(self):
        if self.args["image_detect_path"] != "":
            self.yolo.detect_image_and_show(
                self.args["image_detect_path"],
                self.color_dict,
                0
            )

        if self.args["video_detect_path"] != "":
            self.yolo.detect_video_and_show(
                self.args["video_detect_path"],
                self.color_dict,
            )
        if not any([self.args["do_train"], self.args["do_eval"], self.args["do_test"]]):
            return None

        print('-' * 20 + 'Reading data' + '-' * 20, flush=True)
        data_train = self.data_loader.get_data_train() if self.args["do_train"] else []
        data_eval = self.data_loader.get_data_eval() if self.args["do_eval"] else []
        data_test = self.data_loader.get_data_test() if self.args["do_test"] else []

        print('-' * 20 + 'Preprocessing data' + '-' * 20, flush=True)
        for data_id in range(len(data_train)):
            data_train[data_id][0] = self.yolo.preprocess_image(*(data_train[data_id]), cvt_RGB=True)[0]
        for data_id in range(len(data_eval)):
            data_eval[data_id][0] = self.yolo.preprocess_image(*(data_eval[data_id]), cvt_RGB=True)[0]
        for data_id in range(len(data_test)):
            data_test[data_id][0] = self.yolo.preprocess_image(*(data_test[data_id]), cvt_RGB=True)[0]

        if self.args["graph_save_dir"] != "":
            self.yolo.save_graph(self.args["graph_save_dir"])
        for epoch in range(self.args["num_epochs"]):
            if self.args["do_train"]:
                """Train"""
                print('-' * 20 + 'Training epoch %d' % epoch + '-' * 20, flush=True)
                time.sleep(0.5)
                random.shuffle(data_train)  # 打乱训练数据
                for start in tqdm(
                        range(0, len(data_train), self.args["train_batch_size"]),
                        desc='Training batch: '
                ):
                    end = min(start + self.args["train_batch_size"], len(data_train))
                    loss = self.yolo.train(data_train[start:end])
                    print(loss)
                """Save current model"""
                if self.args["model_save_dir"] != "":
                    self.yolo.save(os.path.join(
                        self.args["model_save_dir"],
                        time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + "_" + str(epoch) + ".pth"
                    ))

            if self.args["do_eval"]:
                """Evaluate"""
                print('-' * 20 + 'Evaluating epoch %d' % epoch + '-' * 20, flush=True)
                time.sleep(0.5)
                pred_results = []
                for start in tqdm(
                        range(0, len(data_eval), self.args["eval_batch_size"]),
                        desc='Evaluating batch: '
                ):
                    end = min(start + self.args["eval_batch_size"], len(data_eval))
                    pred_results += self.yolo.predict([data[0] for data in data_eval[start:end]], num_processes=0)
                mmAP = self.yolo.get_mmAP(data_eval, pred_results)
                print("mmAP =", mmAP)
                if not self.args["do_train"]:
                    break

            if self.args["do_test"]:
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
                if not self.args["do_train"]:
                    break
            print()


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate.")
    parser.add_argument('--image_detect_path', type=str, default='',
                        help='Image path for detection. '
                             'If empty, the detection will not perform.')
    parser.add_argument('--video_detect_path', type=str, default='',
                        help='Image path for detection. '
                             'If zero, OpenCV will predict through camera. '
                             'If empty, the detection will not perform.')

    parser.add_argument('--dataset_path', type=str, default='',
                        help='Dataset path.')
    parser.add_argument('--preload', action='store_true', default=False,
                        help="Whether to preload the dataset.")
    parser.add_argument('--model_load_path', type=str, default='',
                        help='Input path for models.')
    parser.add_argument('--model_name', type=str, default='yolov1',
                        help='Model type. '
                             'Not required when the loading path of the model is specified.',
                        choices=['yolov1', 'yolov1-tiny', 'yolov3'])
    parser.add_argument('--class_path', type=str, default='data/voc.names',
                        help='Path to a file which stores names of the classes.')
    parser.add_argument('--color_path', type=str, default='data/colors',
                        help='Path to a file which stores colors.')

    parser.add_argument('--graph_save_dir', type=str, default='',
                        help='Output directory for the graph of the model. '
                             'If empty, graph will not be saved.')
    parser.add_argument('--device_ids', type=str, default='-1',
                        help="Device ids. "
                             "Should be seperated by commas. "
                             "-1 means cpu.")
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


if __name__ == '__main__':
    classifier = Classifier(parse_args())
    classifier.run()
