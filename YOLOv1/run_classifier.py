# coding: utf-8

import torch
import os
import cv2
import time
import random
from tqdm import tqdm

from YOLOv1.modules import YOLOv1Backbone, TinyYOLOv1Backbone
from data.loaders import VOC2012Loader
from util import metrics
from util.functions import show_objects
from YOLOv1.models import YOLOv1
from YOLOv1.args import get_args


class Classifier:
    def __init__(self, args):
        print("args = {")
        for k in args:
            print("\t{} = {}".format(k, args[k]))
        print("}")
        self.args = args.copy()

        data_loader = VOC2012Loader(self.args["dataset_path"])
        self.labels = data_loader.get_labels()

        if self.args["use_cuda"] and torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device("cpu")
        if self.args["load_model"]:
            self.backbone = torch.load(self.args["model_load_path"])
        elif self.args["model_type"] == "YOLOv1":
            self.backbone = YOLOv1Backbone()
        elif self.args["model_type"] == "Tiny-YOLOv1":
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

        if self.args["single_image"] != "":
            im = cv2.imread(self.args["single_image"])
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = cv2.resize(im, (448, 448))
            pred_results = self.model.predict([im])
            print(pred_results)
            show_objects(im, pred_results[0], self.args["colors"])
            exit(0)

        print('-' * 20 + 'Reading data' + '-' * 20, flush=True)

        if not self.args["not_train"]:
            self.data_train = data_loader.get_data_train()
        if not self.args["not_eval"]:
            self.data_dev = data_loader.get_data_dev()
        if not self.args["not_test"]:
            self.data_test = data_loader.get_data_test()

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
                    end = min(start + self.args["train_batch_size"], len(self.data_train))
                    loss = self.model.train(self.data_train[start:end])
                    print(loss)
                """Save current model"""
                if self.args["save_model"]:
                    self.model.save(os.path.join(
                        self.args["model_save_dir"],
                        time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + "_" + str(epoch) + ".pth"
                    ))

            if not self.args["not_eval"]:
                """Evaluate"""
                print('-' * 20 + 'Evaluating epoch %d' % epoch + '-' * 20, flush=True)
                time.sleep(0.5)
                self.backbone.eval()
                for start in tqdm(
                        range(0, len(self.data_dev), self.args["dev_batch_size"]),
                        desc='Evaluating batch: '
                ):
                    end = min(start + self.args["dev_batch_size"], len(self.data_dev))
                    # self.model.get_mmAP(self.data_dev[start:end])
                    #
                    objects = self.model.predict([data[0] for data in self.data_dev[start:end]])
                    for i in range(start, end):
                        show_objects(self.data_dev[i][0], self.data_dev[i][1], self.args["colors"])
                        show_objects(self.data_dev[i][0], objects[i - start], self.args["colors"])

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


if __name__ == '__main__':
    classifier = Classifier(get_args())
    classifier.run()
