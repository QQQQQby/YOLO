# coding: utf-8

import torch
import os
import cv2
import time
import random
from tqdm import tqdm
import time

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

        self.data_loader = VOC2012Loader(self.args["dataset_path"], num_processes=4)
        self.labels = self.data_loader.get_labels()

        if self.args["use_cuda"] and torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device("cpu")
        if self.args["model_load_path"] != "":
            self.backbone = torch.load(self.args["model_load_path"])
        elif self.args["model_type"] in ["", "YOLOv1"]:
            self.backbone = YOLOv1Backbone()
        elif self.args["model_type"] == "Tiny-YOLOv1":
            self.backbone = TinyYOLOv1Backbone()
        self.backbone = self.backbone.to(self.device)

        if self.args["model_save_dir"] != "" and not os.path.exists(self.args["model_save_dir"]):
            os.makedirs(self.args["model_save_dir"])
        if self.args["graph_save_dir"] != "" and not os.path.exists(self.args["graph_save_dir"]):
            os.makedirs(self.args["graph_save_dir"])

        self.model = YOLOv1(
            self.backbone,
            self.device,
            self.labels,
            self.args
        )

    def run(self):
        if self.args["image_detect_path"] != "":
            self.model.detect_image_and_show(
                self.args["image_detect_path"],
                self.args["colors"],
                0
            )

        if not any([self.args["do_train"], self.args["do_eval"], self.args["do_test"]]):
            return None

        print('-' * 20 + 'Reading data' + '-' * 20, flush=True)
        data_train = self.data_loader.get_data_train() if self.args["do_train"] else None
        data_eval = self.data_loader.get_data_eval() if self.args["do_eval"] else None
        data_test = self.data_loader.get_data_test() if self.args["do_test"] else None
        if self.args["graph_save_dir"] != "":
            self.model.save_graph(self.args["graph_save_dir"])
        for epoch in range(self.args["num_epochs"]):
            if self.args["do_train"]:
                """Train"""
                print('-' * 20 + 'Training epoch %d' % epoch + '-' * 20, flush=True)
                time.sleep(0.5)
                random.shuffle(data_train)  # 打乱训练数据
                self.backbone.train()
                for start in tqdm(
                        range(0, len(data_train), self.args["train_batch_size"]),
                        desc='Training batch: '
                ):
                    end = min(start + self.args["train_batch_size"], len(data_train))
                    loss = self.model.train(data_train[start:end])
                    print(loss)
                """Save current model"""
                if self.args["model_save_dir"] != "":
                    self.model.save(os.path.join(
                        self.args["model_save_dir"],
                        time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + "_" + str(epoch) + ".pth"
                    ))

            if self.args["do_eval"]:
                """Evaluate"""
                print('-' * 20 + 'Evaluating epoch %d' % epoch + '-' * 20, flush=True)
                time.sleep(0.5)
                self.backbone.eval()
                pred_results = []
                for start in tqdm(
                        range(0, len(data_eval), self.args["eval_batch_size"]),
                        desc='Evaluating batch: '
                ):
                    end = min(start + self.args["eval_batch_size"], len(data_eval))
                    pred_results += self.model.predict([data[0] for data in data_eval[start:end]], num_processes=4)
                mmAP = self.model.get_mmAP(data_eval, pred_results)
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


if __name__ == '__main__':
    classifier = Classifier(get_args())
    classifier.run()
