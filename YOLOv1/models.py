# coding: utf-8

from YOLOv1.modules import YOLOv1Backbone
from util.functions import iou

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class YOLOv1:
    def __init__(self, labels, lr, momentum, lambda_coord, lambda_noobj, use_cuda, output_path):
        self.labels = labels
        self.lr = lr
        self.momentum = momentum
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.backbone = YOLOv1Backbone().to(self.device)
        with SummaryWriter(log_dir=output_path) as writer:
            writer.add_graph(self.backbone, [torch.rand(3,  448, 448, 3)])

        # self.op = optim.SGD(self.backbone.parameters(), lr=self.lr, momentum=self.momentum)
        self.op = optim.Adam(self.backbone.parameters(), lr=self.lr)


    def predict(self, batch):
        pass
        # for d in batch:


    def train(self, batch):
        self.backbone.train()
        loss = self.get_loss(batch)
        loss.backward()
        self.op.step()
        print(loss)

    def get_loss(self, batch):
        output_tensor = self.backbone(torch.from_numpy(np.array([d[0] for d in batch])).
                                      to(self.device))  # batch_size, 7, 7, 30
        output_dict = {"probs": output_tensor[:, :, :, :20]}
        names = ['c0', 'c1', 'x0', 'y0', 'w0', 'h0', 'x1', 'y1', 'w1', 'h1']
        for i, name in enumerate(names):
            output_dict[name] = torch.sigmoid(output_tensor[:, :, :, 20 + i])

        batch_dicts = [d[1] for d in batch]

        loss = torch.tensor(0.0)
        for data_id in range(len(batch)):
            has_positive = np.zeros((7, 7), np.bool)
            for true_dict in batch_dicts[data_id]:
                """获取label对应的栅格所在的行和列"""
                row = true_dict['x'] // 64
                col = true_dict['y'] // 64
                """计算预测的坐标"""
                pred_coord_dict = {
                    "x0": (output_dict['x0'][data_id, row, col] + col) * (448 / 7 - 1),
                    "y0": (output_dict['y0'][data_id, row, col] + row) * (448 / 7 - 1),
                    "w0": output_dict['w0'][data_id, row, col] * 448,
                    "h0": output_dict['h0'][data_id, row, col] * 448,
                    "x1": (output_dict['x1'][data_id, row, col] + col) * (448 / 7 - 1),
                    "y1": (output_dict['y1'][data_id, row, col] + row) * (448 / 7 - 1),
                    "w1": output_dict['w1'][data_id, row, col] * 448,
                    "h1": output_dict['h1'][data_id, row, col] * 448
                }

                """分别计算两个bounding box的IOU"""
                iou0 = iou((float(pred_coord_dict["x0"]),
                            float(pred_coord_dict["y0"]),
                            float(pred_coord_dict["w0"]),
                            float(pred_coord_dict["h0"])),
                           (true_dict['x'],
                            true_dict['y'],
                            true_dict['w'],
                            true_dict['h']),
                           448, 448)
                iou1 = iou((float(pred_coord_dict["x1"]),
                            float(pred_coord_dict["y1"]),
                            float(pred_coord_dict["w1"]),
                            float(pred_coord_dict["h1"])),
                           (true_dict['x'],
                            true_dict['y'],
                            true_dict['w'],
                            true_dict['h']),
                           448, 448)
                """取IOU较大的bounding box进行坐标损失的计算，取其置信度为1"""
                if iou0 > iou1:
                    loss += self.lambda_coord * ((pred_coord_dict["x0"] - true_dict['x']) ** 2 +
                                                 (pred_coord_dict["y0"] - true_dict['y']) ** 2 +
                                                 (pred_coord_dict["y0"] ** 0.5 - true_dict['w'] ** 0.5) ** 2 +
                                                 (pred_coord_dict["y0"] ** 0.5 - true_dict['h'] ** 0.5) ** 2)
                    loss += (output_dict['c0'][data_id, row, col] - 1) ** 2 + \
                            self.lambda_noobj * (output_dict['c1'][data_id, row, col] - 0) ** 2
                else:
                    loss += self.lambda_coord * ((pred_coord_dict["x1"] - true_dict['x']) ** 2 +
                                                 (pred_coord_dict["y1"] - true_dict['y']) ** 2 +
                                                 (pred_coord_dict["y1"] ** 0.5 - true_dict['w'] ** 0.5) ** 2 +
                                                 (pred_coord_dict["y1"] ** 0.5 - true_dict['h'] ** 0.5) ** 2)
                    loss += (output_dict['c1'][data_id, row, col] - 1) ** 2 + \
                            self.lambda_noobj * (output_dict['c0'][data_id, row, col] - 0) ** 2
                """概率损失"""
                prob_loss = nn.MSELoss(reduction="sum")
                true_porbs = torch.zeros((20,))
                true_porbs[self.labels.index(true_dict['name'])] = 1
                loss += prob_loss(
                    output_dict['probs'][data_id, row, col],
                    true_porbs
                )
                has_positive[row, col] = True
            for i in range(7):
                for j in range(7):
                    if not has_positive[i, j]:
                        loss += self.lambda_noobj * (output_dict['c0'][data_id, i, j] - 0) ** 2

        return loss
