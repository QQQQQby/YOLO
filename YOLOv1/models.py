# coding: utf-8

from YOLOv1.modules import YOLOv1Backbone, YOLOv1TinyBackbone
from util.functions import iou

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class YOLOv1:
    def __init__(self, labels, lr, momentum, lambda_coord, lambda_noobj, use_cuda, log_path):
        self.labels = labels
        self.lr = lr
        self.momentum = momentum
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        # self.backbone = YOLOv1Backbone().to(self.device)
        self.backbone = YOLOv1TinyBackbone().to(self.device)

        self.log_path = log_path
        with SummaryWriter(log_dir=log_path) as writer:
            writer.add_graph(self.backbone, [torch.rand(3, 448, 448, 3)])

        # self.optimizer = optim.SGD(self.backbone.parameters(), lr=self.lr, momentum=self.momentum)
        self.optimizer = optim.Adam(self.backbone.parameters(), lr=self.lr)

    def predict(self, batch):
        # batch: [[image, object_info_dict] * batch_size]
        self.backbone.eval()
        output_dict = self.get_output_dict([data[0] for data in batch])
        for data_id, data in enumerate(batch):
            pass

    def train(self, batch):
        # batch: [[image, object_info_dict] * batch_size]
        self.backbone.train()
        loss = self.get_loss(batch)
        loss.backward()
        self.optimizer.step()
        print(loss)
        return loss

    def get_loss(self, batch):
        output_dict = self.get_output_dict([data[0] for data in batch])
        object_info_dicts = [data[1] for data in batch]

        loss = torch.tensor(0.0)
        for data_id in range(len(batch)):
            has_positive = np.zeros((7, 7), np.bool)
            for true_dict in object_info_dicts[data_id]:
                """获取label对应的栅格所在的行和列"""
                row = true_dict['x'] // 64
                col = true_dict['y'] // 64
                ious = []
                pred_coord_dict = {}
                for bbox_id in range(2):
                    x_label = "x" + str(bbox_id)
                    y_label = "y" + str(bbox_id)
                    w_label = "w" + str(bbox_id)
                    h_label = "h" + str(bbox_id)
                    """计算预测的坐标"""
                    pred_coord_dict[x_label] = (output_dict[x_label][data_id, row, col] + col) * (448 / 7 - 1)
                    pred_coord_dict[y_label] = (output_dict[y_label][data_id, row, col] + row) * (448 / 7 - 1)
                    pred_coord_dict[w_label] = output_dict[w_label][data_id, row, col] * 448
                    pred_coord_dict[h_label] = output_dict[h_label][data_id, row, col] * 448

                    """计算gt与dt的IOU"""
                    ious.append(iou((float(pred_coord_dict[x_label]),
                                     float(pred_coord_dict[y_label]),
                                     float(pred_coord_dict[w_label]),
                                     float(pred_coord_dict[h_label])),
                                    (true_dict['x'],
                                     true_dict['y'],
                                     true_dict['w'],
                                     true_dict['h']),
                                    448, 448))
                """取IOU较大的bounding box进行坐标损失和置信度损失的计算"""
                chosen_bbox_id = np.argmax(ious)
                x_label = "x" + str(chosen_bbox_id)
                y_label = "y" + str(chosen_bbox_id)
                w_label = "w" + str(chosen_bbox_id)
                h_label = "h" + str(chosen_bbox_id)
                c_label = "c" + str(chosen_bbox_id)
                loss += self.lambda_coord * ((pred_coord_dict[x_label] - true_dict['x']) ** 2 +
                                             (pred_coord_dict[y_label] - true_dict['y']) ** 2 +
                                             (pred_coord_dict[w_label] ** 0.5 - true_dict['w'] ** 0.5) ** 2 +
                                             (pred_coord_dict[h_label] ** 0.5 - true_dict['h'] ** 0.5) ** 2)
                loss += (output_dict[c_label][data_id, row, col] - 1) ** 2
                """未被选中的(即IOU较小的)bounding box，取其置信度为0进行损失计算"""
                for bbox_id in range(2):
                    if bbox_id == chosen_bbox_id:
                        continue
                    c_label = "c" + str(bbox_id)
                    loss += self.lambda_noobj * (output_dict[c_label][data_id, row, col] - 0) ** 2
                """概率损失"""
                prob_loss = nn.MSELoss(reduction="sum")
                true_porbs = torch.zeros((20,))
                true_porbs[self.labels.index(true_dict['name'])] = 1
                loss += prob_loss(
                    output_dict['probs'][data_id, row, col],
                    true_porbs
                )
                """统计有gt的栅格"""
                has_positive[row, col] = True
            """取未被选中的(即IOU较小的)栅格中的两个bounding box置信度为0"""
            for i in range(7):
                for j in range(7):
                    if not has_positive[i, j]:
                        for bbox_id in range(2):
                            c_label = "c" + str(bbox_id)
                            loss += self.lambda_noobj * (output_dict[c_label][data_id, i, j] - 0) ** 2
        return loss

    def get_output_dict(self, images):
        output_tensor = self.backbone(torch.from_numpy(np.array(images)).
                                      to(self.device))  # batch_size, 7, 7, 30
        output_dict = {"probs": output_tensor[:, :, :, :20]}
        names = ['c0', 'c1', 'x0', 'y0', 'w0', 'h0', 'x1', 'y1', 'w1', 'h1']
        for i, name in enumerate(names):
            output_dict[name] = torch.sigmoid(output_tensor[:, :, :, 20 + i])
        return output_dict

    def save(self, path):
        torch.save(self.backbone, path)
