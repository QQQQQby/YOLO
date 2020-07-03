# coding: utf-8

from YOLOv1.modules import YOLOv1Backbone, YOLOv1TinyBackbone
from util.functions import iou

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class YOLOv1:
    def __init__(
            self,
            labels,
            lr,
            momentum,
            lambda_coord,
            lambda_noobj,
            use_cuda,
            model_path=None,
            clip_max_norm=None,
            score_threshold=0.5,
            iou_threshold=0.5
    ):
        self.labels = labels
        self.lr = lr
        self.momentum = momentum
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
        self.backbone = YOLOv1TinyBackbone() if model_path is None else torch.load(model_path)
        # self.backbone = YOLOv1Backbone() if model_path is None else torch.load(model_path)
        self.backbone = self.backbone.to(self.device)

        self.optimizer = optim.SGD(self.backbone.parameters(), lr=self.lr, momentum=self.momentum)
        # self.optimizer = optim.Adam(self.backbone.parameters(), lr=self.lr)
        # self.optimizer = optim.(self.backbone.parameters(), lr=self.lr, momentum=self.momentum)

        self.clip_max_norm = clip_max_norm
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def train(self, batch):

        self.predict([data[0] for data in batch])

        self.backbone.train()
        self.optimizer.zero_grad()
        loss = self.get_loss(batch)
        loss.backward()
        if self.clip_max_norm is not None:
            nn.utils.clip_grad_norm(self.backbone.parameters(), self.clip_max_norm)
        self.optimizer.step()
        return loss

    def get_loss(self, batch):
        output_dict = self.get_output_dict([data[0] for data in batch])
        object_info_dicts = [data[1] for data in batch]

        loss = torch.tensor(0.0)
        for data_id in range(len(batch)):
            has_positive = np.zeros((7, 7), np.bool)
            for true_dict in object_info_dicts[data_id]:
                """获取label对应的栅格所在的行和列"""
                row = true_dict['y'] // 64
                col = true_dict['x'] // 64
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
                # print(pred_coord_dict, row, col)
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

    @torch.no_grad()
    def predict(self, images):
        self.backbone.eval()
        output_dict = self.get_output_dict(images)
        results = []
        for image_id, image in enumerate(images):
            results.append([])
            """使用NMS算法选择检测的目标"""
            for category in self.labels:
                candidates = []
                for row in range(7):
                    for col in range(7):
                        for bbox_id in range(2):
                            x_label = "x" + str(bbox_id)
                            y_label = "y" + str(bbox_id)
                            w_label = "w" + str(bbox_id)
                            h_label = "h" + str(bbox_id)
                            c_label = "c" + str(bbox_id)
                            score = (output_dict[c_label][image_id, row, col] *
                                     output_dict["probs"][image_id, row, col, self.labels.index(category)]).detach()
                            if score >= self.score_threshold:
                                candidates.append([
                                    row,
                                    col,
                                    score,
                                    float((output_dict[x_label][image_id, row, col] + col).detach() * (448 / 7 - 1)),
                                    float((output_dict[y_label][image_id, row, col] + row).detach() * (448 / 7 - 1)),
                                    float((output_dict[w_label][image_id, row, col]).detach() * 448),
                                    float((output_dict[h_label][image_id, row, col]).detach() * 448)
                                ])
                candidates.sort(key=lambda x: -x[2])  # 将所有候选bounding box按分数从高到低排列
                for c_i in range(len(candidates) - 1):
                    if candidates[c_i][2] > 0:
                        for c_j in range(c_i + 1, len(candidates)):
                            if iou(candidates[c_i][3:], candidates[c_j][3:], 448, 448) > self.iou_threshold:
                                candidates[c_j][2] = -1
                for c_i in range(len(candidates)):
                    if candidates[c_i][2] > 0:
                        results[-1].append({
                            "x": candidates[c_i][3],
                            "y": candidates[c_i][4],
                            "w": candidates[c_i][5],
                            "h": candidates[c_i][6],
                            "name": category
                        })
        # print(results)
        return results

    def get_output_dict(self, images):
        output_tensor = self.backbone(torch.from_numpy(np.array(images)).
                                      to(self.device))  # batch_size, 7, 7, 30
        output_dict = {"probs": output_tensor[:, :, :, :20]}
        names = ['c0', 'c1', 'x0', 'y0', 'w0', 'h0', 'x1', 'y1', 'w1', 'h1']
        for i, name in enumerate(names):
            output_dict[name] = torch.sigmoid(output_tensor[:, :, :, 20 + i])
        return output_dict

    def save(self, model_save_path):
        torch.save(self.backbone, model_save_path)

    def save_log(self, log_save_dir):
        with SummaryWriter(log_dir=log_save_dir) as writer:
            writer.add_graph(self.backbone, [torch.rand(1, 448, 448, 3)])
