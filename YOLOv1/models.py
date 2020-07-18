# coding: utf-8

from util.functions import iou
from util.metrics import get_precision_and_recall, get_AP

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np


class YOLOv1:
    def __init__(
            self,
            backbone,
            device,
            labels,
            lr,
            momentum,
            lambda_coord,
            lambda_noobj,
            clip_max_norm=None,
            score_threshold=0.5,
            iou_threshold_pred=0.2,
            iou_thresholds_mmAP=None
    ):
        self.backbone = backbone
        self.device = device
        self.labels = labels
        self.lr = lr
        self.momentum = momentum
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

        self.optimizer = optim.SGD(self.backbone.parameters(), lr=self.lr, momentum=self.momentum)
        # self.optimizer = optim.Adam(self.backbone.parameters(), lr=self.lr)
        # self.optimizer = optim.(self.backbone.parameters(), lr=self.lr, momentum=self.momentum)

        self.clip_max_norm = clip_max_norm
        self.score_threshold = score_threshold
        self.iou_threshold_pred = iou_threshold_pred
        if iou_thresholds_mmAP is None:
            self.iou_thresholds_mmAP = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        else:
            self.iou_thresholds_mmAP = iou_thresholds_mmAP

    def train(self, batch):
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
                row = int(true_dict['y'] // (448 / 7))  # [0, 6]
                col = int(true_dict['x'] // (448 / 7))  # [0, 6]
                ious = []
                pred_coord_dict = {}
                for bbox_id in range(2):
                    x_label = "x" + str(bbox_id)
                    y_label = "y" + str(bbox_id)
                    w_label = "w" + str(bbox_id)
                    h_label = "h" + str(bbox_id)
                    """计算预测的坐标"""
                    pred_coord_dict[x_label] = (output_dict[x_label][data_id, row, col] + col) * ((448 - 1) / 7)
                    # [0, 448 - 1]
                    pred_coord_dict[y_label] = (output_dict[y_label][data_id, row, col] + row) * ((448 - 1) / 7)
                    # [0, 448 - 1]
                    pred_coord_dict[w_label] = output_dict[w_label][data_id, row, col] * 448
                    pred_coord_dict[h_label] = output_dict[h_label][data_id, row, col] * 448
                    """计算gt与dt的IOU"""
                    ious.append(iou(
                        (int(pred_coord_dict[x_label]),
                         int(pred_coord_dict[y_label]),
                         int(pred_coord_dict[w_label]),
                         int(pred_coord_dict[h_label])),
                        (true_dict['x'],
                         true_dict['y'],
                         true_dict['w'],
                         true_dict['h']))
                    )
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
                # print("Coordinate loss =",
                #       self.lambda_coord * ((pred_coord_dict[x_label] - true_dict['x']) ** 2 +
                #                            (pred_coord_dict[y_label] - true_dict['y']) ** 2 +
                #                            (pred_coord_dict[w_label] ** 0.5 - true_dict['w'] ** 0.5) ** 2 +
                #                            (pred_coord_dict[h_label] ** 0.5 - true_dict['h'] ** 0.5) ** 2))
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
                # print("Prob loss =", prob_loss(output_dict['probs'][data_id, row, col], true_porbs))
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

    def predict(self, images, output_dict=None):
        if output_dict is None:
            with torch.no_grad():
                output_dict = self.get_output_dict(images)
        results = []
        for image_id in range(len(images)):
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
                                     output_dict["probs"][image_id, row, col, self.labels.index(category)]) \
                                .detach().cpu().numpy()
                            # print("score = " + str(score))
                            if score >= self.score_threshold:
                                candidates.append({
                                    "name": category,
                                    "score": score,
                                    "x": float(
                                        (output_dict[x_label][image_id, row, col] + col).detach() * ((448 - 1) / 7)),
                                    "y": float(
                                        (output_dict[y_label][image_id, row, col] + row).detach() * ((448 - 1) / 7)),
                                    "w": float((output_dict[w_label][image_id, row, col]).detach()) * 448,
                                    "h": float((output_dict[h_label][image_id, row, col]).detach()) * 448
                                })
                candidates.sort(key=lambda x: -x["score"])  # 将所有候选bounding box按分数从高到低排列
                for c_i in range(len(candidates) - 1):
                    if candidates[c_i]["score"] > 0:
                        for c_j in range(c_i + 1, len(candidates)):
                            if iou(
                                    (candidates[c_i]["x"],
                                     candidates[c_i]["y"],
                                     candidates[c_i]["w"],
                                     candidates[c_i]["h"]),
                                    (candidates[c_j]["x"],
                                     candidates[c_j]["y"],
                                     candidates[c_j]["w"],
                                     candidates[c_j]["h"])
                            ) > self.iou_threshold_pred:
                                candidates[c_j]["score"] = -1
                for c_i in range(len(candidates)):
                    # print(candidates[c_i])
                    if candidates[c_i]["score"] > 0:
                        results[-1].append(candidates[c_i])
        # print(results)
        return results

    def get_mmAP(self, batch, pred_results=None):
        if pred_results is None:
            pred_results = self.predict([data[0] for data in batch])
        for data_id in range(len(batch)):
            dt_list = pred_results[data_id]
            dt_dict = {}
            for category in self.labels:
                dt_dict[category] = []
            for dt in dt_list:
                dt_dict[dt["name"]].append(dt)

            gt_list = batch[data_id][1]
            gt_dict = {}
            for category in self.labels:
                gt_dict[category] = []
            for gt in gt_list:
                gt_dict[gt["name"]].append(gt)

            mAPs = []
            for iou_threshold in self.iou_thresholds_mmAP:
                APs = []
                for category in self.labels:
                    dts = dt_dict[category]
                    gts = gt_dict[category]
                    dts.sort(key=lambda x: -x["score"])  # Sort dts in descending order
                    precisions = []
                    recalls = []
                    for n in range(1, len(dts)):
                        p, r = get_precision_and_recall(dts[:n], gts, iou_threshold)
                        precisions.append(p)
                        recalls.append(r)
                    APs.append(get_AP(precisions, recalls))
                mAPs.append(sum(APs) / len(APs))
            mmAP = sum(mAPs) / len(mAPs)

    def get_output_dict(self, images):
        output_tensor = self.backbone(
            torch.from_numpy(np.array(images) / 255.).to(self.device)
        )  # batch_size, 1470

        num_classes = 20
        S = 7
        B = 2

        probs = output_tensor[:, :S * S * num_classes].view(-1, S, S, num_classes)
        confs = output_tensor[:, S * S * num_classes:S * S * (num_classes + B)].view(-1, S, S, B)
        coords = output_tensor[:, S * S * (num_classes + B):].view(-1, S, S, B, 4)

        output_dict = {}
        for bbox_id in range(B):
            x_label = "x" + str(bbox_id)
            y_label = "y" + str(bbox_id)
            w_label = "w" + str(bbox_id)
            h_label = "h" + str(bbox_id)
            c_label = "c" + str(bbox_id)
            output_dict[x_label] = coords[..., bbox_id, 0]
            output_dict[y_label] = coords[..., bbox_id, 1]
            output_dict[w_label] = coords[..., bbox_id, 2]**2
            output_dict[h_label] = coords[..., bbox_id, 3]**2
            output_dict[c_label] = confs[..., bbox_id]
        output_dict['probs'] = probs

        # for row in range(S):
        #     for col in range(S):
        #         for bbox_id in range(B):
        #             coords[:, row, col, bbox_id, 0] = (coords[:, row, col, bbox_id, 0] + col) / S
        #             coords[:, row, col, bbox_id, 1] = (coords[:, row, col, bbox_id, 1] + row) / S
        #             coords[:, row, col, bbox_id, 2] = coords[:, row, col, bbox_id, 2] ** 2  # 2 or 0.5?
        #             coords[:, row, col, bbox_id, 3] = coords[:, row, col, bbox_id, 3] ** 2

        return output_dict

    def save(self, model_save_path):
        torch.save(self.backbone, model_save_path)

    def save_log(self, log_save_dir):
        with SummaryWriter(log_dir=log_save_dir) as writer:
            writer.add_graph(self.backbone, [torch.rand(1, 448, 448, 3)])
