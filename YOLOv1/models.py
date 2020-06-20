from YOLOv1.modules import YOLOv1Backbone
from util.functions import iou

import torch
import numpy as np


class YOLOv1:
    def __init__(self, labels, lambda_coord, lambda_noobj):
        self.backbone = YOLOv1Backbone()
        self.labels = labels
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def train(self, batch):
        self.backbone.train()
        out = self.backbone(torch.from_numpy(np.array([d[0] for d in batch])))  # batch_size, 7, 7, 30

        d = {}
        d["probs"] = out[:, :, :, :20]
        names = ['c0', 'c1',
                 'x0', 'y0', 'w0', 'h0',
                 'x1', 'y1', 'w1', 'h1']
        for i, name in enumerate(names):
            d[name] = torch.sigmoid(out[:, :, :, 20 + i])

        label_batch = [d[1] for d in batch]

        loss = torch.tensor(0.0)

        for data_id in range(len(batch)):
            for label in label_batch[data_id]:
                """获取label对应的栅格所在的行和列"""
                row = label['x'] // 64
                col = label['y'] // 64
                """计算IOU"""
                iou0 = iou((d['x0'][data_id, row, col], d['y0'][data_id, row, col],
                            d['w0'][data_id, row, col], d['h0'][data_id, row, col]),
                           (label['x'], label['y'],
                            label['w'], label['h']),
                           448, 448)
                iou1 = iou((d['x1'][data_id, row, col], d['y1'][data_id, row, col],
                            d['w1'][data_id, row, col], d['h1'][data_id, row, col]),
                           (label['x'], label['y'],
                            label['w'], label['h']),
                           448, 448)
                # if iou0 > iou1:


                """坐标预测"""
                loss += self.lambda_coord * ((d['x0'][data_id, row, col] * (448 - 1) - label['x']) ** 2 +
                                             (d['y0'][data_id, row, col] * (448 - 1) - label['y']) ** 2 +
                                             (d['x1'][data_id, row, col] * (448 - 1) - label['x']) ** 2 +
                                             (d['y1'][data_id, row, col] * (448 - 1) - label['y']) ** 2)
                """"""

        print(loss)
