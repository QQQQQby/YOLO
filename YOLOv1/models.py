# coding: utf-8

from util.functions import *
from util.metrics import determine_TPs, get_AP, get_precision, get_recall
from YOLOv1.modules import YOLOv1Backbone

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool
import threading


class YOLOv1:
    def __init__(self, backbone, device, classes, args):
        self.backbone = backbone
        self.device = device
        self.classes = classes

        self.lr = args["lr"]
        self.momentum = args["momentum"]
        self.lambda_coord = args["lambda_coord"]
        self.lambda_noobj = args["lambda_noobj"]

        # self.optimizer = optim.SGD(self.backbone.parameters(), lr=self.lr, momentum=self.momentum)
        self.optimizer = optim.Adam(self.backbone.parameters(), lr=self.lr)
        # self.optimizer = optim.(self.backbone.parameters(), lr=self.lr, momentum=self.momentum)

        self.clip_max_norm = args["clip_max_norm"]
        self.score_threshold = args["score_threshold"]
        self.iou_threshold = args["iou_threshold"]
        self.iou_thresholds_mmAP = args["iou_thresholds_mmAP"]

        self.num_classes = len(classes)
        self.image_size = 448
        self.S = 7
        if isinstance(backbone, YOLOv1Backbone):
            self.B = 3
        else:
            self.B = 2

    def train(self, batch):
        self.optimizer.zero_grad()
        loss = self.get_loss(batch)
        loss.backward()
        if self.clip_max_norm != 0:
            nn.utils.clip_grad_norm(self.backbone.parameters(), self.clip_max_norm)
        self.optimizer.step()
        return loss

    def get_loss(self, batch):
        output_dict = self.get_output_dict([data[0] for data in batch])
        object_info_dicts = [data[1] for data in batch]

        loss = torch.tensor(0.0)
        for data_id in range(len(batch)):
            has_positive = np.zeros((self.S, self.S), np.bool)
            for true_dict in object_info_dicts[data_id]:
                """获取label对应的栅格所在的行和列"""
                row = int(true_dict['y'] // (self.image_size / self.S))  # [0, 6]
                col = int(true_dict['x'] // (self.image_size / self.S))  # [0, 6]
                ious = []
                pred_coord_dict = {}
                for bbox_id in range(self.B):
                    x_label = "x" + str(bbox_id)
                    y_label = "y" + str(bbox_id)
                    w_label = "w" + str(bbox_id)
                    h_label = "h" + str(bbox_id)
                    """计算预测的坐标"""
                    pred_coord_dict[x_label] = output_dict[x_label][data_id, row, col]  # [0, self.image_size - 1]
                    pred_coord_dict[y_label] = output_dict[y_label][data_id, row, col]  # [0, self.image_size - 1]
                    pred_coord_dict[w_label] = output_dict[w_label][data_id, row, col]
                    pred_coord_dict[h_label] = output_dict[h_label][data_id, row, col]
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
                                             (pred_coord_dict[w_label] - true_dict['w'] ** 0.5) ** 2 +
                                             (pred_coord_dict[h_label] - true_dict['h'] ** 0.5) ** 2)
                # print("Coordinate loss =",
                #       self.lambda_coord * ((pred_coord_dict[x_label] - true_dict['x']) ** 2 +
                #                            (pred_coord_dict[y_label] - true_dict['y']) ** 2 +
                #                            (pred_coord_dict[w_label] ** 0.5 - true_dict['w'] ** 0.5) ** 2 +
                #                            (pred_coord_dict[h_label] ** 0.5 - true_dict['h'] ** 0.5) ** 2))
                loss += (output_dict[c_label][data_id, row, col] - 1) ** 2
                """未被选中的(即IOU较小的)bounding box，取其置信度为0进行损失计算"""
                for bbox_id in range(self.B):
                    if bbox_id == chosen_bbox_id:
                        continue
                    c_label = "c" + str(bbox_id)
                    loss += self.lambda_noobj * (output_dict[c_label][data_id, row, col] - 0) ** 2
                """概率损失"""
                prob_loss = nn.MSELoss(reduction="sum")
                true_porbs = torch.zeros(self.num_classes)
                true_porbs[self.classes.index(true_dict['name'])] = 1
                # print("Prob loss =", prob_loss(output_dict['probs'][data_id, row, col], true_porbs))
                loss += prob_loss(
                    output_dict['probs'][data_id, row, col],
                    true_porbs
                )
                """统计有gt的栅格"""
                has_positive[row, col] = True
            """取未被选中的(即IOU较小的)栅格中的两个bounding box置信度为0"""
            for i in range(self.S):
                for j in range(self.S):
                    if not has_positive[i, j]:
                        for bbox_id in range(self.B):
                            c_label = "c" + str(bbox_id)
                            loss += self.lambda_noobj * (output_dict[c_label][data_id, i, j] - 0) ** 2
        return loss

    def predict(self, images, output_dict=None, num_processes=8):
        if output_dict is None:
            with torch.no_grad():
                output_dict = self.get_output_dict(images)
        for key in output_dict.keys():
            output_dict[key] = output_dict[key].detach().cpu().numpy()
        # a = time.time()
        if num_processes == 0:
            results = []
            for image_id in range(len(images)):
                results.append(NMS(output_dict, image_id, self ,self.classes))
        else:
            p = Pool(num_processes)
            inputs = []
            for image_id in range(len(images)):
                inputs.append((output_dict, image_id, self))
            results = p.map(
                NMS_multi_process,
                inputs
            )
            p.close()
            p.join()
        # b = time.time()
        # print(b - a, "s")
        return results

    def get_mmAP(self, batch, pred_results=None):
        if pred_results is None:
            pred_results = self.predict([data[0] for data in batch])
        gt_dict = {}
        dt_dict = {}
        for class_name in self.classes:
            gt_dict[class_name] = []
            dt_dict[class_name] = []
            for i in range(len(batch)):
                gt_dict[class_name].append([])
                dt_dict[class_name].append([])
        for data_id in range(len(batch)):
            gt_list = batch[data_id][1]
            for gt in gt_list:
                gt_dict[gt["name"]][data_id].append(gt)
            dt_list = pred_results[data_id]
            for dt in dt_list:
                dt_dict[dt["name"]][data_id].append(dt)

        mAPs = []
        for threshold in self.iou_thresholds_mmAP:
            APs = []
            for class_name in self.classes:
                are_tps = []
                for data_id in range(len(batch)):
                    dts = dt_dict[class_name][data_id]
                    gts = gt_dict[class_name][data_id]
                    are_tps.append(determine_TPs(dts, gts, threshold))
                detections = []
                fn = 0
                for data_id in range(len(batch)):
                    fn += len(gt_dict[class_name][data_id])
                    for index, dt in enumerate(dt_dict[class_name][data_id]):
                        detections.append((data_id, dt, are_tps[data_id][index]))
                detections.sort(key=lambda x: -x[1]["score"])
                precisions, recalls = [], []
                tp, fp = 0, 0
                for dt_tuple in detections:
                    if dt_tuple[2]:
                        tp += 1
                        fn -= 1
                    else:
                        fp += 1
                    precisions.append(get_precision(tp, fp))
                    recalls.append(get_recall(tp, fn))
                # plt.plot(recalls, precisions)
                # plt.show()
                APs.append(get_AP(precisions, recalls))
                # print("AP of", class_name, "=", get_AP(precisions, recalls))
            mAPs.append(sum(APs) / len(APs))
        mmAP = sum(mAPs) / len(mAPs)
        return mmAP

    def get_output_dict(self, images):
        output_tensor = self.backbone(
            torch.from_numpy(np.array(images) / 255.).to(self.device)
        )  # batch_size, 1470 or 1715

        probs = output_tensor[:, :self.S * self.S * self.num_classes].view(-1, self.S, self.S, self.num_classes)
        confs = output_tensor[:, self.S * self.S * self.num_classes:self.S * self.S * (self.num_classes + self.B)]
        confs = confs.view(-1, self.S, self.S, self.B)
        coords = output_tensor[:, self.S * self.S * (self.num_classes + self.B):].view(-1, self.S, self.S, self.B, 4)

        output_dict = {}
        for bbox_id in range(self.B):
            x_label = "x" + str(bbox_id)
            y_label = "y" + str(bbox_id)
            w_label = "w" + str(bbox_id)
            h_label = "h" + str(bbox_id)
            c_label = "c" + str(bbox_id)
            # output_dict[x_label] = coords[..., bbox_id, 0]
            # output_dict[y_label] = coords[..., bbox_id, 1]
            output_dict[x_label] = torch.zeros_like(coords[..., bbox_id, 0])
            output_dict[y_label] = torch.zeros_like(coords[..., bbox_id, 1])
            output_dict[w_label] = coords[..., bbox_id, 2] ** 2
            output_dict[w_label] = output_dict[w_label] * self.image_size
            output_dict[h_label] = coords[..., bbox_id, 3] ** 2
            output_dict[h_label] = output_dict[h_label] * self.image_size
            output_dict[c_label] = confs[..., bbox_id]
        output_dict["probs"] = probs

        for row in range(self.S):
            for col in range(self.S):
                for bbox_id in range(self.B):
                    x_label = "x" + str(bbox_id)
                    y_label = "y" + str(bbox_id)
                    output_dict[x_label][:, row, col] = (coords[:, row, col, bbox_id, 0] + col) * \
                                                        ((self.image_size - 1) / self.S)
                    output_dict[y_label][:, row, col] = (coords[:, row, col, bbox_id, 1] + row) * \
                                                        ((self.image_size - 1) / self.S)
        return output_dict

    def save(self, model_save_path):
        torch.save(self.backbone, model_save_path)

    def save_graph(self, graph_save_dir):
        with SummaryWriter(log_dir=graph_save_dir) as writer:
            writer.add_graph(self.backbone, [torch.rand(1, self.image_size, self.image_size, 3)])

    def detect_image_and_show(self, image_path, color_dict, delay):
        im, unresized_im, paddings = self.preprocess_image(image_path, cvt_RGB=True)
        pred_results = self.predict([im], num_processes=0)[0]
        pred_results = self.unpreprocess_objects(pred_results, unresized_im.shape, paddings)
        print(pred_results)
        show_objects(unresized_im, pred_results, color_dict, delay)

    def detect_video_and_show(self, video_path, color_dict):
        if video_path == "0":
            capture = cv2.VideoCapture(0)
        else:
            capture = cv2.VideoCapture(video_path)
        while capture.isOpened():
            ret, frame = capture.read()
            frame, unresized_frame, paddings = self.preprocess_image(frame, cvt_RGB=True)
            pred_results = self.predict([frame], num_processes=0)[0]
            pred_results = self.unpreprocess_objects(pred_results, unresized_frame.shape, paddings)
            unresized_frame = draw_image(unresized_frame, pred_results, color_dict)
            cv2.namedWindow("Object Detection", 0)
            cv2.resizeWindow("Object Detection", unresized_frame.shape[1], unresized_frame.shape[0])
            cv2.imshow("Object Detection", unresized_frame)
            cv2.waitKey(1)

    def preprocess_image(self, image, objects=None, cvt_RGB=True):
        if isinstance(image, str):
            image = cv2.imread(image)
        if cvt_RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        org_height, org_width = image.shape[:2]
        """resize"""
        if org_width > org_height:
            pre_width, pre_height = int(self.image_size), int(org_height / org_width * self.image_size)
        else:
            pre_width, pre_height = int(org_width / org_height * self.image_size), int(self.image_size)
        # print(pre_width, pre_height)
        padding_top = int((self.image_size - pre_height) / 2)
        padding_bottom = self.image_size - padding_top - pre_height
        padding_left = int((self.image_size - pre_width) / 2)
        padding_right = self.image_size - padding_left - pre_width
        unresized_image = image
        image = image.copy()
        image = cv2.resize(image, (pre_width, pre_height))
        image = cv2.copyMakeBorder(image, padding_top, padding_bottom, padding_left, padding_right,
                                   cv2.BORDER_CONSTANT, (0, 0, 0))

        if objects is not None:
            for obj in objects:
                if obj.get("fixed", False):
                    continue
                obj["x"] = obj["x"] / (org_width - 1) * (pre_width - 1)
                obj["y"] = obj["y"] / (org_height - 1) * (pre_height - 1)
                obj["x"] += padding_left
                obj["y"] += padding_top
                obj["w"] = obj["w"] / org_width * pre_width
                obj["h"] = obj["h"] / org_height * pre_height
                obj["fixed"] = True

        return image, unresized_image, (padding_top, padding_bottom, padding_left, padding_right)

    def unpreprocess_objects(self, objects, org_shape, paddings):
        prop = max(org_shape[:2]) / self.image_size
        for obj in objects:
            obj["x"] -= paddings[2]
            obj["y"] -= paddings[0]
            obj["x"] *= prop
            obj["y"] *= prop
            obj["w"] *= prop
            obj["h"] *= prop
            obj["fixed"] = False
        return objects
