# coding: utf-8

from util.functions import *
from util.metrics import determine_TPs, get_AP, get_precision, get_recall
from YOLOv1.modules import YOLOv1Backbone, TinyYOLOv1Backbone
from YOLOv3.modules import YOLOv3Backbone

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool


class YOLO:
    def __init__(self, classes, model_name=None, model_load_path=None, device_ids='-1'):
        self.optimizer = None

        self.classes = classes
        self.num_classes = len(classes)

        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
        if device_ids == "-1":
            self.device = torch.device('cpu')
        elif torch.cuda.is_available():
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            self.device = torch.device('cuda')
        else:
            raise EnvironmentError("Cannot find devices.")

        if model_load_path is None:
            if model_name == "yolov1":
                self.backbone = YOLOv1Backbone()
            elif model_name == "yolov1-tiny":
                self.backbone = TinyYOLOv1Backbone()
            elif model_name == "yolov3":
                self.backbone = YOLOv3Backbone()
            else:
                raise ValueError("Wrong arguments of model name or path!")
                # initialize with zero
            for layer_key in self.backbone.state_dict():
                torch.nn.init.zeros_(self.backbone.state_dict()[layer_key])
        else:
            self.backbone = torch.load(model_load_path)
        self.backbone = self.backbone.to(self.device)

        if isinstance(self.backbone, YOLOv1Backbone) or isinstance(self.backbone, TinyYOLOv1Backbone):
            self.image_size = 448
        elif isinstance(self.backbone, YOLOv3Backbone):
            self.image_size = 416
        # self.S = 7
        # if isinstance(self.backbone, YOLOv1Backbone):
        #     self.B = 3
        # else:
        #     self.B = 2

    def get_optimizer(self, name, lr, momentum):
        if self.optimizer is None:
            if name == "sgd":
                self.optimizer = optim.SGD(self.backbone.parameters(), lr=lr, momentum=momentum)
            elif name == "adam":
                self.optimizer = optim.Adam(self.backbone.parameters(), lr=lr)
            else:
                pass
        return self.optimizer

    def train(self, batch, optimizer_name="sgd", lr=0.0005, momentum=0.9, clip_max_norm=0,
              lambda_coord=1, lambda_noobj=5):
        self.backbone.train()
        self.get_optimizer(optimizer_name, lr, momentum).zero_grad()
        loss = self.get_loss(batch, lambda_coord, lambda_noobj)
        loss.backward()
        if clip_max_norm != 0:
            nn.utils.clip_grad_norm(self.backbone.parameters(), clip_max_norm)
        self.optimizer.step()
        return loss

    def get_loss(self, batch, lambda_coord, lambda_noobj):
        output_dict = self.get_output_dicts([data[0] for data in batch])
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
                loss += lambda_coord * ((pred_coord_dict[x_label] - true_dict['x']) ** 2 +
                                        (pred_coord_dict[y_label] - true_dict['y']) ** 2 +
                                        (pred_coord_dict[w_label] - true_dict['w'] ** 0.5) ** 2 +
                                        (pred_coord_dict[h_label] - true_dict['h'] ** 0.5) ** 2)
                # print("Coordinate loss =",
                #       lambda_coord * ((pred_coord_dict[x_label] - true_dict['x']) ** 2 +
                #                            (pred_coord_dict[y_label] - true_dict['y']) ** 2 +
                #                            (pred_coord_dict[w_label] ** 0.5 - true_dict['w'] ** 0.5) ** 2 +
                #                            (pred_coord_dict[h_label] ** 0.5 - true_dict['h'] ** 0.5) ** 2))
                loss += (output_dict[c_label][data_id, row, col] - 1) ** 2
                """未被选中的(即IOU较小的)bounding box，取其置信度为0进行损失计算"""
                for bbox_id in range(self.B):
                    if bbox_id == chosen_bbox_id:
                        continue
                    c_label = "c" + str(bbox_id)
                    loss += lambda_noobj * (output_dict[c_label][data_id, row, col] - 0) ** 2
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
                            loss += lambda_noobj * (output_dict[c_label][data_id, i, j] - 0) ** 2
        return loss

    def predict(self, images, score_threshold, iou_threshold, output_dicts=None, num_processes=8):
        self.backbone.eval()
        if output_dicts is None:
            with torch.no_grad():
                output_dicts = self.get_output_dicts(images)

        a = time.time()
        for output_dict in output_dicts:
            num_classes = output_dict["num_classes"]
            B = output_dict["B"]
            S = output_dict["S"]

            """Make x, y, w and h offsets after normalization"""
            if isinstance(self.backbone, YOLOv1Backbone) or isinstance(self.backbone, TinyYOLOv1Backbone):
                for bbox_id in range(B):
                    w_label = "w" + str(bbox_id)
                    h_label = "h" + str(bbox_id)
                    output_dict[w_label] = output_dict[w_label] ** 2
                    output_dict[h_label] = output_dict[h_label] ** 2
            elif isinstance(self.backbone, YOLOv3Backbone):
                for bbox_id in range(B):
                    x_label = "x" + str(bbox_id)
                    y_label = "y" + str(bbox_id)
                    w_label = "w" + str(bbox_id)
                    h_label = "h" + str(bbox_id)
                    c_label = "c" + str(bbox_id)
                    """"""

            for bbox_id in range(B):
                x_label = "x" + str(bbox_id)
                y_label = "y" + str(bbox_id)
                w_label = "w" + str(bbox_id)
                h_label = "h" + str(bbox_id)
                output_dict[w_label] = output_dict[w_label] * self.image_size
                output_dict[h_label] = output_dict[h_label] * self.image_size
                # Could be faster
                for row in range(S):
                    output_dict[y_label][:, row, :] = output_dict[y_label][:, row, :] + row
                for col in range(S):
                    output_dict[x_label][:, :, col] = output_dict[x_label][:, :, col] + col
                output_dict[x_label] = output_dict[x_label] * (self.image_size - 1) / S
                output_dict[y_label] = output_dict[y_label] * (self.image_size - 1) / S
        b = time.time()
        print("normalize:", b - a, "s")

        a = time.time()
        candidates = [[[] for j in self.classes] for i in images]
        for output_dict in output_dicts:
            num_classes = output_dict["num_classes"]
            B = output_dict["B"]
            S = output_dict["S"]

            scores = torch.zeros(B, len(images), S, S, num_classes)
            for bbox_id in range(B):
                c_label = "c" + str(bbox_id)
                p_label = "p" + str(bbox_id)
                scores[bbox_id] = (output_dict[c_label].unsqueeze(-1) * output_dict[p_label])
            are_candidates = scores >= score_threshold
            scores = scores.cpu().detach().numpy()
            are_candidates = are_candidates.cpu().detach().numpy()
            for bbox_id in range(B):
                x_label = "x" + str(bbox_id)
                y_label = "y" + str(bbox_id)
                w_label = "w" + str(bbox_id)
                h_label = "h" + str(bbox_id)
                for class_id, class_name in enumerate(self.classes):
                    for image_id in range(len(images)):
                        for row in range(S):
                            for col in range(S):
                                if are_candidates[bbox_id, image_id, row, col, class_id]:
                                    candidates[image_id][class_id].append({
                                        "name": class_name,
                                        "score": scores[bbox_id, image_id, row, col, class_id],
                                        "x": float(output_dict[x_label][image_id, row, col]),
                                        "y": float(output_dict[y_label][image_id, row, col]),
                                        "w": float(output_dict[w_label][image_id, row, col]),
                                        "h": float(output_dict[h_label][image_id, row, col])
                                    })
        b = time.time()
        print("get candidates:", b - a, "s")

        a = time.time()
        results = []
        if num_processes == 0:
            for image_id in range(len(images)):
                current_result = []
                for class_id in range(len(self.classes)):
                    current_result += NMS(candidates[image_id][class_id], iou_threshold)
                results.append(current_result)
        else:
            p = Pool(num_processes)
            inputs = []
            for image_id in range(len(images)):
                for class_id in range(len(self.classes)):
                    inputs.append((candidates[image_id][class_id], iou_threshold))
            results_temp = p.map(
                NMS_multi_process,
                inputs
            )
            p.close()
            p.join()
            for image_id in range(len(images)):
                results.append([])
                for class_id in range(len(self.classes)):
                    results[-1] += results_temp[image_id * len(self.classes) + class_id]

        # for image_id in range(len(images)):
        #     current_result = []
        #     for class_id in range(len(self.classes)):
        #         current_result += candidates[image_id][class_id]
        #     results.append(current_result)

        b = time.time()
        print("NMS:", b - a, "s")
        return results

    def get_mmAP(self, batch, pred_results=None, iou_thresholds=None):
        if pred_results is None:
            pred_results = self.predict([data[0] for data in batch])
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
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
        for threshold in iou_thresholds:
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

    def get_output_dicts(self, images):
        a = time.time()
        prob_list, conf_list, coord_list = self.backbone(
            torch.from_numpy(np.array(images) / 255.).to(self.device)
        )
        output_dicts = []
        for probs, confs, coords in zip(prob_list, conf_list, coord_list):
            S = probs.shape[2]
            num_classes = probs.shape[3]
            B = confs.shape[3]
            current_dict = {}
            for bbox_id in range(B):
                x_label = "x" + str(bbox_id)
                y_label = "y" + str(bbox_id)
                w_label = "w" + str(bbox_id)
                h_label = "h" + str(bbox_id)
                c_label = "c" + str(bbox_id)
                current_dict[x_label] = coords[..., bbox_id, 0]
                current_dict[y_label] = coords[..., bbox_id, 1]
                current_dict[w_label] = coords[..., bbox_id, 2]
                current_dict[h_label] = coords[..., bbox_id, 3]
                current_dict[c_label] = confs[..., bbox_id]
            if isinstance(self.backbone, YOLOv1Backbone) or isinstance(self.backbone, TinyYOLOv1Backbone):
                for bbox_id in range(B):
                    p_label = "p" + str(bbox_id)
                    current_dict[p_label] = probs
            elif isinstance(self.backbone, YOLOv3Backbone):
                for bbox_id in range(B):
                    p_label = "p" + str(bbox_id)
                    current_dict[p_label] = probs[..., bbox_id, :]
            current_dict.update(S=S, B=B, num_classes=num_classes)
            output_dicts.append(current_dict)
        b = time.time()
        print("get output:", b - a, "s")
        return output_dicts

    def save(self, model_save_path):
        torch.save(self.backbone, model_save_path)

    def save_graph(self, graph_save_dir):
        with SummaryWriter(log_dir=graph_save_dir) as writer:
            writer.add_graph(self.backbone, [torch.rand(1, self.image_size, self.image_size, 3)])

    def detect_image_and_show(self, image_path, score_threshold, iou_threshold, color_dict, delay):
        a = time.time()
        im, unresized_im, paddings = self.preprocess_image(image_path, cvt_RGB=True)
        pred_results = self.predict([im], score_threshold, iou_threshold, num_processes=0)[0]
        pred_results = self.unpreprocess_objects(pred_results, unresized_im.shape, paddings)
        print(pred_results)
        b = time.time()
        print("total time:", b - a, "s")
        print()
        show_objects(unresized_im, pred_results, color_dict, delay)

    def detect_video_and_show(self, video_path, score_threshold, iou_threshold, color_dict, delay):
        if video_path == "0":
            capture = cv2.VideoCapture(0)
        else:
            capture = cv2.VideoCapture(video_path)
        while capture.isOpened():
            a = time.time()
            ret, frame = capture.read()
            frame, unresized_frame, paddings = self.preprocess_image(frame, cvt_RGB=True)
            pred_results = self.predict([frame], score_threshold, iou_threshold, num_processes=0)[0]
            pred_results = self.unpreprocess_objects(pred_results, unresized_frame.shape, paddings)
            unresized_frame = draw_image(unresized_frame, pred_results, color_dict)
            cv2.namedWindow("Object Detection", 0)
            cv2.resizeWindow("Object Detection", unresized_frame.shape[1], unresized_frame.shape[0])
            cv2.imshow("Object Detection", unresized_frame)
            b = time.time()
            print("total time:", b - a, "s")
            print()
            cv2.waitKey(delay)

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
