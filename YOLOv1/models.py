# coding: utf-8

from util.functions import iou, show_objects, NMS, NMS_multi_process, draw_image
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
    def __init__(self, backbone, device, labels, args):
        self.backbone = backbone
        self.device = device
        self.labels = labels

        self.lr = args["lr"]
        self.momentum = args["momentum"]
        self.lambda_coord = args["lambda_coord"]
        self.lambda_noobj = args["lambda_noobj"]

        self.optimizer = optim.SGD(self.backbone.parameters(), lr=self.lr, momentum=self.momentum)
        # self.optimizer = optim.Adam(self.backbone.parameters(), lr=self.lr)
        # self.optimizer = optim.(self.backbone.parameters(), lr=self.lr, momentum=self.momentum)

        self.clip_max_norm = args["clip_max_norm"]
        self.score_threshold = args["score_threshold"]
        self.iou_threshold = args["iou_threshold"]
        self.iou_thresholds_mmAP = args["iou_thresholds_mmAP"]

        self.num_classes = len(labels)
        self.image_size = 448
        self.S = 7
        if isinstance(backbone, YOLOv1Backbone):
            self.B = 3
        else:
            self.B = 2

    def train(self, batch, do_preprocess=False):
        if do_preprocess:
            for data_id in range(len(batch)):
                batch[data_id][0] = self.preprocess(batch[data_id][0])
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
                                             (pred_coord_dict[w_label] ** 0.5 - true_dict['w'] ** 0.5) ** 2 +
                                             (pred_coord_dict[h_label] ** 0.5 - true_dict['h'] ** 0.5) ** 2)
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
                true_porbs = torch.zeros((self.num_classes))
                true_porbs[self.labels.index(true_dict['name'])] = 1
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
                results.append(NMS(output_dict, image_id, self))
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
            # results = [[] for i in range(len(images))]
            #
            # num_images_per_thread = (len(images)-1)//num_threads+1
            # num_threads = (len(images)-1)//num_images_per_thread+1
            # image_id_slices = []
            # for thread_id in range(num_threads):
            #     image_id_slices.append(list(range(
            #         thread_id*num_images_per_thread,
            #         min((thread_id+1)*num_images_per_thread, len(images))
            #     )))
            # print(image_id_slices)
            #
            # thread_pool = []
            # for thread_id in range(num_threads):
            #     thread_pool.append(threading.Thread(
            #         target=NMS_multi_images,
            #         args=(results, output_dict, image_id_slices[thread_id], self.labels, self.score_threshold, self.iou_threshold)
            #     ))
            # for thread_id in range(num_threads):
            #     thread_pool[thread_id].start()
            # for thread_id in range(num_threads):
            #     thread_pool[thread_id].join()
        # b = time.time()
        # print(b - a, "s")
        return results

    def get_mmAP(self, batch, pred_results=None):
        if pred_results is None:
            pred_results = self.predict([data[0] for data in batch])
        gt_dict = {}
        dt_dict = {}
        for category in self.labels:
            gt_dict[category] = [[] for i in range(len(batch))]
            dt_dict[category] = [[] for i in range(len(batch))]
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
            for category in self.labels:
                are_tps = []
                for data_id in range(len(batch)):
                    dts = dt_dict[category][data_id]
                    gts = gt_dict[category][data_id]
                    are_tps.append(determine_TPs(dts, gts, threshold))
                detections = []
                fn = 0
                for data_id in range(len(batch)):
                    fn += len(gt_dict[category][data_id])
                    for index, dt in enumerate(dt_dict[category][data_id]):
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
                # print("AP of", category, "=", get_AP(precisions, recalls))
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
            output_dict[x_label] = coords[..., bbox_id, 0]
            output_dict[y_label] = coords[..., bbox_id, 1]
            output_dict[w_label] = coords[..., bbox_id, 2] ** 2 * self.image_size
            output_dict[h_label] = coords[..., bbox_id, 3] ** 2 * self.image_size
            output_dict[c_label] = confs[..., bbox_id]
        output_dict['probs'] = probs

        for row in range(self.S):
            for col in range(self.S):
                for bbox_id in range(self.B):
                    coords[:, row, col, bbox_id, 0] = (coords[:, row, col, bbox_id, 0] + col) \
                                                      * ((self.image_size - 1) / self.S)
                    coords[:, row, col, bbox_id, 1] = (coords[:, row, col, bbox_id, 1] + row) \
                                                      * ((self.image_size - 1) / self.S)

        return output_dict

    def save(self, model_save_path):
        torch.save(self.backbone, model_save_path)

    def save_graph(self, graph_save_dir):
        with SummaryWriter(log_dir=graph_save_dir) as writer:
            writer.add_graph(self.backbone, [torch.rand(1, self.image_size, self.image_size, 3)])

    def detect_image_and_show(self, image_path, color_dict, delay):
        im = self.preprocess(image_path, cvt_RGB=True)
        pred_results = self.predict([im], num_processes=0)
        print(pred_results)
        show_objects(im, pred_results[0], color_dict, delay)

    def detect_video_and_show(self, video_path, color_dict):
        if video_path == "0":
            capture = cv2.VideoCapture(0)
        else:
            capture = cv2.VideoCapture(video_path)
        while capture.isOpened():
            ret, frame = capture.read()
            # frame = cv2.rotate(frame, 0)
            frame = self.preprocess(frame, cvt_RGB=True)
            frame = draw_image(frame, self.predict([frame], num_processes=0)[0], color_dict)
            cv2.namedWindow("Object Detection", 0)
            cv2.resizeWindow("Object Detection", self.image_size, self.image_size)
            cv2.imshow("Object Detection", frame)
            """设置每帧时间"""
            cv2.waitKey(1)

    def preprocess(self, image, cvt_RGB=False):
        if isinstance(image, str):
            image = cv2.imread(image)
        if cvt_RGB:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return cv2.resize(image, (self.image_size, self.image_size)).copy()
