# coding: utf-8

import cv2
import numpy as np
from typing import List, Tuple, Dict

from functools import wraps

def iou(bbox0: Tuple[float, float, float, float], bbox1: Tuple[float, float, float, float]) -> float:
    x0, y0, w0, h0 = bbox0[:4]
    x1, y1, w1, h1 = bbox1[:4]
    upper_left0 = (round(x0 - w0 / 2), round(y0 - h0 / 2))
    upper_left1 = (round(x1 - w1 / 2), round(y1 - h1 / 2))
    lower_right0 = (round(x0 + w0 / 2), round(y0 + h0 / 2))
    lower_right1 = (round(x1 + w1 / 2), round(y1 + h1 / 2))
    inter_upper_left = (max(upper_left0[0], upper_left1[0]), max(upper_left0[1], upper_left1[1]))
    inter_lower_right = (min(lower_right0[0], lower_right1[0]), min(lower_right0[1], lower_right1[1]))

    inter_area = 0 if inter_upper_left[0] > inter_lower_right[0] or inter_upper_left[1] > inter_lower_right[1] \
        else (inter_lower_right[0] - inter_upper_left[0] + 1) * (inter_lower_right[1] - inter_upper_left[1] + 1)
    outer_area = w0 * h0 + w1 * h1 - inter_area
    result = inter_area / outer_area if outer_area != 0 else 0
    return result


def show_objects(image_array: np.ndarray, objects, color_dict, delay=0):
    image = draw_image(image_array, objects, color_dict)
    cv2.imshow('Object Detection', image)
    cv2.waitKey(delay)


def draw_image(image_array: np.ndarray, objects, color_dict):
    image = image_array[:, :, ::-1].copy()
    for o in objects:
        upper_left = (int(o['x'] - o['w'] / 2), int(o['y'] - o['h'] / 2))
        lower_right = (int(o['x'] + o['w'] / 2), int(o['y'] + o['h'] / 2))
        cv2.rectangle(image, upper_left, lower_right, color_dict[o['name']][::-1], 2)
        # cv2.rectangle(image, upper_left, lower_right, (255, 0, 0)[::-1], 1)
        cv2.rectangle(
            image,
            upper_left,
            (upper_left[0] + len(o['name'] * 8), max(upper_left[1] - 12, 16)),
            color_dict[o['name']][::-1],
            -1
        )

        upper_left = (upper_left[0] + 2, max(upper_left[1] - 4, 12))
        cv2.putText(image, o['name'], upper_left, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255)[::-1], 1)
    return image


def NMS(output_dict, image_id, model):
    results = []
    for category in model.labels:
        candidates = []
        for row in range(model.S):
            for col in range(model.S):
                for bbox_id in range(model.B):
                    x_label = "x" + str(bbox_id)
                    y_label = "y" + str(bbox_id)
                    w_label = "w" + str(bbox_id)
                    h_label = "h" + str(bbox_id)
                    c_label = "c" + str(bbox_id)
                    score = float(output_dict[c_label][image_id, row, col] *
                                  output_dict["probs"][image_id, row, col, model.labels.index(category)])
                    if score >= model.score_threshold:
                        candidates.append({
                            "name": category,
                            "score": score,
                            "x": float(output_dict[x_label][image_id, row, col]),
                            "y": float(output_dict[y_label][image_id, row, col]),
                            "w": float(output_dict[w_label][image_id, row, col]),
                            "h": float(output_dict[h_label][image_id, row, col])
                        })
        # print(candidates)
        candidates.sort(key=lambda x: -x["score"])
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
                    ) > model.iou_threshold:
                        candidates[c_j]["score"] = -1
        results += list(filter(lambda x: x["score"] >= 0, candidates))
    return results


def NMS_multi_process(inp):
    # output_dict, image_id, labels, score_threshold, iou_threshold
    return NMS(*inp)
