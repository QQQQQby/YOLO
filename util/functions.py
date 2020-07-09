# coding: utf-8

import cv2
import numpy as np
from typing import List, Tuple, Dict


def iou(bbox0: Tuple[int, int, int, int], bbox1: Tuple[int, int, int, int], max_width: int, max_height: int) -> float:
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
    return inter_area / outer_area if outer_area != 0 else 0


def show_objects(image_array: np.ndarray, objects, color_dict):
    image = draw_image(image_array, objects, color_dict)
    cv2.imshow('Object Detection', image)
    cv2.waitKey(750)


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

# def AP()
