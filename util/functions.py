# coding: utf-8

import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import List, Tuple, Dict, Union


def get_same_paddings(input_size, kernel_size, stride):
    output_size = (np.ceil(input_size[0] / stride[0]), np.ceil(input_size[1] / stride[1]))
    pad_h = max(int((output_size[0] - 1) * stride[0] + kernel_size[0] - input_size[0]), 0)
    pad_w = max(int((output_size[1] - 1) * stride[1] + kernel_size[1] - input_size[1]), 0)
    pad_bottom = pad_h // 2
    pad_top = pad_h - pad_bottom
    pad_right = pad_w // 2
    pad_left = pad_w - pad_right
    return pad_left, pad_right, pad_top, pad_bottom


def iou(bbox0: Tuple, bbox1: Tuple) -> float:
    x0, y0, w0, h0 = bbox0[:4]
    x1, y1, w1, h1 = bbox1[:4]
    upper_left0 = (round(x0 - w0 // 2), round(y0 - h0 // 2))
    upper_left1 = (round(x1 - w1 // 2), round(y1 - h1 // 2))
    lower_right0 = (round(x0 + w0 // 2), round(y0 + h0 // 2))
    lower_right1 = (round(x1 + w1 // 2), round(y1 + h1 // 2))
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


# def draw_image(image: np.ndarray, objects, color_dict):
#     image = image.copy()
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     h, w = image.shape[:2]
#     upper_left_list = [(max(int(o['x'] - o['w'] // 2), 0), max(int(o['y'] - o['h'] // 2), 0)) for o in objects]
#     lower_right_list = [(min(int(o['x'] + o['w'] // 2), w), min(int(o['y'] + o['h'] // 2), h)) for o in objects]
#     for i, o in enumerate(objects):
#         cv2.rectangle(image, upper_left_list[i], lower_right_list[i], color_dict[o['name']][::-1], 2)
#         # cv2.rectangle(image, upper_left, lower_right, (255, 0, 0)[::-1], 1)
#     for i, o in enumerate(objects):
#         cv2.rectangle(
#             image,
#             upper_left_list[i],
#             (upper_left_list[i][0] + len(o['name'] * 10), max(upper_left_list[i][1] - 16, 0)),
#             color_dict[o['name']][::-1],
#             -1
#         )
#         upper_left = (upper_left_list[i][0] + 2, max(upper_left_list[i][1] - 4, 12))
#         cv2.putText(image, o['name'], upper_left, cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 255, 255)[::-1], 1)
#     return image


def draw_image(image: np.ndarray, objects, color_dict, fontSize=20):
    image = image.copy()
    h, w = image.shape[:2]
    upper_left_list = [(max(int(o['x'] - o['w'] // 2), 0), max(int(o['y'] - o['h'] // 2), 0)) for o in objects]
    lower_right_list = [(min(int(o['x'] + o['w'] // 2), w), min(int(o['y'] + o['h'] // 2), h)) for o in objects]
    img = Image.fromarray(image)
    draw = ImageDraw.Draw(img)
    for i, o in enumerate(objects):
        draw.rectangle([upper_left_list[i], lower_right_list[i]], outline=color_dict[o['name']], width=3)
    font = ImageFont.truetype("simsun.ttf", fontSize, encoding="utf-8")
    for i, o in enumerate(objects):
        upper_left = (max(upper_left_list[i][0], 0), max(upper_left_list[i][1] - fontSize, 0))
        if upper_left[0] >= w or upper_left[1] >= h:
            continue
        lower_right = (upper_left[0] + len(o['name']) * fontSize, upper_left[1] + fontSize)
        draw.rectangle([upper_left, lower_right], fill=color_dict[o['name']])
        draw.text(
            upper_left,
            o['name'],
            font=font,
            fill=(0, 0, 0))
    image = np.array(img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def NMS(candidates: List, iou_threshold: int) -> List:
    # results = sorted(candidates,key=lambda x: -x["score"])
    # print(results)
    if len(candidates) == 0:
        return []
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
                ) > iou_threshold:
                    candidates[c_j]["score"] = -1
    return list(filter(lambda x: x["score"] >= 0, candidates))


def NMS_multi_process(inp):
    return NMS(*inp)


# def sigmoid(array: np.ndarray) -> np.ndarray:
#     array = array.copy()
#     posIndexes = np.where(array > 0)
#     negIndexes = np.where(array <= 0)
#     array[posIndexes] = np.divide(1, (np.add(np.exp(-array[posIndexes]), 1)))
#     array[negIndexes] = np.divide(np.exp(array[negIndexes]), (np.add(np.exp(array[negIndexes]), 1)))
#     return array
