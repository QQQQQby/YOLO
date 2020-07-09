# coding: utf-8

import numpy as np
from util.functions import iou


def precision(tp: float, fp: float) -> float:
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 1


def recall(tp: float, fn: float) -> float:
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 1


def get_precision_and_recall(dts, gts, iou_threshold):
    """根据预测结果和真实结果计算precision和recall，假设dts已按分数从大到小排序"""
    dts = dts.copy()
    gts = gts.copy()
    tp = 0
    for dt in dts:
        if len(gts) == 0:
            break
        ious = [iou(dt, gt, 448, 448) for gt in gts]
        if max(ious) >= iou_threshold:
            tp += 1
            gts.pop(int(np.argmax(ious)))
    fp = len(dts) - tp
    fn = len(gts)
    p = precision(tp, fp)
    r = recall(tp, fn)
    return p, r

def get_AP(precisions, recalls):
    pass