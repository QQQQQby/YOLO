# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
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
        ious = [iou(dt, gt) for gt in gts]
        if max(ious) >= iou_threshold:
            tp += 1
            gts.pop(int(np.argmax(ious)))
    fp = len(dts) - tp
    fn = len(gts)
    p = precision(tp, fp)
    r = recall(tp, fn)
    return p, r


def get_AP(precisions, recalls):
    precisions = precisions.copy()
    recalls = recalls.copy()
    p_r = list(zip(precisions, recalls))
    p_r.sort(key=lambda x: x[1])
    sorted_p, sorted_r = list(zip(*p_r))

    plt.plot(sorted_r, sorted_p)
    plt.show()

    p_candidates = []
    threshold = 0
    while threshold <= 1:
        start = 0
        while start < len(sorted_r):
            if sorted_r[start] > threshold:
                break
            start += 1
        if start >= len(sorted_r):
            p_candidates.append(0)
        else:
            p_candidates.append(max(sorted_p[start:]))
        threshold += 0.1
        threshold = round(threshold * 10) / 10  # Fix accuracy error
    return sum(p_candidates) / len(p_candidates)
