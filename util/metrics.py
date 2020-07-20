# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from util.functions import iou


def get_precision(tp: float, fp: float) -> float:
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return 1


def get_recall(tp: float, fn: float) -> float:
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 1


def determine_TPs(dts, gts, iou_threshold):
    """For each dt in dts, determine whether it is TP"""
    dts = dts.copy()
    gts = gts.copy()
    results = []
    for dt in dts:
        if len(gts) == 0:
            for i in range(len(results)-1, len(dts)-1):
                results.append(False)
            break
        matched_gt_index = 0
        max_iou = iou(
            (dt["x"],
             dt["y"],
             dt["w"],
             dt["h"]),
            (gts[0]["x"],
             gts[0]["y"],
             gts[0]["w"],
             gts[0]["h"])
        )
        for gt_index in range(1, len(gts)):
            current_iou = iou(
                (dt["x"],
                 dt["y"],
                 dt["w"],
                 dt["h"]),
                (gts[gt_index]["x"],
                 gts[gt_index]["y"],
                 gts[gt_index]["w"],
                 gts[gt_index]["h"])
            )
            if max_iou < current_iou:
                matched_gt_index = gt_index
                max_iou = current_iou
        if max_iou >= iou_threshold:
            results.append(True)
            gts.pop(matched_gt_index)
        else:
            results.append(False)
    return results


def get_AP(precisions, recalls):
    if len(precisions) == 0 or len(recalls) == 0:
        return 0
    precisions = precisions.copy()
    recalls = recalls.copy()
    p_r = list(zip(precisions, recalls))
    p_r.sort(key=lambda x: x[1])
    sorted_p, sorted_r = list(zip(*p_r))
    # plt.plot(sorted_r, sorted_p)
    # plt.show()
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
