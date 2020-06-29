# coding: utf-8

def iou(bbox0, bbox1, max_width, max_height):
    upper_left0 = (round(bbox0[0] - bbox0[2] / 2), round(bbox0[1] - bbox0[3] / 2))
    upper_left1 = (round(bbox1[0] - bbox1[2] / 2), round(bbox1[1] - bbox1[3] / 2))
    inter_upper_left = (max(upper_left0[0], upper_left1[0]), max(upper_left0[1], upper_left1[1]))
    lower_right0 = (round(bbox0[0] + bbox0[2] / 2), round(bbox0[1] + bbox0[3] / 2))
    lower_right1 = (round(bbox1[0] + bbox1[2] / 2), round(bbox1[1] + bbox1[3] / 2))
    inter_lower_right = (min(lower_right0[0], lower_right1[0]), min(lower_right0[1], lower_right1[1]))
    inter_area = 0 if inter_upper_left[0] > inter_lower_right[0] or inter_upper_left[1] > inter_lower_right[1] \
        else (inter_lower_right[0] - inter_upper_left[0] + 1) * (inter_lower_right[1] - inter_upper_left[1] + 1)
    outer_area = bbox0[2] * bbox0[3] + bbox1[2] * bbox1[3]
    return inter_area / outer_area
