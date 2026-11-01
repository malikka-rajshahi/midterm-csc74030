"""Match detections to ground-truth boxes and compute precision / recall / F1.

A detection is correct (true positive) if it overlaps a not-yet-matched ground-truth box of the same class in the same
image with IoU >= IOU_MATCH. Detections are matched best (lowest D3 distance) first, so the matches at any distance
threshold are the same as matching only the detections under that threshold.
"""
import numpy as np
import pandas as pd

from detection.postprocessing import box_iou

IOU_MATCH = 0.5
BOX = ["x0", "y0", "x1", "y1"]


def ground_truth_boxes(annotations):
    """Annotations (with cls and bbox_coords = [x, y, w, h]) -> DataFrame of image_path, cls, x0, y0, x1, y1."""
    xywh = np.stack(annotations["bbox_coords"].to_numpy())
    return pd.DataFrame({"image_path": annotations["image_path"].to_numpy(), "cls": annotations["cls"].to_numpy(),
                         "x0": xywh[:, 0], "y0": xywh[:, 1],
                         "x1": xywh[:, 0] + xywh[:, 2], "y1": xywh[:, 1] + xywh[:, 3]})


def match(detections, ground_truth, iou_match=IOU_MATCH):
    """Add a boolean `tp` column to detections (one class, one color space).

    Each ground-truth box can be matched by at most one detection.
    """
    detections = detections.sort_values("distance", kind="stable").copy()
    tp = np.zeros(len(detections), dtype=bool)
    position = pd.Series(np.arange(len(detections)), index=detections.index)
    gt_by_image = dict(tuple(ground_truth.groupby("image_path")))
    for image_path, dets in detections.groupby("image_path", sort=False):
        gt = gt_by_image.get(image_path)
        if gt is None:
            continue
        iou = box_iou(dets[BOX].to_numpy(), gt[BOX].to_numpy())
        used = np.zeros(len(gt), dtype=bool)
        # dets keeps the distance order; only detections touching some box at iou_match can be correct
        for row in np.flatnonzero(iou.max(axis=1) >= iou_match):
            candidates = np.where(used, -1.0, iou[row])
            best = candidates.argmax()
            if candidates[best] >= iou_match:
                used[best] = True
                tp[position[dets.index[row]]] = True
    detections["tp"] = tp
    return detections


def pr_curve(matched, n_ground_truth):
    """Precision, recall and F1 at every distance threshold (= each detection's distance), from match() output."""
    matched = matched.sort_values("distance", kind="stable")
    tp = matched["tp"].to_numpy().cumsum()
    n = np.arange(1, len(matched) + 1)
    precision = tp / n
    recall = tp / max(n_ground_truth, 1)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(len(n)), where=precision + recall > 0)
    return pd.DataFrame({"distance": matched["distance"].to_numpy(), "tp": tp, "fp": n - tp,
                         "precision": precision, "recall": recall, "f1": f1})


def average_precision(curve):
    """Area under the precision-recall curve (all-point interpolation, as in PASCAL VOC)."""
    recall = np.r_[0, curve["recall"].to_numpy(), 1]
    precision = np.r_[0, curve["precision"].to_numpy(), 0]
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    steps = np.flatnonzero(recall[1:] != recall[:-1])
    return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))


def scores_at(curve, max_distance, n_ground_truth):
    """TP, FP, FN, precision, recall and F1 when keeping detections with distance <= max_distance."""
    kept = curve[curve["distance"] <= max_distance]
    tp, fp = (int(kept["tp"].iloc[-1]), int(kept["fp"].iloc[-1])) if len(kept) else (0, 0)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / n_ground_truth if n_ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(tp=tp, fp=fp, fn=n_ground_truth - tp, precision=precision, recall=recall, f1=f1)
