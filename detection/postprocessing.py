"""Turn scored windows into detections: distance threshold + non-maximum suppression (NMS).

Boxes are (x0, y0, x1, y1). Scores are D3 distances, so lower = better match.
"""
from collections import defaultdict

import numpy as np


def box_iou(a, b):
    """Intersection over union of every box in a (n, 4) with every box in b (m, 4). Returns (n, m)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def threshold(boxes, distances, max_distance):
    """Keep boxes whose D3 distance is at most max_distance. Returns (boxes, distances)."""
    keep = distances <= max_distance
    return boxes[keep], distances[keep]


def nms(boxes, distances, iou_threshold=0.3):
    """Greedy NMS: take the best (lowest-distance) box, drop boxes overlapping it by more than iou_threshold, repeat.

    Works across window sizes, so one object hit at several scales ends up as one box.
    Returns the indices of the kept boxes, best first.

    Kept boxes are filed in a grid of square cells, so each box is only compared with the kept boxes that share a cell
    with it (only those can overlap it). Same result as comparing with every kept box, but fast with ~10^5 boxes.
    """
    order = np.argsort(distances, kind="stable")
    if not len(order):
        return order.astype(np.int64)
    cell = max(8, int((boxes[:, 2] - boxes[:, 0]).min()))
    x0, y0, x1, y1 = (boxes[:, k].tolist() for k in range(4))
    cx0, cy0 = (boxes[:, 0] // cell).tolist(), (boxes[:, 1] // cell).tolist()
    cx1, cy1 = ((boxes[:, 2] - 1) // cell).tolist(), ((boxes[:, 3] - 1) // cell).tolist()
    grid = defaultdict(list)  # cell -> kept boxes touching it
    keep = []
    for i in order.tolist():
        area = (x1[i] - x0[i]) * (y1[i] - y0[i])
        cells = [(cx, cy) for cx in range(cx0[i], cx1[i] + 1) for cy in range(cy0[i], cy1[i] + 1)]
        suppressed, checked = False, set()
        for c in cells:
            for j in grid.get(c, ()):
                if j in checked:
                    continue
                checked.add(j)
                iw = min(x1[i], x1[j]) - max(x0[i], x0[j])
                ih = min(y1[i], y1[j]) - max(y0[i], y0[j])
                if iw > 0 and ih > 0:
                    inter = iw * ih
                    if inter / (area + (x1[j] - x0[j]) * (y1[j] - y0[j]) - inter) > iou_threshold:
                        suppressed = True
                        break
            if suppressed:
                break
        if not suppressed:
            keep.append(i)
            for c in cells:
                grid[c].append(i)
    return np.array(keep, dtype=np.int64)
