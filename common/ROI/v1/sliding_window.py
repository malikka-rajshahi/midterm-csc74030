"""Fixed-window proposals and geometric near-duplicate removal; no class decisions."""
import numpy as np

WINDOW_PRESETS = {
    'small': (16, 32, 64),
    'balanced': (16, 32, 64, 128, 256),
    'context': (32, 64, 128, 256, 512, 1024),
}
HEIGHT_WIDTH_RATIOS = (0.5, 1.0, 2.0, 4.0)


def sliding_windows(image_shape, widths=WINDOW_PRESETS['balanced'],
                    height_width_ratios=HEIGHT_WIDTH_RATIOS, stride_fraction=0.5):
    """Half-open image-coordinate boxes, including right/bottom aligned windows."""
    height, width = image_shape[:2]
    if min(height, width) < 1 or not 0 < stride_fraction <= 1:
        raise ValueError('Invalid image shape or stride fraction.')
    if not widths or any(w < 1 or int(w) != w for w in widths):
        raise ValueError('Window widths must be positive integers.')
    if not height_width_ratios or any(r <= 0 for r in height_width_ratios):
        raise ValueError('Height/width ratios must be positive.')
    blocks = []
    geometries = sorted({(int(w), max(1, round(w * r)))
                         for w in widths for r in height_width_ratios})
    for bw, bh in geometries:
        if bw > width or bh > height:
            continue
        xs = np.unique(np.r_[np.arange(0, width - bw + 1, max(1, round(bw * stride_fraction))), width - bw])
        ys = np.unique(np.r_[np.arange(0, height - bh + 1, max(1, round(bh * stride_fraction))), height - bh])
        x, y = np.meshgrid(xs, ys)
        blocks.append(np.column_stack([x.ravel(), y.ravel(), x.ravel() + bw, y.ravel() + bh]))
    return np.concatenate(blocks).astype(np.int32) if blocks else np.empty((0, 4), dtype=np.int32)


def filter_color_windows(boxes, color_masks, min_fraction=0.20, min_pixels=2,
                         proposal_colors=('red', 'orange', 'yellow'), max_green_fraction=0.50):
    """One common foreground-color gate, including full ROI background in its denominator."""
    if not 0 <= min_fraction <= 1 or min_pixels < 1:
        raise ValueError('Invalid color gate.')
    if not len(boxes):
        return boxes.copy()
    if not proposal_colors or set(proposal_colors) - {'red', 'orange', 'yellow', 'blue'}:
        raise ValueError('Invalid proposal colors; green/white cannot seed windows.')
    masks = [color_masks[name] for name in proposal_colors]
    union = np.logical_or.reduce(masks)
    integral = np.pad(union.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    x0, y0, x1, y1 = boxes.T
    counts = integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]
    keep = (counts >= min_pixels) & (counts / ((x1 - x0) * (y1 - y0)) >= min_fraction)
    if max_green_fraction is not None:
        if not 0 <= max_green_fraction <= 1:
            raise ValueError('Invalid green fraction.')
        green = np.pad(color_masks['green'].astype(np.int64), ((1,0),(1,0))).cumsum(0).cumsum(1)
        green_count = green[y1,x1]-green[y0,x1]-green[y1,x0]+green[y0,x0]
        keep &= green_count / ((x1-x0)*(y1-y0)) <= max_green_fraction
    return boxes[keep]


def deduplicate_boxes(boxes, iou_threshold=0.90):
    """Stable geometric deduplication, accelerated by shape/center spatial buckets.

    Returns kept raw indices and a raw-index -> kept-raw-index parent mapping.
    No score, ground truth or class is used. Nested boxes with low IoU remain.
    """
    boxes = np.asarray(boxes).reshape(-1, 4)
    parents = np.arange(len(boxes), dtype=int)
    if iou_threshold is None or not len(boxes):
        return parents.copy(), parents
    if not 0.5 <= iou_threshold <= 1:
        raise ValueError('Deduplication IoU must be in [0.5,1], or None.')
    dimensions = boxes[:, 2:] - boxes[:, :2]
    if (dimensions <= 0).any() or not np.isfinite(boxes).all():
        raise ValueError('Boxes must be finite and have positive dimensions.')
    levels = np.floor(np.log2(dimensions)).astype(int)
    centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    areas = dimensions.prod(axis=1)
    # Width/height ratios must each be >= threshold for a high-IoU pair.
    bounds = {}
    for key in set(map(tuple, levels)):
        subset = dimensions[(levels == key).all(axis=1)]
        bounds[key] = (subset.min(axis=0), subset.max(axis=0))
    buckets, compatible_cache, kept = {}, {}, []
    displacement = (1 - iou_threshold) / (1 + iou_threshold)
    for index, (box, size, center) in enumerate(zip(boxes, dimensions, centers)):
        shape = tuple(size)
        if shape not in compatible_cache:
            compatible_cache[shape] = [key for key, (low, high) in bounds.items()
                if (low <= size / iou_threshold).all() and (high >= size * iou_threshold).all()]
        candidates = []
        for key in compatible_cache[shape]:
            cell = np.exp2(np.array(key, dtype=float)) / 4.0
            radius = displacement * (size + bounds[key][1]) / 2.0
            low = np.floor((center - radius) / cell).astype(int)
            high = np.floor((center + radius) / cell).astype(int)
            for x in range(low[0], high[0] + 1):
                for y in range(low[1], high[1] + 1):
                    candidates.extend(buckets.get((key[0], key[1], x, y), ()))
        if candidates:
            ids = np.asarray(candidates, dtype=int)
            intersection = np.maximum(0, np.minimum(box[2:], boxes[ids, 2:]) -
                                      np.maximum(box[:2], boxes[ids, :2])).prod(axis=1)
            iou = intersection / (areas[index] + areas[ids] - intersection)
            duplicates = ids[iou >= iou_threshold]
            if len(duplicates):
                parents[index] = int(duplicates.min())
                continue
        kept.append(index)
        level = tuple(levels[index])
        cell = np.exp2(levels[index].astype(float)) / 4.0
        x, y = np.floor(center / cell).astype(int)
        buckets.setdefault((level[0], level[1], x, y), []).append(index)
    return np.asarray(kept, dtype=int), parents
