"""Color-anchored window placement and intersection-group closure, without classes."""
import numpy as np


def integral(mask):
    return np.pad(mask.astype(np.int64), ((1, 0), (1, 0))).cumsum(0).cumsum(1)


def fractions(boxes, summed_mask):
    boxes = np.asarray(boxes, dtype=int).reshape(-1, 4)
    x0, y0, x1, y1 = boxes.T
    counts = summed_mask[y1, x1]-summed_mask[y0, x1]-summed_mask[y1, x0]+summed_mask[y0, x0]
    return counts / ((x1-x0)*(y1-y0))


def anchored_windows(anchors, window_sizes, image_shape, padding=0.25, nearest=3, max_anchor_area_ratio=4.0):
    """Tighten anchors upstream; keep context here, then try nearby window positions.

    Every box contains its anchor. No GT or class is used at image inference.
    Returns unique xyxy boxes and the anchor IDs supporting each box.
    """
    height, width = image_shape[:2]
    bank = np.asarray(window_sizes, dtype=float).reshape(-1, 2)
    if not len(bank) or (bank <= 0).any() or not np.isfinite(bank).all():
        raise ValueError('Window sizes must be finite positive width/height pairs.')
    candidates = {}
    for anchor_id, anchor in enumerate(anchors):
        lo, hi = anchor[:2], anchor[2:]
        size = hi-lo
        desired = size*(1+2*padding)
        compatible=(bank>=desired).all(axis=1)&(bank<=desired*3).all(axis=1)
        compatible &= bank.prod(axis=1)<=desired.prod()*max_anchor_area_ratio
        sizes = bank[compatible]
        if len(sizes):
            rank = np.square(np.log(sizes/desired)).sum(axis=1).argsort(kind='stable')[:nearest]
            sizes = sizes[rank]
        sizes = np.vstack([desired, sizes]).round().astype(int)
        for bw, bh in np.unique(sizes, axis=0):
            bw, bh = min(width, max(int(size[0]), bw)), min(height, max(int(size[1]), bh))
            center = (lo+hi)/2
            for dx, dy in [(0,0), (-.15,0), (.15,0), (0,-.15), (0,.15)]:
                start = np.round(center + np.array([dx*bw,dy*bh]) - [bw/2,bh/2]).astype(int)
                # Keep the anchor intact even for offset windows and image-edge boxes.
                start = np.maximum(hi-[bw,bh], np.minimum(start, lo))
                start = np.clip(start, [0,0], [width-bw,height-bh])
                box = tuple(np.r_[start, start+[bw,bh]])
                candidates.setdefault(box, set()).add(anchor_id)
    boxes = np.array(list(candidates), dtype=np.int32).reshape(-1,4)
    return boxes, [sorted(ids) for ids in candidates.values()]


def merge_overlaps(boxes, sources, background_ok):
    """One output per intersecting group, including overlaps created by envelopes.

    Positive-area intersection joins boxes; touching edges do not. Each closure
    round reduces the number of groups, so iteration terminates. This deliberately
    replaces V3's growth/scale constraints to remove residual intersecting outputs.
    Large groups are flagged by the caller, never disguised as compact objects.
    """
    boxes = np.asarray(boxes, dtype=np.int32).reshape(-1, 4)
    sources = [set(ids) for ids in sources]
    while len(boxes):
        parents = np.arange(len(boxes))
        def find(index):
            while parents[index] != index:
                parents[index] = parents[parents[index]]
                index = parents[index]
            return index
        for index in range(len(boxes)):
            size = np.minimum(boxes[index, 2:], boxes[index+1:, 2:]) - np.maximum(boxes[index, :2], boxes[index+1:, :2])
            for other in np.flatnonzero((size > 0).all(axis=1)) + index + 1:
                a, b = find(index), find(other)
                if a != b:
                    parents[b] = a
        groups = {}
        for index in range(len(boxes)):
            groups.setdefault(find(index), []).append(index)
        if len(groups) == len(boxes):
            break
        merged, provenance = [], []
        for ids in groups.values():
            merged.append(np.r_[boxes[ids, :2].min(axis=0), boxes[ids, 2:].max(axis=0)])
            provenance.append(set().union(*(sources[index] for index in ids)))
        boxes = np.asarray(merged, dtype=np.int32).reshape(-1, 4)
        sources = provenance
    accepted = background_ok(boxes)
    return boxes[accepted], [sorted(ids) for ids, keep in zip(sources, accepted) if keep]
