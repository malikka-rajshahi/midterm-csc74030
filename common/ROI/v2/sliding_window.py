"""Color-anchored window placement and bounded overlap merging, without classes."""
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


def merge_overlaps(boxes, sources, background_ok, *, overlap=0.5, growth=1.5,
                   max_area=None, max_scale_ratio=4.0):
    """Intersection/min-area merge; bound growth against a fixed original seed.

    Largest seeds are visited first. A cluster never gets another merging pass.
    Failed unions leave the original candidate available. Containment keeps the
    larger box; no class scores, NMS, repeated transitive closure or top-K cap.
    """
    boxes = np.asarray(boxes, dtype=np.int32).reshape(-1,4)
    if not 0 < overlap <= 1 or growth < 1:
        raise ValueError('Invalid overlap/growth settings.')
    areas = np.prod(boxes[:,2:]-boxes[:,:2],axis=1)
    order = np.argsort(-areas,kind='stable')
    available = np.ones(len(boxes),dtype=bool)
    merged, provenance = [], []
    for index in order:
        if not available[index]:
            continue
        seed = boxes[index].copy()
        box = seed.copy()
        ids = set(sources[index])
        available[index] = False
        # Candidates are original boxes; expansion remains bounded by this seed.
        remaining=order[available[order]]
        intersection=np.maximum(0,np.minimum(seed[2:],boxes[remaining,2:])-
                                np.maximum(seed[:2],boxes[remaining,:2])).prod(axis=1)
        eligible=intersection/np.minimum(areas[index],areas[remaining])>=overlap
        # Do not erase much smaller useful regions merely because a group contains them.
        eligible &= areas[index]/areas[remaining]<=max_scale_ratio
        for other in remaining[eligible]:
            intersection = np.maximum(0, np.minimum(box[2:],boxes[other,2:])-
                                       np.maximum(box[:2],boxes[other,:2])).prod()
            area = np.prod(box[2:]-box[:2])
            if intersection/min(area,areas[other]) < overlap:
                continue
            union = np.r_[np.minimum(box[:2],boxes[other,:2]),np.maximum(box[2:],boxes[other,2:])]
            union_size = union[2:]-union[:2]
            if (union_size > (seed[2:]-seed[:2])*growth).any():
                continue
            if union_size.prod() > areas[index]*growth:
                continue
            if max_area is not None and union_size.prod() > max_area:
                continue
            if not background_ok(union.reshape(1,4))[0]:
                continue
            box=union
            ids.update(sources[other])
            available[other]=False
        merged.append(box)
        provenance.append(sorted(ids))
    # Exact deduplication after merging, preserving all anchor provenance.
    unique={}
    for box,ids in zip(merged,provenance):
        unique.setdefault(tuple(box),set()).update(ids)
    return np.array(list(unique),dtype=np.int32).reshape(-1,4), [sorted(ids) for ids in unique.values()]
