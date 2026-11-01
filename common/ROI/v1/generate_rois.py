"""Class-independent proposals from multi-scale colored-block clusters.

Input is uint8 BGR. Coordinates are original-image, half-open (x0,y0,x1,y1).
No cone/sign decision, feature classifier, morphology, or trained weights live here.
"""
import cv2
import numpy as np
from common.preprocessing import denoise, to_hsv
from .sliding_window import deduplicate_boxes

# Initial broad color evidence, in degrees. Tune only on official training data.
COLOR_BANDS = {
    'red': ((0, 15), (345, 360)),
    'orange': ((15, 45),),
    'yellow': ((45, 65),),
    'green': ((65, 170),),
    'blue': ((190, 260),),
}
SCALES = (1.0, 0.5, 0.25)
BLOCK_SIZES = (4, 8, 16)
PROPOSAL_COLORS = ('red', 'orange', 'yellow')


def color_evidence(image_bgr, saturation_min=0.20, value_min=0.12, yellow_max=65):
    """Boolean estimated color masks, not object labels or ground-truth masks."""
    hsv = to_hsv(image_bgr)
    hue = hsv[..., 0].astype(float) * 2.0  # OpenCV H=0..179 -> degrees
    saturation = hsv[..., 1].astype(float) / 255.0
    value = hsv[..., 2].astype(float) / 255.0
    chromatic = (saturation >= saturation_min) & (value >= value_min)
    masks = {}
    bands_by_color = dict(COLOR_BANDS, yellow=((45, yellow_max),), green=((yellow_max, 170),))
    for name, bands in bands_by_color.items():
        selected = np.zeros(hue.shape, dtype=bool)
        for low, high in bands:
            selected |= (hue >= low) & (hue < high)
        masks[name] = selected & chromatic
    # White is recorded as evidence, never used alone to generate a proposal.
    masks['white'] = (saturation <= 0.18) & (value >= 0.60)
    return masks


def _block_regions(mask, block_size, min_fraction, min_pixels):
    """Group adjacent active blocks; divide edge blocks by their actual area."""
    height, width = mask.shape
    rows = (height + block_size - 1) // block_size
    cols = (width + block_size - 1) // block_size
    padded = np.pad(mask, ((0, rows * block_size - height),
                          (0, cols * block_size - width)))
    counts = padded.reshape(rows, block_size, cols, block_size).sum(axis=(1, 3))
    cell_h = np.minimum(block_size, height - np.arange(rows) * block_size)
    cell_w = np.minimum(block_size, width - np.arange(cols) * block_size)
    area = cell_h[:, None] * cell_w[None, :]
    active = ((counts >= min_pixels) & (counts / area >= min_fraction)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(active, connectivity=8)
    for x, y, w, h, _ in stats[1:]:
        yield (int(x * block_size), int(y * block_size),
               min(width, int((x + w) * block_size)),
               min(height, int((y + h) * block_size)))


def _map_box(box, original_shape, scaled_shape, padding):
    """Use actual resize dimensions; round outward, then clip to the image."""
    height, width = original_shape
    small_h, small_w = scaled_shape
    x0, y0, x1, y1 = box
    x0, x1 = x0 * width / small_w, x1 * width / small_w
    y0, y1 = y0 * height / small_h, y1 * height / small_h
    dx, dy = (x1 - x0) * padding, (y1 - y0) * padding
    return (max(0, int(np.floor(x0 - dx))), max(0, int(np.floor(y0 - dy))),
            min(width, int(np.ceil(x1 + dx))), min(height, int(np.ceil(y1 + dy))))


def generate_proposals(image_bgr, *, apply_denoise=True, scales=SCALES,
                       block_sizes=BLOCK_SIZES, min_color_fraction=0.20,
                       min_colored_pixels=2, padding=0.12,
                       max_area_fraction=0.90, return_masks=False,
                       dedup_iou=0.90, return_raw=False,
                       proposal_colors=PROPOSAL_COLORS, yellow_max=65, max_green_fraction=0.50):
    """Return processed image, boxes and aligned color evidence for every proposal.

    Each color branch and their union produces block-cluster boxes. Individual and
    larger grouped regions are retained. Exact duplicates are merged, then a
    conservative geometric IoU gate removes near duplicates (disable with None).
    Tight and padded variants are retained to protect small targets. No class-specific
    aspect ratio, white requirement, classification score or output cap is used.
    """
    if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8:
        raise ValueError('Expected a uint8 BGR numpy array.')
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) == 0:
        raise ValueError('Expected a nonempty height x width x 3 image.')
    if not scales or any(not 0 < scale <= 1 for scale in scales):
        raise ValueError('Scales must be in (0,1].')
    if not block_sizes or any(not isinstance(size, int) or size < 1 for size in block_sizes):
        raise ValueError('Block sizes must be positive integers.')
    if not 0 <= min_color_fraction <= 1 or not 0 < max_area_fraction <= 1 or padding < 0:
        raise ValueError('Invalid fraction or padding.')
    if not isinstance(min_colored_pixels, int) or min_colored_pixels < 1:
        raise ValueError('min_colored_pixels must be a positive integer.')

    if not proposal_colors or set(proposal_colors) - {'red', 'orange', 'yellow', 'blue'}:
        raise ValueError('Seeds must be red/orange/yellow/blue; green and white are evidence only.')
    if not 45 < yellow_max <= 75:
        raise ValueError('yellow_max must be in (45,75] degrees.')
    if max_green_fraction is not None and not 0 <= max_green_fraction <= 1:
        raise ValueError('max_green_fraction must be in [0,1], or None.')
    processed = denoise(image_bgr) if apply_denoise else image_bgr.copy()
    height, width = processed.shape[:2]
    full_masks = color_evidence(processed, yellow_max=yellow_max)
    boxes_and_sources = {}
    for scale in scales:
        small_w, small_h = max(1, round(width * scale)), max(1, round(height * scale))
        small = processed if (small_h, small_w) == (height, width) else cv2.resize(
            processed, (small_w, small_h), interpolation=cv2.INTER_AREA)
        evidence_masks = color_evidence(small, yellow_max=yellow_max)
        masks = {name: evidence_masks[name] for name in proposal_colors}
        masks['combined'] = np.logical_or.reduce(list(masks.values()))
        for branch, mask in masks.items():
            for size in block_sizes:
                for local in _block_regions(mask, size, min_color_fraction, min_colored_pixels):
                    for pad in ((0.0,) if padding == 0 else (0.0, padding)):
                        box = _map_box(local, (height, width), (small_h, small_w), pad)
                        x0, y0, x1, y1 = box
                        if (x1 - x0) * (y1 - y0) > height * width * max_area_fraction:
                            continue
                        source = dict(color_branch=branch, scale=float(scale),
                                      block_size=size, padding=pad)
                        boxes_and_sources.setdefault(box, []).append(source)

    evidence = []
    for box, sources in boxes_and_sources.items():
        x0, y0, x1, y1 = box
        ratios = {color: float(mask[y0:y1, x0:x1].mean())
                  for color, mask in full_masks.items()}
        evidence.append(dict(color_ratios=ratios, sources=sources,
                             area_fraction=(x1 - x0) * (y1 - y0) / (height * width)))
    boxes = np.array(list(boxes_and_sources), dtype=np.int32).reshape(-1, 4)
    green_rejected = 0
    if max_green_fraction is not None:
        keep = np.array([info['color_ratios']['green'] <= max_green_fraction
                         for info in evidence], dtype=bool)
        green_rejected = int((~keep).sum())
        boxes = boxes[keep]
        evidence = [info for info, accepted in zip(evidence, keep) if accepted]
    raw_boxes = boxes
    kept, parents = deduplicate_boxes(raw_boxes, dedup_iou)
    for index, parent in enumerate(parents):
        if index != parent:
            evidence[parent]['sources'].extend(evidence[index]['sources'])
    return dict(image_bgr=processed, boxes=raw_boxes[kept],
                evidence=[evidence[index] for index in kept], raw_count=len(raw_boxes),
                green_rejected_count=green_rejected,
                raw_boxes=raw_boxes if return_raw else None,
                color_masks=full_masks if return_masks else None)
