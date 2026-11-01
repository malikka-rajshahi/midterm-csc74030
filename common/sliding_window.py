"""Multi-scale sliding windows and their histograms.

Window shape per class = median height / width of that class's training boxes. Window sizes run from a low to a
high percentile of the training box widths, growing by SCALE_STEP each time (the slides' "multiple scales").
Histograms of all windows come from an integral histogram, so each window costs a few lookups per bin.
"""
import cv2
import numpy as np

SCALE_STEP = 1.25
SIZE_PERCENTILES = (10, 90)
# Prescreen: on training images, 2% orange keeps ~1.6% of cone windows while 89% of cones keep a window with IoU >= 0.5
MIN_ORANGE_FRACTION = 0.02


def window_sizes(boxes, step=SCALE_STEP, percentiles=SIZE_PERCENTILES):
    """(width, height) of every window scale for one class.

    boxes: that class's training boxes as [x, y, w, h] rows (the dataset's bbox_coords).
    """
    w, h = np.asarray(boxes, dtype=float)[:, 2:].T
    return size_range(*np.percentile(w, percentiles), np.median(h / w), step)


def size_range(low, high, aspect, step=SCALE_STEP):
    """(width, height) of windows from width low up to (at least) high, growing by step; height = aspect * width."""
    widths = low * step ** np.arange(int(np.ceil(np.log(high / low) / np.log(step))) + 1)
    sizes = [(max(1, round(width)), max(1, round(width * aspect))) for width in widths]
    return list(dict.fromkeys(sizes))  # drop repeats from rounding, keep order


def windows(image_shape, width, height, stride=None):
    """(x0, y0, x1, y1) of every width x height window, sliding by stride (default width // 2).

    The last row and column of windows are aligned to the image edge so the whole image is covered.
    Returns an empty array if the window is bigger than the image.
    """
    rows, cols = image_shape[:2]
    if width > cols or height > rows:
        return np.zeros((0, 4), dtype=np.int32)
    stride = stride or max(1, width // 2)
    xs = np.unique(np.r_[np.arange(0, cols - width + 1, stride), cols - width])
    ys = np.unique(np.r_[np.arange(0, rows - height + 1, stride), rows - height])
    x, y = np.meshgrid(xs, ys)
    x, y = x.ravel(), y.ravel()
    return np.column_stack([x, y, x + width, y + height]).astype(np.int32)


def multiscale_windows(image_shape, sizes, stride_fraction=0.5):
    """Windows of every size in sizes, stacked. Each size slides by stride_fraction of its width."""
    return np.concatenate([windows(image_shape, w, h, max(1, round(w * stride_fraction))) for w, h in sizes])


def window_histograms(img, boxes, bins, channel_max):
    """Per-channel normalized histograms of every box in img, same values as features.color.channel_histograms.

    img: already converted to the color space (e.g. SPACES["hsv"]["convert"](bgr)).
    boxes: (n, 4) array of (x0, y0, x1, y1), any mix of sizes.
    Returns an (n, sum(bins)) float32 array.

    Integral histogram: for each bin, a summed-area table of "pixel is in this bin" gives the bin's count
    in any box from 4 lookups.
    """
    x0, y0, x1, y1 = np.asarray(boxes).T
    area = ((x1 - x0) * (y1 - y0)).astype(np.float32)
    hists = np.empty((len(x0), sum(bins)), dtype=np.float32)
    col = 0
    for c, (n_bins, max_val) in enumerate(zip(bins, channel_max)):
        bin_index = img[..., c].astype(np.int32) * n_bins // max_val
        for b in range(n_bins):
            integral = cv2.integral((bin_index == b).astype(np.uint8))
            hists[:, col] = (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) / area
            col += 1
    return hists


def box_fractions(mask, boxes):
    """Fraction of True pixels of a boolean mask inside each (x0, y0, x1, y1) box, from one integral image."""
    x0, y0, x1, y1 = np.asarray(boxes).T
    integral = cv2.integral(mask.astype(np.uint8))
    return (integral[y1, x1] - integral[y0, x1] - integral[y1, x0] + integral[y0, x0]) / ((x1 - x0) * (y1 - y0))


def prescreen(mask, boxes, min_fraction=MIN_ORANGE_FRACTION):
    """Keep only boxes where at least min_fraction of the pixels are True in mask (e.g. preprocessing.orange_mask)."""
    boxes = np.asarray(boxes)
    return boxes[box_fractions(mask, boxes) >= min_fraction]
