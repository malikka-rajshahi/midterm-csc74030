"""Grayscale histogram and intensity statistics, matching color.py's array API.

Preprocessing is external: pass uint8 grayscale from common.preprocessing.to_gray.
Every pixel in the rectangular input is included, including background.
"""
import numpy as np

INTENSITY_BINS = 256
GRAY_RANGE = (64, 192)
TRIM_FRACTION = 0.10
STATISTIC_NAMES = ('mean', 'median', 'mode', 'variance', 'std', 'trimmed_mean',
                   'entropy', 'gray_range_fraction', 'entropy_contrast',
                   'normalized_contrast', 'spatial_entropy_difference')
LEVELS = np.arange(256, dtype=np.float64)


def _check_gray(img):
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8 or img.ndim != 2 or not img.size:
        raise ValueError('Expected a nonempty uint8 grayscale image.')


def intensity_histogram(img, bins=INTENSITY_BINS):
    """Normalized grayscale histogram; whole image, GT crop or ROI patch input."""
    _check_gray(img)
    if not isinstance(bins, (int, np.integer)) or not 1 <= bins <= 256:
        raise ValueError('bins must be an integer in [1,256].')
    indices = img.ravel().astype(np.int32) * bins // 256
    return np.bincount(indices, minlength=bins) / img.size


def _entropy(counts):
    p = counts[counts > 0] / counts.sum()
    return float(-np.dot(p, np.log2(p)))


def _statistics(img, counts, trim_fraction, gray_range):
    n = img.size
    p = counts / n
    mean = float(np.dot(LEVELS, p))
    variance = float(np.dot((LEVELS - mean) ** 2, p))
    std = np.sqrt(variance)
    cumulative = np.cumsum(counts)
    middle = np.searchsorted(cumulative, [(n-1)//2, n//2], side='right')
    median = float(middle.mean())
    cut = int(np.floor(trim_fraction * n))
    # Count exact surviving order statistics without sorting the patch.
    retained = np.maximum(0, np.minimum(cumulative, n-cut) - np.maximum(cumulative-counts, cut))
    trimmed = float(np.dot(LEVELS, retained) / (n-2*cut))
    entropy = _entropy(counts)
    split = img.shape[0] // 2
    spatial = np.nan
    if split > 0:
        top = np.bincount(img[:split].ravel(), minlength=256)
        bottom = np.bincount(img[split:].ravel(), minlength=256)
        spatial = abs(_entropy(top) - _entropy(bottom))
    low, high = gray_range
    return np.array([mean, median, np.argmax(counts), variance, std, trimmed,
                     entropy, p[low:high+1].sum(), entropy*std, std/(mean+1), spatial])


def intensity_statistics(img, trim_fraction=TRIM_FRACTION, gray_range=GRAY_RANGE):
    """Eleven scalars in STATISTIC_NAMES order; population variance, entropy bits.

    Median averages middle values; mode ties choose the lowest level. Trim both
    tails by floor(trim_fraction*N); range endpoints are inclusive. Spatial entropy
    is abs(H(top)-H(bottom)); height-one patches return NaN for that statistic only.
    """
    _check_gray(img)
    if not 0 <= trim_fraction < .5:
        raise ValueError('trim_fraction must be in [0,0.5).')
    if len(gray_range) != 2 or any(int(v) != v for v in gray_range) or not 0 <= gray_range[0] <= gray_range[1] <= 255:
        raise ValueError('Invalid inclusive gray range.')
    counts = np.bincount(img.ravel(), minlength=256)
    return _statistics(img, counts, trim_fraction, tuple(map(int, gray_range)))


def intensity_features(img):
    """256-bin normalized histogram followed by eleven scalars: 267 values.

    These represent 12 feature methods; the histogram is one vector method.
    Values are not standardized or matched/classified by this extractor.
    """
    _check_gray(img)
    counts = np.bincount(img.ravel(), minlength=256)
    return np.r_[counts/img.size, _statistics(img, counts, TRIM_FRACTION, GRAY_RANGE)]


def window_intensity_features(img, boxes):
    """Return (boxes, vectors) for supplied rectangular half-open xyxy windows.

    Boxes come from common ROI/refinement; no second sliding-window logic here.
    Vectors are float64 (N,267). Use chunks/cast for large saved candidate pools.
    """
    _check_gray(img)
    boxes = np.asarray(boxes)
    if boxes.size == 0:
        return np.empty((0,4), np.int32), np.empty((0,267), float)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all() or (boxes != np.floor(boxes)).any():
        raise ValueError('Expected finite integer-coordinate (N,4) boxes.')
    boxes = boxes.astype(np.int32)
    if (boxes[:,:2] < 0).any() or (boxes[:,2:] > [img.shape[1],img.shape[0]]).any() or (boxes[:,2:] <= boxes[:,:2]).any():
        raise ValueError('Boxes must have positive area inside the grayscale image.')
    features = np.empty((len(boxes),267), dtype=float)
    for index, (x0,y0,x1,y1) in enumerate(boxes):
        features[index] = intensity_features(img[y0:y1,x0:x1])
    return boxes, features
