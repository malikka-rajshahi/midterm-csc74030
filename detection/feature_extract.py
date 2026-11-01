"""Shared Color/Intensity extraction for train GT, validation regions and windows.

No dataset splitting, window generation, matching, threshold fitting or disk I/O.
Feature formulas stay in features/; existing common helpers handle preprocessing.
"""
import cv2
import numpy as np
from common.preprocessing import (denoise, to_gray, to_hsv, to_lab, H, H_COLOR,
                                  TEMPLATE_WINDOW_SIZE, SEARCH_WINDOW_SIZE)
from features.color import (rgb_histogram, hsv_histogram, lab_histogram,
                            RGB_BINS, HSV_BINS, LAB_BINS)
from features.intensity import (intensity_histogram, intensity_features,
                                STATISTIC_NAMES, GRAY_RANGE, TRIM_FRACTION)

COLOR_HISTOGRAMS = {
    'rgb_histogram': (rgb_histogram, RGB_BINS),
    'hsv_histogram': (hsv_histogram, HSV_BINS),
    'lab_histogram': (lab_histogram, LAB_BINS),
}
FEATURE_NAMES = (*COLOR_HISTOGRAMS, 'intensity_histogram', *STATISTIC_NAMES)


def _prepare_boxes(boxes, original_shape, work_shape, boxes_space):
    oh, ow = original_shape
    wh, ww = work_shape
    if boxes_space not in ('original', 'working'):
        raise ValueError('boxes_space must be original or working.')
    if boxes is None:
        working = np.array([[0, 0, ww, wh]], dtype=np.int32)
    else:
        values = np.asarray(boxes, dtype=float)
        if values.size == 0:
            values = np.empty((0, 4))
        if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
            raise ValueError('boxes must be finite (N,4) half-open xyxy coordinates.')
        width, height = (ow, oh) if boxes_space == 'original' else (ww, wh)
        if (values[:, :2] < 0).any() or (values[:, 2:] > [width, height]).any() or (values[:, 2:] <= values[:, :2]).any():
            raise ValueError('Every input box must have positive area inside its coordinate image.')
        if boxes_space == 'original':
            values = values * [ww/ow, wh/oh, ww/ow, wh/oh]
        working = np.c_[np.floor(values[:, :2]), np.ceil(values[:, 2:])].astype(np.int32)
        working = np.clip(working, [0, 0, 0, 0], [ww, wh, ww, wh])
    mapped = working.astype(float) * [ow/ww, oh/wh, ow/ww, oh/wh]
    original = np.c_[np.floor(mapped[:, :2]), np.ceil(mapped[:, 2:])].astype(np.int32)
    original = np.clip(original, [0, 0, 0, 0], [ow, oh, ow, oh])
    return working, original


def _bins(feature_names, overrides):
    overrides = {} if overrides is None else dict(overrides)
    histogram_names = set(COLOR_HISTOGRAMS) | {'intensity_histogram'}
    if set(overrides) - (histogram_names & set(feature_names)):
        raise ValueError('Bin overrides must name requested histogram features.')
    result = {}
    for name in feature_names:
        if name in COLOR_HISTOGRAMS:
            values = tuple(overrides.get(name, COLOR_HISTOGRAMS[name][1]))
            if len(values) != 3 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in values):
                raise ValueError('Color histogram bins must be three positive integers.')
            result[name] = tuple(map(int, values))
        elif name == 'intensity_histogram':
            value = overrides.get(name, 256)
            if not isinstance(value, (int, np.integer)) or not 1 <= value <= 256:
                raise ValueError('Intensity histogram bins must be an integer in [1,256].')
            result[name] = int(value)
    return result


def extract_features(image_bgr, boxes=None, feature_names=('intensity_histogram',), *,
                     work_size=None, boxes_space='working', apply_denoise=True,
                     histogram_bins=None):
    """Prepare once, extract separate representations on exactly the same rectangles.

    image_bgr: nonempty uint8 BGR image. work_size is optional (width,height).
    boxes: original or working half-open xyxy; None means the complete image.
    Floating GT boxes are rounded outward AFTER coordinate scaling. Invalid boxes
    are rejected rather than silently dropped. [] produces aligned empty arrays.
    feature_names: a string or tuple from FEATURE_NAMES. No feature fusion occurs.
    histogram_bins: optional per-histogram override, shared between all data roles.

    Return boxes (working), original_boxes, features {name:(N,D)}, valid_rows
    {name:(N,)}, preprocessing/feature config, and the prepared work_image_bgr.
    Scalars are (N,1). NaN spatial entropy for height-one patches remains explicit;
    use valid_rows when preparing references/calibration instead of replacing NaN.
    """
    if not isinstance(image_bgr, np.ndarray) or image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) == 0:
        raise ValueError('Expected a nonempty uint8 BGR image.')
    names = (feature_names,) if isinstance(feature_names, str) else tuple(feature_names)
    if not names or len(set(names)) != len(names) or set(names) - set(FEATURE_NAMES):
        raise ValueError(f'Choose distinct feature names from {FEATURE_NAMES}.')
    bins = _bins(names, histogram_bins)
    if work_size is None:
        work = image_bgr.copy()
    else:
        if len(work_size) != 2 or any(not isinstance(n, (int, np.integer)) or n < 1 for n in work_size):
            raise ValueError('work_size must be positive integer (width,height).')
        work = cv2.resize(image_bgr, tuple(work_size), interpolation=cv2.INTER_AREA)
    if apply_denoise:
        work = denoise(work)
    working, original = _prepare_boxes(boxes, image_bgr.shape[:2], work.shape[:2], boxes_space)
    spaces = {'rgb_histogram': work}
    if 'hsv_histogram' in names:
        spaces['hsv_histogram'] = to_hsv(work)
    if 'lab_histogram' in names:
        spaces['lab_histogram'] = to_lab(work)
    scalar_names = [name for name in names if name in STATISTIC_NAMES]
    need_gray = bool(scalar_names) or 'intensity_histogram' in names
    gray = to_gray(work) if need_gray else None
    dimensions = {name: sum(bins[name]) if name in COLOR_HISTOGRAMS else
                  bins[name] if name == 'intensity_histogram' else 1 for name in names}
    features = {name: np.empty((len(working), dimensions[name]), float) for name in names}
    for index, (x0, y0, x1, y1) in enumerate(working):
        for name in names:
            if name in COLOR_HISTOGRAMS:
                extractor = COLOR_HISTOGRAMS[name][0]
                features[name][index] = extractor(spaces[name][y0:y1, x0:x1], bins=bins[name])
        if need_gray:
            patch = gray[y0:y1, x0:x1]
            combined = intensity_features(patch) if scalar_names else None
            if 'intensity_histogram' in names:
                features['intensity_histogram'][index] = (combined[:256] if combined is not None and bins['intensity_histogram'] == 256
                                                        else intensity_histogram(patch, bins['intensity_histogram']))
            for name in scalar_names:
                features[name][index, 0] = combined[256 + STATISTIC_NAMES.index(name)]
    config = dict(work_size=[work.shape[1], work.shape[0]], denoise=bool(apply_denoise),
                  denoise_parameters=dict(h=H, h_color=H_COLOR, template_window_size=TEMPLATE_WINDOW_SIZE,
                                          search_window_size=SEARCH_WINDOW_SIZE) if apply_denoise else None,
                  feature_names=list(names), dimensions=dimensions,
                  histogram_bins={name: list(value) if isinstance(value, tuple) else value for name, value in bins.items()},
                  statistics=dict(trim_fraction=TRIM_FRACTION, gray_range=list(GRAY_RANGE),
                                  variance='population', entropy='base2', mode_ties='lowest'))
    return dict(boxes=working, original_boxes=original, features=features,
                valid_rows={name: np.isfinite(value).all(axis=1) for name, value in features.items()},
                config=config, input_coordinate_space=boxes_space, work_image_bgr=work)
