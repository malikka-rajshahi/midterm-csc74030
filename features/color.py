"""Color histogram features.

Each channel is histogrammed separately, divided by the region's pixel count so its bins sum to 1,
and the channel histograms are concatenated into one feature vector.
"""
import numpy as np

from common.preprocessing import to_hsv, to_lab

RGB_BINS = (16, 16, 16)
RGB_CHANNEL_MAX = (256, 256, 256)
# 18 hue bins, 8 saturation bins, 8 value bins. OpenCV hue runs 0-179 (degrees / 2), so each hue bin is 20 degrees
HSV_BINS = (18, 8, 8)
HSV_CHANNEL_MAX = (180, 256, 256)
# 8 lightness bins, 16 bins each for a and b (orange shows up as high a and high b)
LAB_BINS = (8, 16, 16)
LAB_CHANNEL_MAX = (256, 256, 256)

# Color space name -> how to get there from a BGR image (cv2.imread), and how to histogram it.
# channel_histograms(convert(img), bins, channel_max) gives the same result as that space's *_histogram function.
SPACES = {
    "rgb": dict(convert=lambda img: img[..., ::-1], bins=RGB_BINS, channel_max=RGB_CHANNEL_MAX),
    "hsv": dict(convert=to_hsv, bins=HSV_BINS, channel_max=HSV_CHANNEL_MAX),
    "lab": dict(convert=to_lab, bins=LAB_BINS, channel_max=LAB_CHANNEL_MAX),
}


def channel_histograms(img, bins, channel_max):
    """Per-channel histograms of an image, each normalized to sum to 1, concatenated."""
    n_pixels = img.shape[0] * img.shape[1]
    return np.concatenate([
        np.bincount((img[..., c].astype(np.int32) * n_bins // max_val).ravel(), minlength=n_bins) / n_pixels
        for c, (n_bins, max_val) in enumerate(zip(bins, channel_max))
    ])


def rgb_histogram(img, bins=RGB_BINS):
    """R, G and B histograms of a uint8 BGR image (as read by cv2.imread), in R, G, B order.

    img can be a whole image, a template crop or a sliding window cut from a scene.
    Returns sum(bins) values: the first bins[0] are R, then G, then B.
    """
    return channel_histograms(img[..., ::-1], bins, RGB_CHANNEL_MAX)


def hsv_histogram(img, bins=HSV_BINS):
    """H, S and V histograms of an HSV image (from common.preprocessing.to_hsv).

    img can be a whole image, a template crop or a sliding window cut from a scene.
    Returns sum(bins) values: the first bins[0] are H, then S, then V.
    """
    return channel_histograms(img, bins, HSV_CHANNEL_MAX)


def lab_histogram(img, bins=LAB_BINS):
    """L, a and b histograms of a Lab image (from common.preprocessing.to_lab).

    img can be a whole image, a template crop or a sliding window cut from a scene.
    Returns sum(bins) values: the first bins[0] are L, then a, then b.
    """
    return channel_histograms(img, bins, LAB_CHANNEL_MAX)

