"""Preprocessing shared by every feature: noise reduction, color space conversions, orange mask.

Works the same on Vistas template crops and full scene images.
"""
import cv2
import numpy as np

# NLM parameters: h = filter strength for luminance, h_color = for color components
H = 5
H_COLOR = 5
TEMPLATE_WINDOW_SIZE = 7
SEARCH_WINDOW_SIZE = 21


def denoise(img, h=H, h_color=H_COLOR, template_window_size=TEMPLATE_WINDOW_SIZE,
            search_window_size=SEARCH_WINDOW_SIZE):
    """OpenCV Non-Local Means denoising of a uint8 color image (BGR or RGB, channel order is kept)."""
    return cv2.fastNlMeansDenoisingColored(img, None, h, h_color, template_window_size, search_window_size)


def to_hsv(img):
    """Convert a uint8 BGR image (as read by cv2.imread) to HSV.

    OpenCV ranges: H in [0, 179], S and V in [0, 255].
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def to_lab(img):
    """Convert a uint8 BGR image (as read by cv2.imread) to CIE Lab.

    OpenCV 8-bit ranges: L in [0, 255] (L* scaled by 255/100), a and b in [0, 255] with 128 = neutral
    (a: green -> red, b: blue -> yellow).
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


def to_gray(img):
    """Convert a uint8 BGR image (as read by cv2.imread) to 8-bit grayscale in [0, 255].

    Luminance 0.299 R + 0.587 G + 0.114 B, the same formula as PIL's convert('L').
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)



# Orange in Lab: hue angle atan2(b*, a*) in ORANGE_HUE degrees and chroma sqrt(a*^2 + b*^2) > ORANGE_MIN_CHROMA.
# Chosen on training-split cone boxes: catches the most cone pixels per background pixel. (The Vistas templates
# were built with hue 42-75, chroma > 30, which misses the redder cones in this dataset.)
ORANGE_HUE = (25, 65)
ORANGE_MIN_CHROMA = 15


def orange_mask(img, hue=ORANGE_HUE, min_chroma=ORANGE_MIN_CHROMA):
    """True where a uint8 BGR image is orange."""
    lab = to_lab(img).astype(np.float32)
    a, b = lab[..., 1] - 128, lab[..., 2] - 128
    angle = np.degrees(np.arctan2(b, a))
    return (angle >= hue[0]) & (angle <= hue[1]) & (np.hypot(a, b) > min_chroma)
