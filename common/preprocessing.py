"""Preprocessing shared by every feature: noise reduction and color space conversions.

Works the same on Vistas template crops and full scene images.
"""
import cv2

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

