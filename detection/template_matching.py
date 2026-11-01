"""Template histograms from the Vistas crops, for matching against scene windows.

Each template is a whole Vistas crop (no mask, so it matches unmasked sliding windows), denoised,
then histogrammed in every color space.
"""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common.preprocessing import denoise
from features.color import SPACES, channel_histograms

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"


def load_templates(role="template"):
    """Rows of templates.csv with the given role ("template", or "validation" for the held-out Vistas crops)."""
    templates = pd.read_csv(TEMPLATE_DIR / "templates.csv")
    return templates[templates["role"] == role].reset_index(drop=True)


def template_histograms(templates=None):
    """Histograms of every template crop in every color space.

    Returns {space: {class: array of shape (number of templates of that class, histogram length)}}.
    """
    templates = load_templates() if templates is None else templates
    crops = [denoise(cv2.imread(str(TEMPLATE_DIR / crop))) for crop in templates["crop"]]
    hists = {}
    for space, sp in SPACES.items():
        all_hists = np.array([channel_histograms(sp["convert"](crop), sp["bins"], sp["channel_max"]) for crop in crops])
        hists[space] = {cls: all_hists[(templates["class"] == cls).to_numpy()] for cls in sorted(templates["class"].unique())}
    return hists


def d3(a, b):
    """D3 distance from the slides, sum((a - b)^2 / max(a, b)), between every row of a and every row of b.

    a: (n, k), b: (m, k). Returns (n, m). Bins that are empty in both histograms contribute 0.
    """
    a, b = a[:, None, :], b[None, :, :]
    m = np.maximum(a, b)
    return np.sum(np.divide((a - b) ** 2, m, out=np.zeros(m.shape, dtype=np.float32), where=m > 0), axis=-1)


def nearest_template_distance(window_hists, template_hists, chunk=1024):
    """D3 distance from each window histogram to its closest template histogram. Returns (n,).

    Computed chunk windows at a time so memory stays at chunk x templates x bins.
    """
    window_hists = np.asarray(window_hists, dtype=np.float32)
    template_hists = np.asarray(template_hists, dtype=np.float32)
    return np.concatenate([d3(window_hists[i:i + chunk], template_hists).min(axis=1)
                           for i in range(0, len(window_hists), chunk)] or [np.zeros(0, np.float32)])


REFERENCE_FILE = Path(__file__).resolve().parent.parent / "data" / "reference.npz"


def load_reference(path=REFERENCE_FILE):
    """Template histograms saved by scripts/build_reference.py, as {space: {class: array}}."""
    with np.load(path) as data:
        reference = {}
        for key in data.files:
            space, cls = key.split("/")
            reference.setdefault(space, {})[cls] = data[key]
    return reference
