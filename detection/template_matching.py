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
