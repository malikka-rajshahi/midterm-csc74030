"""Shared candidate proposals; no feature classifier or object-class decision."""
from .generate_rois import color_evidence, generate_proposals
from .sliding_window import sliding_windows, filter_color_windows, deduplicate_boxes

__all__ = ['color_evidence', 'generate_proposals', 'sliding_windows',
           'filter_color_windows', 'deduplicate_boxes']
