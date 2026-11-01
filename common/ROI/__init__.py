"""Default ROI API: V4. Historical versions remain independent."""
from .v4 import generate_proposals, color_evidence, load_config
__all__ = ['generate_proposals', 'color_evidence', 'load_config']
