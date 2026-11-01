"""Extract the Vistas template features: RGB, HSV and Lab histograms of every template crop.

    python scripts/build_reference.py

Saves data/reference.npz (load it with detection.template_matching.load_reference).
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detection.template_matching import REFERENCE_FILE, template_histograms  # noqa: E402

if __name__ == "__main__":
    hists = template_histograms()
    REFERENCE_FILE.parent.mkdir(exist_ok=True)
    np.savez(REFERENCE_FILE, **{f"{space}/{cls}": h for space, by_class in hists.items() for cls, h in by_class.items()})
    for space, by_class in hists.items():
        print(space, {cls: h.shape for cls, h in by_class.items()})
    print(f"saved {REFERENCE_FILE}")
