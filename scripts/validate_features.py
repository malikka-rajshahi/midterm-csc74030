"""Tune the D3 matching threshold of each color space and class on the validation split.

    python scripts/run_detection.py --split validation     # first
    python scripts/validate_features.py

For every (space, class): match the saved detections to the validation boxes (IoU >= 0.5), sweep the distance
threshold, and pick the threshold with the best F1. Also reports average precision (AP), which does not depend
on the threshold.

Saves results/thresholds.json ({space: {class: threshold}}), results/validation_scores.csv and
results/validation_curves.parquet (the full precision-recall curves).
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.dataset import SPLIT_FILE, load_dataset, map_labels  # noqa: E402
from evaluation.metrics import average_precision, ground_truth_boxes, match, pr_curve, scores_at  # noqa: E402

RESULTS_DIR = ROOT / "results"


def main():
    detections = pd.read_parquet(RESULTS_DIR / "detections_validation.parquet")
    images = set(json.loads(SPLIT_FILE.read_text())["validation"])
    missing = len(images - set(detections["image_path"]))
    if missing:
        print(f"warning: {missing} validation images have no detections yet; scoring only the {len(images) - missing} done")
        images &= set(detections["image_path"])
    annotations = map_labels(load_dataset())
    gt = ground_truth_boxes(annotations[annotations["image_path"].isin(images)])

    thresholds, scores, curves = {}, [], []
    for (space, cls), dets in detections.groupby(["space", "cls"]):
        cls_gt = gt[gt["cls"] == cls]
        curve = pr_curve(match(dets, cls_gt), len(cls_gt))
        best = curve.loc[curve["f1"].idxmax()] if len(curve) else None
        threshold = float(best["distance"]) if best is not None else 0.0
        thresholds.setdefault(space, {})[cls] = threshold
        scores.append(dict(space=space, cls=cls, ground_truth=len(cls_gt), threshold=threshold,
                           ap=average_precision(curve), **scores_at(curve, threshold, len(cls_gt))))
        curves.append(curve.assign(space=space, cls=cls))

    scores = pd.DataFrame(scores)
    (RESULTS_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=1))
    scores.to_csv(RESULTS_DIR / "validation_scores.csv", index=False)
    pd.concat(curves, ignore_index=True).to_parquet(RESULTS_DIR / "validation_curves.parquet")
    print(f"validation: {len(images)} images")
    print(scores.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
