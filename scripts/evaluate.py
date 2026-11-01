"""Score saved detections with the frozen thresholds from validation (nothing is tuned here).

    python scripts/run_detection.py --split test     # first
    python scripts/evaluate.py --split test

For every (space, class): match the detections to the ground truth (IoU >= 0.5), compute the scores at the
threshold in results/thresholds.json, and average precision (AP).

Saves results/<split>_scores.csv and results/<split>_curves.parquet, with the same columns as the validation files.
"""
import argparse
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["test"])
    args = parser.parse_args()

    detections = pd.read_parquet(RESULTS_DIR / f"detections_{args.split}.parquet")
    thresholds = json.loads((RESULTS_DIR / "thresholds.json").read_text())
    images, annotations = set(json.loads(SPLIT_FILE.read_text())[args.split]), map_labels(load_dataset())
    missing = len(images - set(detections["image_path"]))
    if missing:
        print(f"warning: {missing} {args.split} images have no detections yet; scoring only the {len(images) - missing} done")
        images &= set(detections["image_path"])
    gt = ground_truth_boxes(annotations[annotations["image_path"].isin(images)])

    scores, curves = [], []
    for (space, cls), dets in detections.groupby(["space", "cls"]):
        cls_gt = gt[gt["cls"] == cls]
        curve = pr_curve(match(dets, cls_gt), len(cls_gt))
        threshold = thresholds[space][cls]
        scores.append(dict(space=space, cls=cls, ground_truth=len(cls_gt), threshold=threshold,
                           ap=average_precision(curve), **scores_at(curve, threshold, len(cls_gt))))
        curves.append(curve.assign(space=space, cls=cls))

    scores = pd.DataFrame(scores)
    scores.to_csv(RESULTS_DIR / f"{args.split}_scores.csv", index=False)
    pd.concat(curves, ignore_index=True).to_parquet(RESULTS_DIR / f"{args.split}_curves.parquet")
    print(f"{args.split}: {len(images)} images")
    print(scores.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
