"""Full detection pipeline: run every image of a split through color-histogram template matching.

    python scripts/run_detection.py --split validation
    python scripts/run_detection.py --split validation --limit 5     # quick check on 5 images
    python scripts/run_detection.py --split nyc                      # the 300 NYC images

Per image: denoise -> multi-scale windows per class (window sizes from the training boxes) -> cone windows are
prescreened by orange fraction -> RGB / HSV / Lab histogram of every window -> D3 distance to the nearest template
of that class -> NMS per class and color space.

No distance threshold is applied: NMS keeps the lowest-distance box first, so filtering the saved boxes by any
threshold later gives exactly what thresholding before NMS would. Thresholds are tuned in validate_features.py.

Saves results/detections_<split>.parquet with columns
image_path, space, cls, x0, y0, x1, y1, distance. Re-running continues where it stopped.

Window sizes come from the HF training boxes. nyc (street-level images, no annotations used: a qualitative check)
has no boxes to measure, so it uses the same multi-scale rule over a fixed width range (NYC_WIDTHS, NYC_ASPECTS),
sliding by NYC_STRIDE_FRACTION of the window width instead of half.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common.dataset import DATASET_DIR, NYC_ASPECTS, NYC_DIR, NYC_STRIDE_FRACTION, NYC_WIDTHS, SPLIT_FILE, load_dataset, map_labels, nyc_images  # noqa: E402
from common.preprocessing import denoise, orange_mask  # noqa: E402
from common.sliding_window import multiscale_windows, prescreen, size_range, window_histograms, window_sizes  # noqa: E402
from detection.postprocessing import nms  # noqa: E402
from detection.template_matching import load_reference, nearest_template_distance  # noqa: E402
from features.color import SPACES  # noqa: E402

RESULTS_DIR = ROOT / "results"
NMS_IOU = 0.3
SAVE_EVERY = 20  # images

# Set in each worker process by init_worker
REFERENCE = SIZES = IMAGE_DIR = STRIDE_FRACTION = None


def init_worker(sizes, image_dir, stride_fraction):
    global REFERENCE, SIZES, IMAGE_DIR, STRIDE_FRACTION
    cv2.setNumThreads(1)  # one image per process; avoid oversubscribing cores
    REFERENCE, SIZES, IMAGE_DIR, STRIDE_FRACTION = load_reference(), sizes, image_dir, stride_fraction


def detect(image_path):
    """All post-NMS candidate boxes of one image, for every color space and class, as a DataFrame."""
    img = denoise(cv2.imread(str(IMAGE_DIR / image_path)))
    boxes = {"cone": prescreen(orange_mask(img), multiscale_windows(img.shape, SIZES["cone"], STRIDE_FRACTION)),
             "sign": multiscale_windows(img.shape, SIZES["sign"], STRIDE_FRACTION)}
    all_boxes = np.concatenate(list(boxes.values()))
    rows = []
    for space, sp in SPACES.items():
        hists = window_histograms(sp["convert"](img), all_boxes, sp["bins"], sp["channel_max"])
        start = 0
        for cls, cls_boxes in boxes.items():
            cls_hists, start = hists[start:start + len(cls_boxes)], start + len(cls_boxes)
            distances = nearest_template_distance(cls_hists, REFERENCE[space][cls])
            keep = nms(cls_boxes, distances, NMS_IOU)
            rows.append(pd.DataFrame({"image_path": image_path, "space": space, "cls": cls,
                                      "x0": cls_boxes[keep, 0], "y0": cls_boxes[keep, 1],
                                      "x1": cls_boxes[keep, 2], "y1": cls_boxes[keep, 3],
                                      "distance": distances[keep]}))
    return image_path, pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test", "nyc"])
    parser.add_argument("--limit", type=int, help="only the first N images (for a quick check)")
    parser.add_argument("--workers", type=int, default=max(1, (__import__("os").cpu_count() or 2) - 2))
    args = parser.parse_args()

    if args.split == "nyc":
        images, image_dir, stride_fraction = nyc_images()[:args.limit], NYC_DIR, NYC_STRIDE_FRACTION
        sizes = {cls: size_range(*NYC_WIDTHS[cls], aspect) for cls, aspect in NYC_ASPECTS.items()}
    else:
        split = json.loads(SPLIT_FILE.read_text())
        images, image_dir, stride_fraction = split[args.split][:args.limit], DATASET_DIR, 0.5
        # HF window sizes always come from the training boxes, whatever split is being run
        train = map_labels(load_dataset())
        train = train[train["image_path"].isin(split["train"])]
        sizes = {cls: window_sizes(np.stack(g["bbox_coords"].to_numpy())) for cls, g in train.groupby("cls")}

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"detections_{args.split}.parquet"
    done_file = RESULTS_DIR / f"detections_{args.split}_done.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    results = [pd.read_parquet(out)] if out.exists() and done else []
    todo = [p for p in images if p not in done]
    print(f"{args.split}: {len(images)} images, {len(images) - len(todo)} already done, {args.workers} workers")

    def save():
        # Write to a temporary file and swap it in, so an interrupted save never leaves a broken file
        tmp = out.with_suffix(".tmp")
        pd.concat(results, ignore_index=True).to_parquet(tmp)
        tmp.replace(out)
        done_file.write_text("\n".join(sorted(done)))

    with ProcessPoolExecutor(args.workers, initializer=init_worker, initargs=(sizes, image_dir, stride_fraction)) as pool:
        for n, (image_path, df) in enumerate(pool.map(detect, todo), 1):
            results.append(df)
            done.add(image_path)
            if n % SAVE_EVERY == 0 or n == len(todo):
                save()
                print(f"{n}/{len(todo)} images", flush=True)
    if results:
        save()
    print(f"saved {out}")


if __name__ == "__main__":
    main()
