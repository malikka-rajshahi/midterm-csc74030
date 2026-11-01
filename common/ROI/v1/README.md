# Preserved V1

This is an independent historical version. Use the version-qualified commands below; the parent ROI entry point now uses V3. Only the shared preprocessing import was relocated; generation behavior is preserved.

# Class-independent ROI proposals

This module generates candidate boxes and color evidence. It makes no cone/sign decisions. Shared `common/dataset.py`, `common/preprocessing.py`, source annotations and the official test split are unchanged.

## Files

- `generate_rois.py`: multi-scale connected color-block proposals and aligned color evidence.
- `sliding_window.py`: fixed-window banks, whole-window color gating and conservative geometric deduplication.
- `evaluate_rois.py`: training-only comparisons, coverage tables and visual diagnosis.
- `__init__.py`: public generator imports.
- `results/`: reproducible window comparison; open `index.html` for both experiment reports. `color_comparison` tests current color/background rules; `window_comparison` preserves the original size/stride ablation, not a backup.
- The supplied reference-paper PDF is retained unchanged.

## What a size means

The original 4/8/16 values are **color block sizes**, not final ROI widths. Neighboring active blocks form a connected region whose box may be much larger. Scales 1, 0.5 and 0.25 further change their original-image footprint. Both tight boxes and 12%-per-side padded boxes are generated.

The new sliding windows have explicit original-image pixel widths:

| Bank | Widths | Height / width |
|---|---|---|
| small | 16, 32, 64 | 0.5, 1, 2, 4 |
| balanced | 16, 32, 64, 128, 256 | 0.5, 1, 2, 4 |
| context | 32, 64, 128, 256, 512, 1024 | 0.5, 1, 2, 4 |

The step is `round(window_dimension * stride_fraction)`, independently in x and y, with a minimum of one pixel. Right/bottom aligned final windows are included. A 0.75 stride scans fewer locations than 0.5 but can reduce localization coverage. Windows larger than the image are skipped. These presets are exploratory, not universal optimal sizes.

Annotations are boxes, not foreground masks. They legitimately include background and sometimes groups of objects. A tight color region may therefore be visually useful yet have poor IoU with the supplied box. Larger windows can address this mismatch but must not replace all small windows: distant objects still need small candidates.

## Color and duplicates

Current defaults use only red/orange/yellow as seeds, with yellow restricted to [45,65) degrees. Blue and green [65,170) are recorded only as evidence. Reject proposals whose green fraction is strictly greater than 0.50; set `max_green_fraction=None` to disable this background gate. A few green pixels inside an ROI are allowed.

Common `denoise()` is applied once per image; `to_hsv()` expects BGR. OpenCV hue is multiplied by 2 for degrees. Red: [0,15) or [345,360); orange: [15,45); yellow: [45,65); blue: [190,260). Chromatic pixels require S >= 0.20 and V >= 0.12. White requires S <= 0.18 and V >= 0.60; it is evidence only, never a standalone seed or a mandatory cone rule.

For connected blocks, at least two colored pixels and 20% color per block activate a cell; 8-connected active cells form regions. Boxes exceeding 90% of image area are rejected. Sliding windows instead require at least two colored pixels and 20% color over the **whole window**, including background, using red/orange/yellow seeds only. The same optional green gate follows. Thus comparisons between these generators change the grouping mechanism as well as size. Within sliding-window comparisons the color gate stays fixed.

Exact boxes are merged. The generator then removes near duplicates at IoU >= 0.90, in stable generation order, without class labels, feature scores or a top-K cap. This is conservative geometric suppression, not learned detection NMS. Different-sized nested boxes can remain deliberately: one might cover an individual object and another its annotated group. Lowering the threshold can reduce more boxes but can also discard useful alternatives. Similar geometry alone cannot identify every duplicate object detection.

Color masks are estimated evidence, not true object masks. Colors can belong to background. Evidence ratios include **all pixels in the retained box**. When boxes are suppressed, their generation sources are attached to the retained representative; its ratios still describe the representative box only.

## Generate candidates for other features

From the project root:

```python
import cv2
from common.ROI.v1 import generate_proposals

image = cv2.imread('/absolute/path/image.jpg')
result = generate_proposals(image, return_masks=True, return_raw=True)
for box, evidence in zip(result['boxes'], result['evidence']):
    x0, y0, x1, y1 = box
    roi_bgr = result['image_bgr'][y0:y1, x0:x1]
    ratios = evidence['color_ratios']
    # Feed the same roi_bgr to color, intensity or other feature extractors.
```

`boxes` is an int32 (N,4) array of half-open original-coordinate xyxy boxes. Empty output is (0,4). `raw_count` is the exact-merged count before near-duplicate suppression. `raw_boxes` is populated only with `return_raw=True`; `color_masks` only with `return_masks=True`. Input pixels are not modified. `dedup_iou=None` reproduces unsuppressed proposals. Other controls include `block_sizes`, `scales`, `padding`, `min_color_fraction` and `max_area_fraction`, `proposal_colors`, `yellow_max` and `max_green_fraction`.

To test another explicit window bank:

```python
from common.ROI.v1.sliding_window import sliding_windows, filter_color_windows, deduplicate_boxes

windows = sliding_windows(image.shape[:2], widths=(32, 64, 128, 256), stride_fraction=0.75)
windows = filter_color_windows(windows, result['color_masks'])
keep, parents = deduplicate_boxes(windows, iou_threshold=0.90)
boxes = windows[keep]
```

Use `common.preprocessing.to_gray(roi_bgr)` for grayscale intensity features. Fit classifiers and feature thresholds outside ROI, with consistent preprocessing. No source labels are supplied to either candidate generator.

## Reproduce the comparison

From `/Users/yingying/Documents/midterm-csc74030`:

```sh
python -m common.ROI.v1.evaluate_rois --compare-colors \
  --data /Users/yingying/Downloads/datasets/Roadwork_Cones_Dataset/roadwork_cones_dataset \
  --limit 12 --output common/ROI/results/color_comparison
```

Dependencies: numpy, OpenCV, pandas, pyarrow, matplotlib. Omit both comparison flags for a single-generator diagnosis. Use `--compare-windows --legacy-colors --output common/ROI/results/window_comparison` to reproduce the saved original blue-inclusive window study; omit `--legacy-colors` to study the current warm-color defaults. `--limit 0` uses all official training images; `--output` selects a separate report directory. `--no-denoise` is a preprocessing ablation.

Only `train/` image paths are opened. Existing train/test membership is preserved. Exact image/class/coordinate annotation duplicates are removed in memory; no annotations are edited or semantically renamed. The camera-balanced sample is deterministic (seed 42). This development sample is small and unbalanced: 7 cone, 3 roadworks and 63 vertical_pannel boxes. The original label spelling is preserved. Results do not establish final dataset accuracy.

Report files:

- `comparison.csv`: mean candidates and independent source-box coverage at IoU 0.3/0.5/0.7; number of baseline-covered boxes lost.
- `coverage_by_group.csv`: coverage by original class and source-box size (sqrt area <32, 32–128, >=128 pixels).
- `per_image.csv`: generated, color-passing and retained counts.
- `per_box.csv`: every source box's best IoU per method.
- `summary.json`: development scope and preliminary selection.
- `summary.png`, `images/`, `index.html`: count/coverage comparison and side-by-side images. All candidate boxes are drawn, with no display cap.

Coverage means at least one proposal overlaps a supplied box at the threshold. It is not classification accuracy, verified detection recall, precision or F1. One proposal may cover multiple labels. Source labels may be inaccurate or incomplete, so unmatched candidates cannot automatically be called false positives.

The report identifies the lowest-count method that retains every box covered by the raw baseline at IoU 0.5 **on this sample**. This does not prove a globally best window size, and does not automatically change production block sizes. Review size/class-specific losses before extending the sweep to a larger training subset. Keep the official test reserved for the frozen final pipeline.

The connected-color-block approach is adapted from Piccioli et al., Section 3.1.2; the multi-scale settings, padding alternatives and fixed-window study are project adaptations.

## Original window-study results and interpretation

The saved `results/window_comparison` uses the previous red/orange/yellow/blue seeds, yellow up to 75 degrees, and **no green rejection**. These results explain why full-image sliding windows were not adopted. They are not the current defaults.

On the 12-image development sample (73 canonical source boxes):

| Method | Mean candidates/image | Coverage at IoU 0.5 | Baseline-covered boxes lost |
|---|---:|---:|---:|
| Original blocks, exact merge only | 1,531.8 | 26/73 (35.6%) | 0 |
| Original blocks, near-duplicate IoU 0.90 | 1,446.2 | 26/73 (35.6%) | 0 |
| Original blocks, near-duplicate IoU 0.80 | 1,329.4 | 26/73 (35.6%) | 0 |
| Medium blocks 8/16/32, dedup 0.90 | 681.1 | 23/73 (31.5%) | 3 |
| Coarse blocks 16/32/64, dedup 0.90 | 300.7 | 14/73 (19.2%) | 12 |
| Small windows, stride 0.50 | 97,008.3 | 31/73 (42.5%) | 6 |
| Balanced windows, stride 0.50 | 98,702.4 | 37/73 (50.7%) | 3 |
| Balanced windows, stride 0.75 | 44,249.1 | 18/73 (24.7%) | 14 |
| Context windows, stride 0.75 | 11,837.3 | 8/73 (11.0%) | 20 |

**Geometric duplicates can be reduced, but duplicate suppression alone is insufficient.** IoU 0.90 removes 5.6% of candidates; 0.80 removes 13.2%, with no lost baseline-covered boxes on this sample. The latter is the preliminary lowest-count coverage-preserving setting, not proof of safety on the entire training set. The generator default remains conservative at 0.90; use `dedup_iou=0.80` explicitly to reproduce that ablation. Distinct nested boxes still serve different target/context scales.

**Making blocks larger reduces counts by discarding useful detail.** Medium and coarse blocks lose 3 and 12 previously covered source boxes respectively. Thus fewer boxes do not imply better ROIs. Larger context windows alone also fail small/distant targets and need suitable positions and aspect ratios.

**Full-image sliding windows expose a weakness in the broad color gate.** A window passes when 20% of its pixels match any supported chromatic band. Yellow vegetation, blue-looking shadows, reflections and other colored background can therefore pass; the gate has no object semantics. This is a weakness of the current gate, not evidence that those backgrounds are valid objects. Larger windows and wider strides reduce the enormous grid, but do not solve background rejection. The balanced 0.50 grid covers more labels overall while losing three that the block baseline covered, and costs roughly 64 times as many candidates. It is unsuitable as the current default.

**Next adjustment:** retain local color-region proposals as position anchors; test nearby window sizes/offsets to include reasonable context rather than scanning the whole scene with this gate. Evaluate coherent local color support and background rejection on a reviewed training subset. Do not simply remove yellow/blue: real work-zone objects use these colors. A color ratio is evidence, not an acceptance decision. Classification stays outside ROI, and no hard cone/sign rule is introduced here.

The overall coverage remains low. Shadowed/weak-color objects, grouping versus individual-box mismatch, background color fragments and imperfect source labels still require diagnosis. This study establishes a count/coverage tradeoff, not a solved detector or a reliable final optimum.

## Current color/background comparison

Same 12 training images, 73 source boxes, common preprocessing and dedup IoU 0.90:

| Rules | Mean candidates/image | Covered source boxes, IoU 0.5 | Lost versus warm-only |
|---|---:|---:|---:|
| Previous blue-inclusive / yellow to 75 degrees | 1,446.2 | 26/73 | — |
| Red/orange/yellow only; yellow to 65 degrees | 886.7 | 27/73 | 0 |
| Warm seeds; reject green >40% | 787.0 | 27/73 | 0 |
| Warm seeds; reject green >50% (current default) | 808.5 | 27/73 | 0 |

Removing blue seeds and narrowing yellow cuts candidates by 38.7% versus the previous deduplicated configuration, with no baseline-covered box lost and one newly covered. Removing seed branches can change which near-duplicate representative survives; this is not proof of improved semantic recognition. Both green gates retain all 27 warm-only covered boxes here. Green >50% cuts another 8.8% of candidates (44.1% relative to the previous configuration). The 40% gate is the lowest-count color setting tested on this sample, but 50% is the conservative default to allow more background context.

**The green gate is a background heuristic, not an object rule.** Real small targets near grass/trees may occupy less than half the annotation box. Therefore green >50% can still reject a valid ROI. Green is measured from estimated HSV membership (S>=0.20, V>=0.12), not a perfect vegetation mask. Shadowed/desaturated vegetation may escape it, and yellow-looking foliage may still trigger warm colors. Boxes with little green may also be pure background. This adjustment does not solve all background rejection.

Only 27/73 supplied boxes are covered overall: the low coverage and annotation/group mismatch remain. Expand the reviewed training subset before declaring an optimum. Official test was not used. To compare green thresholds yourself, call `generate_proposals(image, max_green_fraction=0.4)` or `0.5`; disable with `None`. Green fraction is over every pixel in each candidate box; equality is retained.
