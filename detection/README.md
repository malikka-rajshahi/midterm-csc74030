# Shared Color / Intensity detection components

The two modules below are the common entry points for Color and Intensity. Feature formulas remain in `features/color.py` and `features/intensity.py`; existing `common` preprocessing and ROI/refinement code are unchanged. There is one extraction entry named **`feature_extract.py`**, not duplicated `extract.py`/`feature_extraction.py` implementations.

## Responsibilities

| File / function | Responsibility |
|---|---|
| `feature_extract.py::extract_features()` | Prepare one image, extract selected Color/Intensity features on the same supplied rectangles, keep features and coordinates aligned |
| `feature_matching.py::build_references()` | Prepare references from Training GT feature arrays only |
| `feature_matching.py::calibrate_thresholds()` | Determine each feature/metric's thresholds on separately supplied Validation features and labels |
| `feature_matching.py::match_features()` | Choose nearest reference class, accept with its frozen threshold or label `Background` |
| `feature_matching.py::postprocess_detections()` | Optional same-class NMS on already accepted detections |

Data loading, dataset splits, selecting GT/background validation regions, candidate-window generation, file saving, evaluation and plotting are caller responsibilities. No automatic split/downloading, GT labeling, feature fusion or reference fitting at inference time is performed. Separate functions keep training/calibration/inference/postprocessing distinct even though they share one module.

## Shared extraction

`extract_features(image_bgr, boxes, feature_names, ...)` accepts a nonempty uint8 BGR image. It resizes once if requested, calls existing common denoising once, converts the necessary color spaces once, then crops the same rectangles for each requested feature.

Feature names:

- `rgb_histogram`, `hsv_histogram`, `lab_histogram` delegate to `features/color.py`.
- `intensity_histogram` delegates to the normalized gray histogram.
- `mean`, `median`, `mode`, `variance`, `std`, `trimmed_mean`, `entropy`, `gray_range_fraction`, `entropy_contrast`, `normalized_contrast`, `spatial_entropy_difference` select individual values from `features/intensity.py`.

Default histogram dimensions are RGB48, HSV34, Lab40 and gray256. Each color channel sums1 before concatenation, so a three-channel vector sums3; the gray vector sums1. No hidden global renormalization occurs. Statistics still use exact256-level grayscale values even when the displayed/matched gray histogram uses a different bin count.

Each feature is returned **separately** as an `(N,D)` array in `result['features']`; scalars are `(N,1)`. They are not concatenated into an unscaled fusion vector. `histogram_bins` optionally overrides requested histograms, for example `{'hsv_histogram': (9,4,4), 'intensity_histogram':64}`. Use identical definitions for Training, Validation and inference.

### Coordinates and input roles

- `work_size=(width,height)` selects the common working resolution, e.g. `(1440,930)` for current ROI windows. `None` keeps the input resolution.
- `boxes_space='original'` scales original GT/validation boxes to that working image.
- `boxes_space='working'` accepts S2/FINAL working-coordinate windows without rescaling them again.
- Boxes are half-open xyxy. Float coordinates are rounded outward **after** scaling. Invalid/out-of-image boxes raise an error, rather than silently dropping rows.
- `boxes=None` extracts the whole image; `boxes=[]` returns aligned empty arrays.
- Returned `boxes` are working-coordinate pixel rectangles. `original_boxes` map those sampled rectangles outward to the original image, so they may include rounding context beyond the original floating GT rectangle.

Every region retains all its original rectangular pixels, including background. No foreground mask or black-filled background is used. `work_image_bgr` is the actual prepared image for display/cropping. Input pixels are not modified.

`valid_rows` contains one finite-value mask per feature; no row is silently removed. Spatial entropy difference is undefined for a height-one patch and remains NaN for that feature only. Apply the mask to features, labels and coordinates together before matching/calibrating; do not fill it with a fake zero.

`config` records working size, preprocessing defaults, selected feature names, bins and statistic definitions. It does not encode the role (Training/Validation/query); those roles are controlled externally. If an image is already prepared consistently, set `apply_denoise=False` and avoid resizing it again. Do not accidentally denoise twice.

## Training references

`build_references(training_features, training_labels, strategy=...)` accepts one representation at a time. Training labels are target classes, not Background. Original dataset names such as `roadworks` and `vertical_pannel` are not renamed.

- `mean` (default): one mean feature vector per class. Mean histograms preserve the extractor's channel normalization.
- `median`: one median per class for a **single scalar statistic**. It is not applied binwise to histograms because that could destroy normalization.
- `templates`: keep all Training GT vectors and their supplied order as a template database.

Return `features`, `labels`, `strategy`. Mean/median prototypes use class-name order; templates preserve input order. Matching tie rules therefore remain deterministic. Choosing prototype versus template strategies is a training/validation decision, not an inference-time adjustment.

## Validation threshold calibration

`calibrate_thresholds(ref_features, ref_labels, val_features, val_labels, ...)` uses the same nearest-class-first rule as inference. Train, Validation and query images must be disjoint under the caller's grouping policy; array inputs alone cannot verify image identity. This module never changes dataset membership.

**Default `mode='balanced_accuracy'`:** require target examples for every reference class and explicitly labeled `Background` validation examples. For each class, search observed nearest-class distances plus zero. Select the nonnegative threshold maximizing one-vs-rest `(TPR+TNR)/2`; ties choose the lower threshold. Wrong-nearest-class positives remain false negatives when computing target retention. Other target classes and Background are negatives for the current class.

Do not silently label every unmatched/unannotated region Background: incomplete GT can hide real objects. Background labels must be provided/reviewed by the caller. Per-class diagnostics report target retention, false-positive rate, correctly-nearest-positive count and background acceptance on Validation. These diagnostics are not held-out test performance.

**Explicit `mode='positive_quantile'`:** when only positive validation examples are available, choose a quantile of distances for correctly-nearest positives. This controls retention of that subset; overall target retention can still be lower due to nearest-class mistakes. Without Background samples no background-rejection ability is measured or claimed. Classes with no correctly-nearest positives cannot use this mode.

The result contains `thresholds`, `metric`, feature dimension, mode and per-class diagnostics. Pass its `thresholds` into matching, with the same representation and metric. No threshold is learned from query candidates. Recalibrating or adjusting thresholds is never an implicit part of `match_features()`.

## Distances and decision

| `metric` | Formula |
|---|---|
| `l1` | sum abs(a_i-b_i) |
| `squared_l2` | sum (a_i-b_i)^2, **without a square root** |
| `chi_square_max` | sum (a_i-b_i)^2/max(a_i,b_i); zero/zero contributes0 |
| `absolute` | abs(a-b), for a single statistic |

The first three follow the supplied professor's formulas exactly, including the max denominator. The max variant is not replaced by a sum-denominator chi-square. Histograms must be nonnegative for it. Mean/Variance/etc. may use `absolute`.

For each candidate: select the nearest reference, take its class, compare that distance with that class's Validation threshold, then return the class or `Background`. A class with several templates uses its minimum template distance. Do not fall back to a farther class with a wider threshold. Equal distances follow the first reference.

Output arrays are aligned: `labels`, `accepted`, `nearest_classes`, `distances`, `reference_indices`. Rejected rows retain their nearest-match diagnostics. Distances are not probabilities. `pairwise_distances()` supports small analyses; `match_features()` streams blocks without allocating the full candidate-by-reference matrix. Empty candidates are allowed; nonfinite features or invalid references/thresholds are rejected.

## Optional postprocessing

`postprocess_detections(boxes, matches, iou_threshold=.5)` receives the **full aligned** matcher output. It ignores rejected Background rows, sorts by lower distance first within each class, and suppresses same-class boxes at IoU>=threshold. Different classes are processed separately. Ties use source order. Return `source_indices`, unchanged selected box coordinates, labels, distances and reference indices.

`iou_threshold=None` returns accepted boxes without NMS. There is no Top-K cap, merged-box expansion or cross-feature score mixing. Invoke this only at the appropriate detection-output stage: earlier intensity experiments showed that aggressive ranking/suppression can discard useful compact alternatives. Initial filtering can keep `result['accepted']` without NMS until downstream features are applied.

## Example: one feature through the complete stages

```python
import numpy as np
from detection.feature_extract import extract_features
from detection.feature_matching import (
    build_references, calibrate_thresholds, match_features, postprocess_detections,
)

names = ('hsv_histogram', 'intensity_histogram', 'mean', 'variance', 'entropy')
# The caller loads/group-splits images and supplies rectangles and labels.
train = extract_features(train_bgr, train_gt_xyxy, names,
                         work_size=(1440,930), boxes_space='original')
validation = extract_features(val_bgr, val_xyxy, names,
                              work_size=(1440,930), boxes_space='original')
candidates = extract_features(query_bgr, s2_work_boxes, names,
                              work_size=(1440,930), boxes_space='working')

name = 'hsv_histogram'  # choose ONE representation per calibration/matching call
metric = 'chi_square_max'
references = build_references(train['features'][name], train_labels, strategy='mean')
calibration = calibrate_thresholds(
    references['features'], references['labels'],
    validation['features'][name], val_labels,  # includes reviewed Background labels
    metric=metric,
)
matched = match_features(references['features'], references['labels'],
                         candidates['features'][name], calibration['thresholds'], metric=metric)
# Optional final box selection; omit this while keeping initial candidate alternatives.
output = postprocess_detections(candidates['boxes'], matched)
```

Call extraction per image, then concatenate aligned arrays externally for multi-image Training/Validation sets. The example does not create a dataset split or a Background annotation policy. To test Mean/Variance/Std/Entropy, choose that single feature and use `metric='absolute'`; scalar `strategy='median'` is available.

Color and Intensity remain separate tests. Feature units, binning, dimensions, preprocessing and distance formulas must match their references/thresholds. Do not reuse an L1 threshold for squared L2 or add raw color and intensity distances. Any future fusion needs separately fitted scaling and validation; no fusion is implemented.

No dataset rerun, new trained thresholds or detector accuracy is claimed by this structural refactor. Existing feature formulas, common code, historical analyses and saved results are preserved. Only these shared detection interfaces and documentation are updated.
