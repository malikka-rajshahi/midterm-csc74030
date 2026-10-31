# Testing color.py and other features on the same ROIs

## Purpose and current status

This guide explains how to reuse the existing ROI generation and A/B/C/D experiment framework for a different feature extractor. It does not add a new feature experiment or change the saved results.

The current code has **no `--feature` option**. `run_experiments.py` is a frozen **grayscale Variance** experiment. Running it with a different output directory does not switch the feature.

No `color.py` was found in the Downloads project when this guide was written. Its actual functions, input color space and output shape must be inspected when the file becomes available. Names used in the integration example below describe a proposed interface, not an existing color.py API.

## Which file does what?

| File | Responsibility | Reuse for another feature? |
|---|---|---|
| `generate_rois.py` | Candidate windows, deterministic background sampling, source-object ROIs and optional RGB crop export | Yes: keep the same windows, seed and ROI identities |
| `run_experiments.py` | A/B/C/D prescreens, Variance extraction, frozen references/thresholds, scoring, ranking, NMS, matching and reports | Reuse the pipeline, but replace the feature-specific pieces in a separate feature runner |
| `README.md` | Results and interpretation of the existing frozen Variance experiment | Preserve; do not present it as a new feature's results |

The main integration points in `run_experiments.py` are:

| Existing component | What to do for a new feature |
|---|---|
| `variance_in_boxes(gray, boxes)` | Replace with extraction of the chosen feature from each ROI |
| `REFERENCES` | Fit new cone/sign reference vectors with the same extractor and preprocessing |
| `THRESHOLDS` | Recalibrate class thresholds on development calibration data |
| `predict(...)` | Replace the scalar distance calculation, class-margin calculation and ranking score consistently |
| `color_ratios(...)`, `prescreen(...)` | Keep fixed while comparing downstream features |
| `nms(...)`, `match(...)` | Keep settings and evaluation logic fixed |
| `scene_report(...)`, `write_report(...)` | Label the actual feature and use its measured results; replace Variance-specific prose and conclusions |

## Two different tests

### 1. ROI-level feature comparison

Question: with an object region already supplied, how well does the feature distinguish cone from sign, and how does it behave on background candidates?

From the project root, export the existing deterministic RGB crops if needed:

```sh
cd /Users/yingying/Downloads/midterm-csc74030-main
python ROI/generate_rois.py --output ROI/roi_images
```

This produces:

```text
ROI/roi_images/
  background_candidate/   # 1,920 unverified background candidate crops
  cone/                   # 152 crops from supplied cone annotations
  sign/                   # 76 crops from supplied sign annotations
```

These source-object crops are annotation-sized rectangles and may already contain background. They are not Vistas mask-only samples. Background candidates include random scale-balanced and deliberately color-hard samples; they are not certified background.

Use the same crop identities for every feature. Do not regenerate a more favorable hard-color sample set for each extractor. The sampler always uses the original broad HSV rules, including when evaluating D or a new downstream feature.

The exporter stores RGB images, but it does not write a provenance manifest. For grouped evaluation, retain the `image`, `roi_id`, `label`, `sampling` and `box` fields returned by `roi_records(...)` in memory or in the separate feature experiment's own outputs. A source-object filename `obj_0084.png` maps to original annotation row 84; do not infer its source image from a random crop-level split.

ROI-level results do **not** establish localization performance. A recognition score on supplied object boxes must not be called full-scene detection accuracy.

### 2. Full-scene feature comparison

Question: using the same independently generated candidate boxes, does the feature improve actual box selection and detection?

Keep the candidate bank and chosen prescreen fixed. For each candidate:

```text
Original RGB image
    → same multiscale candidate box
    → fixed A, B, C or D prescreen
    → extract the selected feature from the complete candidate ROI
    → new feature-specific references and calibrated class thresholds
    → fixed ranking policy, top-1000, NMS and max-100/class
    → same one-to-one IoU matching and same-scene image report
```

Ground-truth/source annotation boxes are used for evaluation only, never to generate the scene candidate boxes. Masks are not available or required at inference.

To isolate the effect of the feature, a useful development table is:

| Fixed prescreen | Variance | color.py feature | Mean | Entropy | Other feature |
|---|---|---|---|---|---|
| C, for example | Existing baseline | New result | New result | New result | New result |

Choose and declare the prescreen before examining the new comparison. C has the highest provisional score in the existing run, but its labels are noisy and its broad hue rules admit yellow. If the intended target definition requires D, fix D instead and state that choice. Do not silently change both the prescreen and the feature and attribute the gain to the feature alone.

A full A/B/C/D × feature comparison is also possible. It is a separate, larger experiment: every feature needs its own fitted references and calibrated thresholds, with the shared settings below held fixed.

## Using color.py

First check what color.py actually computes:

1. Does it expect RGB, BGR, HSV, Lab or grayscale?
2. Does it return a scalar, histogram, vector, probability or distance?
3. Is it a feature extractor, an initial filter, or an entire detector?
4. Does it need fitting, resizing, normalization, a mask or trained weights?

If it is an RGB/color feature, pass the original RGB crop. Do not convert it to grayscale first. If it expects BGR, convert explicitly once; do not silently treat RGB as BGR. The current project uses PIL RGB and PIL HSV, not OpenCV HSV units. PIL hue here is `H * 360 / 256`; an OpenCV-style 0–179 hue threshold cannot be copied unchanged.

If color.py is only another HSV rejection rule, integrate it as a **prescreen experiment**, not as a replacement for Variance. Otherwise the comparison would describe a different task from a downstream feature test.

### Proposed extractor contract

The following is an integration pattern to implement in a separate feature runner. It is **not** a command or API already supported by the current files:

```python
import numpy as np
from PIL import Image

# Implement this function using the actual color.py API when available.
# Input: uint8 RGB crop, shape (height, width, 3).
# Output: finite vector with fixed length; a scalar becomes a length-1 vector.
def extract_feature(rgb_roi):
    raise NotImplementedError("Connect the actual feature extractor here")


def features_for_boxes(rgb, boxes):
    rows = []
    for x0, y0, x1, y1 in boxes:
        crop = rgb[y0:y1, x0:x1]
        vector = np.asarray(extract_feature(crop), dtype=float).reshape(-1)
        if vector.size == 0 or not np.isfinite(vector).all():
            raise ValueError("Empty or undefined feature; report the exclusion")
        rows.append(vector)
    return np.stack(rows)  # (number_of_boxes, feature_dimensions)


# Example replacement for a grayscale Mean feature, not a color.py API:
def mean_feature(rgb_roi):
    gray = np.array(Image.fromarray(rgb_roi).convert("L"))
    return np.array([gray.mean()], dtype=float)
```

The crop loop is a correctness-first example and can be slow over the full candidate bank. Mean and variance can use integral images; other features may need batching. Verify an optimized implementation against direct ROI extraction on representative small regions before relying on its speed.

For intensity methods, perform the grayscale conversion inside the extractor, then compute the chosen statistic. Keep definitions consistent with references: population variance, the same histogram bins, the same entropy base, fixed trimming and fixed gray-range bounds. Spatial features must preserve 2D positions.

## Fit references and calibrate new thresholds

**Do not reuse the current Variance reference values or ±682.205379 thresholds for a different feature.** Even replacing Variance with Standard Deviation changes the scale and decisions.

For a controlled extension of the existing pipeline:

1. Reuse the original source-separated Vistas reference/calibration/held-out allocation. The current reference portion has 183 cone and 36 sign objects; calibration has 60 cone and 11 sign; held-out has 77 cone and 35 sign. Recover the allocation from `../intensity_features/intensity_results/vistas_global/split_manifest.csv` (relative to ROI), or reproduce its seed-42 source-image split. Do not randomly split individual crops from one source image across roles.
2. Fit each class reference using the new extractor on reference objects only, with equal source-image weighting as in the existing analysis. For a pixel-statistic extractor, use mask pixels only. For a color histogram, count only mask pixels. For a spatial extractor that cannot accept a mask, explicitly define a consistent alternative rather than painting the outside black and counting those artificial pixels.
3. Use complete calibration crops, including their existing background, to select each new feature's class-margin thresholds. If the thresholds are instead calibrated on NYC development candidates, disclose that protocol change and do not describe the same images as an independent validation set.
4. Fix feature preprocessing, references, scaling, metric and thresholds before independent evaluation. Keep all samples from a source image in one split.
5. Evaluate the exact same held-out crops and/or NYC development scene candidates for every feature. The current NYC validation comparison remains development work, not an untouched benchmark.

A compatible replacement for the current binary class scoring is:

```text
own_distance   = distance(feature_vector, own_class_reference)
other_distance = distance(feature_vector, other_class_reference)
margin         = other_distance - own_distance
accept class if margin >= calibrated_threshold_for_this_feature_and_class
rank accepted ROIs by own_distance, ascending
```

Use a declared distance appropriate to the representation. Absolute difference is a natural controlled baseline for one scalar; normalized histogram distance is appropriate for histograms. If vector coordinates have different scales, fit scaling from reference data only. Do not compute scaling from held-out or test features.

This remains cone-versus-sign discrimination. It does not learn a background class. Training a background-aware classifier is another experiment and requires reviewed negatives; unverified `background_candidate` crops must not be treated as certified ground truth without review.

## What stays fixed during a feature comparison?

| Component | Existing setting |
|---|---|
| Scene data | NYC `Test Dataset/validation`, 120 images |
| Candidate widths | 8, 16, 24, 32, 48, 64 |
| Candidate height/width | 1 or 2 |
| Strides | Half the window size; edge-aligned last window |
| ROI sample seed | 42 |
| Hard-color sample selection | Original broad HSV score, unchanged across features |
| Prescreen | One declared A/B/C/D configuration, or the full declared grid |
| Ranking policy | Ascending own-reference distance; original index resolves ties |
| Before NMS | Top 1,000 candidates per class |
| NMS | IoU 0.3 |
| Final output cap | 100 per class |
| Matching | Same class, one-to-one IoU ≥0.5; source duplicates retained |

The feature extractor, fitted reference vectors and calibrated thresholds must change together. The selection criterion and tie rules should be declared before evaluating outcomes.

## Separate outputs and report accurately

Preserve the existing ABCD results and their README. Put new experiments in separate output directories, for example:

```text
ROI/
  feature_results/
    color/
    mean/
    entropy/
```

The current CLI can reproduce **only the existing Variance run**, including to another output location:

```sh
python ROI/run_experiments.py --output ROI/feature_results/variance_baseline
```

There is no working `python color.py --roi ...` command established by this project. A new color/feature runner must connect the actual extractor and fit its own parameters first. Its CLI should be documented when implemented.

Also update the report generator: `write_report(...)` currently includes frozen Variance-specific numeric discussion, and its interpretation is not a generic template for arbitrary features. A new run's report must name the new feature, show its own parameters and measured outcomes, and replace inherited conclusions. Merely changing the output directory or chart title is insufficient.

For each new feature, report:

- ROI-level recognition separately from full-scene localization.
- Cone/sign precision, recall and F1 separately, plus the declared overall criterion.
- Initial-filter object retention and background rejection separately from downstream feature decisions.
- Random and color-hard background candidates separately.
- Excluded/undefined features, inference time and output count.
- The same scene shown with the existing baseline and the new method, with all retained predictions visible.
- Unverified-label limitations and any protocol change, rather than attributing every unmatched box to the model.

## Final-test rule

Do all feature, threshold and color-rule selection on development data. Keep NYC test and the formal dataset under `Downloads/datasets` untouched until the pipeline is frozen. The current `read_development(...)` helper is intentionally fixed to the validation directory; the current scripts are not a loader for the formal dataset. A future final-test adapter must match that dataset's image/annotation format and use frozen settings without fitting or selecting a winner from its scores.
