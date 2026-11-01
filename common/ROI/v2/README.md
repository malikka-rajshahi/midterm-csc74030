# Preserved V2

This is an independent historical version. Use the version-qualified commands below; the parent ROI entry point now uses V3. Only the shared preprocessing import was relocated; generation behavior is preserved.

# ROI Candidate Generation — V2

Generate class-independent regions that contain complete targets with reasonable context. This stage has no Cone/Sign classification, intensity features, color-histogram matching, shape classifier or learned weights. Existing `common/dataset.py` and `common/preprocessing.py` are unchanged. Only official Roadwork training data is used for configuration and development evaluation; the official test split is reserved.

## Files

- `generate_rois.py`: working-resolution preprocessing, one red/orange support mask, anchors, background rejection and public API.
- `sliding_window.py`: local window placement, whole-box mask fractions and bounded overlap merging. It does not scan the complete image on a fixed grid.
- `evaluate_rois.py`: training-size configuration fitting, containment metrics, comparisons and image report.
- `roi_config.json`: required reproducible working size, global window bank, thresholds, merge limits and training provenance.
- `__init__.py`: public imports.
- `results/`: current V2 report; open `index.html`. Previous color/window comparison results used different rules and are replaced to avoid mixing versions.
- The supplied reference-paper PDF is preserved unchanged.

## 1. Working resolution and training-sized windows

Images are resized to **1440 x 930 (width x height)** before common NLM denoising and HSV conversion. GT is independently scaled by the same x/y factors for evaluation. Current Roadwork images are 2880 x 1860, so this preserves their aspect ratio. The generator returns an explicit mapping back to original coordinates.

`--fit-config` reads only official `train/` annotations and image headers, validates coordinates, and removes exact image/class/coordinate duplicates in memory. The fitting population is 2,871 training images and 14,738 canonical source boxes. Neither labels nor raw files are edited; no new random split is created.

For each original source label, split boxes into six area-quantile bins (0/10/25/50/75/90/100%). Take the paired median width and height per bin, multiply both by 1.25 for initial context, round outward and merge duplicate shapes. This produces **17 global width/height templates**, including small targets and larger/group boxes. Their aspect ratios come directly from these shapes. Labels balance the global template bank; they are never passed to per-image inference and do not classify candidates.

At the working resolution, median GT sizes are approximately:

| Original source label | Median width x height |
|---|---:|
| cone | 10.79 x 24.56 |
| vertical_pannel | 9.76 x 33.51 |
| roadworks | 83.99 x 80.05 |

Vistas crop averages (67 x 126 and 117 x 138) are reference information, not fixed Roadwork windows. Configuration fitting and evaluation use the same official training population: results are **in-sample development evidence**, not held-out accuracy.

## 2. One red/orange localization mask

OpenCV's uint8 HSV hue is multiplied by two to obtain degrees. Default chromatic requirements: **S >=0.35, V >=0.15** (normalized 0–1).

| Evidence | Hue in degrees | Role |
|---|---|---|
| Red | [0,15), [345,360) | Seed |
| Orange | [15,45) | Seed |
| Yellow | [45,65) | Evidence only |
| Green | [65,170) | Background evidence only |
| Blue | [190,260) | Background evidence only |
| White | Any hue; S <=0.18, V >=0.60 | Evidence only; allowed inside boxes |

Combine red and orange before any grouping. There are no separate red/orange/union proposal branches. Yellow, green, blue and white do not independently seed candidates. Hue intervals are exploratory, not perfect semantic vegetation/road masks. Cyan outside these bands is unassigned.

Group the single support mask using 2/4/8-pixel blocks with 8-connectivity. A block needs at least two support pixels and 15% support, using the true area at image edges. Connected regions need at least four support pixels. Tighten each **anchor** to its supporting red/orange pixels and remove duplicate anchors. This tightening does not crop away colors inside final ROIs.

## 3. Anchored windows, expansion and tightening

For every anchor, generate its contextual box (25% extra width/height per side) and up to three compatible training-derived templates. Try centered placement and 15%-of-window offsets left/right/up/down; every placement must still contain its anchor and remain in the image.

Compatible templates must fit the contextual anchor, be at most three times its width/height, and at most four times its area. This prevents a small red speck from triggering a huge template. The contextual anchor box is still available when no template fits. A large supported group may generate a large ROI; one ROI need not mean one target.

Reject candidate boxes exceeding **25% of image area**. No candidate is generated from GT at inference, and there is no top-K output cap. Final boxes remain rectangles containing all original pixel colors, including white stripes, bases and background. This is not pixel segmentation or a promise to contain every object.

## 4. Whole-box background filtering

Default `green_blue_50` rejects an entire ROI if:

`pixels classified green OR blue / all ROI pixels > 0.50`

Green/blue bands are disjoint, so their integral counts give the union. Equality is retained. No background pixels are cut out of the box. Alternatives are `green_40` (green alone >40%) and `none`. All V2 comparison variants use the same red/orange seed logic, unlike the previous V1 color study.

A target near trees/grass can still be rejected; desaturated vegetation can evade the HSV mask. White is not required and is not part of rejection. Recheck the background gate after every proposed merge.

## 5. Bounded overlap merging

Overlap is `intersection / min(area_A, area_B)`. At overlap >=0.50, try the smallest enclosing rectangle; full containment keeps the larger box **when the safety limits below permit the merge**.

Visit original candidates largest-first. Each output cluster has one fixed original seed, never another merging pass. Merge eligibility must overlap that original seed, preventing transitive overlap chains from spreading across the scene. Limits:

- Candidate areas must be within a factor of **4**. Much smaller nested candidates remain, preserving compact alternatives alongside group boxes.
- Final union area <= **1.5 x original seed area**.
- Union width/height each <= **1.5 x original seed width/height**.
- Union area <= **25% of image area**, and union must pass the background gate.

If a union fails, keep the original candidate available; do not delete it. Remove exact duplicate boxes after merging and combine anchor provenance. Some overlap intentionally remains when limits prevent merging. This is geometric consolidation, not object-level NMS.

**Why the scale limit matters:** a preliminary implementation bounded outward growth but allowed every large ROI to swallow tiny enclosed ROIs. It reduced counts while producing enormous conditional area ratios. The scale-compatibility limit applies to containment too; it implements the V2 instruction to retain originals when merging would undermine size control.

## 6. API and coordinates

```python
import cv2
from common.ROI.v2 import generate_proposals

image = cv2.imread('/absolute/path/to/image.jpg')  # uint8 BGR
result = generate_proposals(image)

for box, evidence in zip(result['boxes'], result['evidence']):
    x0, y0, x1, y1 = box
    roi = image[y0:y1, x0:x1]  # original-resolution pixels, including background
    print(evidence['color_ratios'])
```

- `boxes`: int32 (N,4), half-open xyxy in original coordinates, outward-rounded.
- `work_boxes`: aligned int32 xyxy in working coordinates; evaluation uses these.
- `work_image_bgr`: actual resized and denoised input used for generation.
- `image_bgr`: unmodified original-resolution input copy, aligned with `boxes`. Unlike V1, this is not the denoised working image.
- `anchors`: work-coordinate tightened red/orange anchors.
- `evidence`: per-box work-resolution color/support ratios, anchor IDs and image-area fraction. Ratios are not class probabilities or true masks.
- `counts`: anchor, generated, rejected, pre-merge and final counts.
- `runtime_seconds`: resize through generation/filtering/merge/evidence time, excluding image-file I/O and visualization.
- `color_masks`: work-resolution estimated masks only if `return_masks=True`.

Empty results have `boxes.shape == work_boxes.shape == (0,4)` and empty evidence. Input arrays are not modified. `background_mode='green_40'` or `'none'` runs the ablations; `merge=False` disables merging. Supply a config dictionary to override settings consistently. Changing work size requires refitting the window bank.

No classifier is included. Later intensity/color features should explicitly choose original-coordinate crops or working-coordinate crops and keep that choice consistent during fitting and testing.

## 7. Evaluation and reproduction

From `/Users/yingying/Documents/midterm-csc74030`:

```sh
# Reuse the saved, training-derived configuration:
python -m common.ROI.v2.evaluate_rois --limit 12

# Refit global templates from official training GT, then evaluate:
python -m common.ROI.v2.evaluate_rois --fit-config --limit 12
```

`--limit 0` uses all official training images. `--output` changes the report directory, `--config` selects a configuration, and `--no-denoise` is an explicit preprocessing ablation. Dependencies: numpy, OpenCV, pandas, pyarrow, Pillow and matplotlib. Old V1 comparison flags/API are intentionally removed.

For each GT, compute intersection / GT area against all candidates:

- **GT coverage >=95%:** at least one ROI covers 95% of the GT rectangle.
- **GT coverage =100%:** at least one ROI contains all four GT boundaries; strict coordinate comparisons, not rounded score equality.
- **IoU >=0.5:** historical comparison only, not the V2 objective.
- **ROI/GT area ratio:** smallest-area eligible ROI for each GT, separately at 95% and 100%. Uncovered GT has no ratio (blank/NaN); report median and 90th percentile where applicable.
- **Compact coverage:** additional diagnostic, GT >=95% with eligible ROI/GT area <=4. This discourages treating giant boxes as success.
- **Mean ROI count and runtime:** all candidates, with no display/output cap. Timings are local development measurements, not a benchmark.

Each GT is independent; one ROI can cover multiple GTs and a GT can appear in several ROIs. Supplied labels can be inaccurate or group-level, so uncovered GT or unmatched proposals are not verified classification errors.

Outputs: `comparison.csv`, `by_label.csv`, `per_gt.csv`, `per_image.csv`, `diagnosis.csv`, `proposals.jsonl` (both coordinate systems), `summary.json` (full config/provenance), `summary.png`, per-image figures and `index.html`. `compact_roi_95/100` in the per-GT table indexes the saved work boxes. Diagnosis records whether a missed GT has little red/orange support; these are observations, not verified causal labels.

The deterministic camera-balanced 12-image sample contains 7 cone, 3 roadworks and 63 vertical_pannel source boxes. Images/GT are selected before proposals, seed 42. Global GT distribution is used only to fit the bank; image-specific GT is used only after generation. The official test image pixels and test labels are never used to fit or evaluate.

## Executed V2 development results

| Variant | Mean ROIs/image | GT >=95% | Strict GT 100% | Median compact area ratio (>=95%) |
|---|---:|---:|---:|---:|
| No background filter, merged | 71.25 | 32/73 | 32/73 | 3.32 |
| Green >40%, merged | 71.25 | 32/73 | 32/73 | 3.32 |
| Green-or-blue >50%, unmerged | 498.50 | 31/73 | 30/73 | 1.60 |
| Green-or-blue >50%, merged (default) | 71.25 | 32/73 | 32/73 | 3.32 |
| Green-or-blue >50%, S >=0.25 | 88.33 | 44/73 | 43/73 | 4.06 |

Default runtime averages approximately 0.31 s/image on this local run, including preprocessing but excluding file reads/plots. Default compact ratio p90 is 15.01; some returned regions are still too large. Full containment and >=95% happen to match at the aggregate level for the default; the metrics are implemented separately.

**Merging trades compactness for a smaller pool.** It removes 85.7% of the unmerged candidate pool and adds two strictly contained GTs, but makes the best eligible boxes larger. Historical IoU>=0.5 matches decrease from 31/73 unmerged to 13/73 merged. This does not establish a universally better detector.

**The V2 background filter has no demonstrated effect here.** Neither green gate rejects a candidate in these 12 images; output equals the no-filter variant. Earlier V1 filtering results used different proposal rules and cannot be reused as evidence.

**Relaxing saturation improves containment but introduces oversized context.** S >=0.25 adds covered GTs, but its compact area-ratio p90 rises to 281.12. Only 21/73 GTs meet coverage >=95% with ratio <=4, versus 20/73 for the default. Keep S >=0.35 as a provisional default, not a proven optimum. The saturation ablation applies to all chromatic evidence masks.

**Main remaining misses:** default strict containment is cone 7/7, roadworks 0/3 and vertical_pannel 25/63. Of the 41 GTs missed at >=95%, 35 have fewer than four qualifying red/orange pixels inside the GT rectangle; six have support but no containing proposal. Inspect low-color/tiny targets, group annotations and window placement next. These are diagnostic observations, not verified causes. See `results/diagnosis.csv`, individual figures and `results/analysis.md`.

The small, imbalanced in-sample study is not sufficient to freeze thresholds or claim final dataset accuracy. Intensity/color classification remains outside ROI.
