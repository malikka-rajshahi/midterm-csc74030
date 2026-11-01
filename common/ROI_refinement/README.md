# Final ROI refinement: S2 + S1 scale-boundary fallback

**Selected configuration: `FINAL` — S2 as the main window bank, plus a small S1 size-boundary fallback.** Keep rectangular windows entirely inside the frozen V4 `green_blue_50` coarse ROIs. No S3 perspective constraint is applied. This is the final candidate-generation configuration for the next feature stage, not a final object detector.

## How the combination works

S2 is already a subset of S1. Taking the union of every S1 and S2 window would simply restore S1's large search pool. Instead:

1. Keep all 19 S2 window shapes and the existing half-window stride.
2. Retain the original globally fitted lower width/height bounds and aspect-ratio bounds.
3. Round only each label-population's upper width/height limits **up to the next power-of-two scale**, matching the discrete S1 window bank.
4. Add S1 shapes passing these boundary-aligned limits for any source population, then remove exact duplicate shapes/windows.

This adds **128x512, 256x512 and 512x512** work-resolution shapes, for a total of22. The rule is global and deterministic. It does not look at an image's GT to decide where or whether to add a window. An extra shape is used only when it fits completely within a coarse ROI. These extra shapes never create a whole-image search.

The original S2 bounds come from the canonical official training GT distribution: per source label, 2nd/98th width, height and width/height percentiles, expanded by0.8/1.25. Original labels are `cone`, `roadworks`, `vertical_pannel`; they balance the prior rather than classify runtime windows. The new rule addresses a quantile upper bound falling between two tested grid scales. It does not reintroduce S1's tiny/flat shapes rejected by S2.

Window positions are unchanged: stride is0.5 of each window dimension, minimum one pixel, with right/bottom aligned positions added. Only exact duplicates are removed. Window overlap is intentional because these are alternatives for later feature scoring; merging them would undo refinement.

## Saved final results

Same 12 training images,73 supplied canonical GT boxes. The frozen V4 input already covers36 GTs at >=95%; refinement cannot exceed this ceiling without expanding the coarse search region.

| Method | Mean windows/image | GT >=95% | Strict100% | Retained among36 V4-covered GT |
|---|---:|---:|---:|---:|
| Historical S1 | 148,582 | 34/73 | 32/73 | 34/36 (94.4%) |
| Historical S2 | 30,158 | 33/73 | 31/73 | 33/36 (91.7%) |
| **FINAL: S2 + S1 boundary fallback** | **30,159** | **34/73** | **32/73** | **34/36 (94.4%)** |

On this sample, the fallback adds12 windows in total, recovers the one GT lost by S2 relative to S1, and matches S1's aggregate coverage with about79.7% fewer windows. It is not the union of all S1/S2 windows. The remaining two coarse-covered GTs are still not >=95% covered by local windows.

Median eligible local-window/GT area ratio is3.26, compared with271.72 for the smallest eligible coarse ROI on **the same covered GT population**. This evaluates whether an appropriate compact patch exists, not whether a feature matcher will select it. Figure rectangles are GT-selected diagnostic windows (oracle), never runtime predictions. Heat maps show all window-center density.

This rule was developed and checked on the training population. The12-image sample is small, imbalanced and in-sample; the observed recovery does not establish held-out recall or global optimality. No official test data was used.

## Files and results

- `generate_windows.py`: `FINAL` generation; historical S1/S2/S3 methods remain explicitly available.
- `evaluate.py`: fitting, final or historical evaluation and report generation.
- `config.json`: unchanged S1/S2/S3 priors plus the selected FINAL rule/shape list and provenance.
- `result_official/`: **final S1+S2 refinement output**, windows, metrics and figures.
- `results/`: **preserved original S1/S2/S3 experiments**, not overwritten by the final run.

[Final report](result_official/index.html) · [Original experiments](results/index.html)

Existing V4 results, official coarse results, shared preprocessing/dataset code and raw data are unchanged. This is an independent module under `common/ROI_refinement`.

## Run

From `/Users/yingying/Documents/midterm-csc74030`:

```sh
# FINAL; default output is result_official:
python -m common.ROI_refinement.evaluate
# Explicitly refit all global priors from official training GT:
python -m common.ROI_refinement.evaluate --fit-config
# Historical comparison; default output is results:
python -m common.ROI_refinement.evaluate --experiments
```

Inputs/options: `--data`, `--coarse`, `--config`, `--output`. Default coarse input is `common/ROI/v4/result_official`. Do not point at V4's mixed-experiment `results`. Work-size and annotation provenance are checked. Number of images follows the frozen coarse JSONL; use a separately saved larger V4 run to expand the study without replacing historical outputs.

```python
from common.ROI_refinement import generate_windows

result = generate_windows(coarse_work_boxes)  # FINAL is the default
windows = result['boxes']
# Historical methods: generate_windows(coarse_work_boxes, method='S1'/'S2'/'S3')
```

Input/output boxes are half-open int32 xyxy at1440x930 working resolution, not original-image coordinates. Crop a consistently resized work image, or map x by original_width/1440 and y by original_height/930 with outward rounding. Do not crop the unscaled original using working coordinates. Arrays are (0,4) when nothing fits.

All windows lie inside one coarse ROI. S2 is a subset of FINAL; FINAL is a subset of S1, with the same grid. No colors, image-specific GT, intensity features, classifier scores or top-K cap enter window generation. Rectangular patches retain background and need no mask or perspective warp.

Outputs include `summary.csv/json/png`, `by_label.csv`, `per_gt.csv`, `per_image.csv`, `windows/*.npz`, per-image figures, `analysis.md` and `index.html`. All generated windows are stored in NPZ `work_boxes`. Diagnostic window indices refer to those saved arrays. Timing includes local window generation/filtering/dedup, excluding V4 generation, fitting, file I/O, plots and feature extraction.

## Remaining work

The candidate pool is still around30,000 windows/image and deliberately contains overlap. It is not30,000 recognized objects. Feature extraction/matching has not yet been tested; HSV/Lab, intensity or HOG must consume shared rectangular patches and consistent preprocessing. Determine an appropriate scorer, cost budget and selection stage next.

V4-uncovered targets remain unreachable from strictly contained windows. Some group annotations do not match single-object size priors; source labels remain unverified. Inspect the two refinement losses and evaluate on a larger reviewed training subset before freezing settings. S3's global perspective gate remains historical only: it lost additional covered targets in the first study.
