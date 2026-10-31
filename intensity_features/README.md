# Intensity — whole-object grayscale features

No sliding windows, NMS or IoU matching. Each object produces 12 feature methods: Histogram, Mean, Median, Mode, Variance, Std/RMS contrast, Trimmed mean, Entropy, Gray-range fraction, Entropy × Contrast, Normalized contrast, Spatial entropy difference. The last three are engineered additions, not slide formulas. Formulas and conventions are in 01/02.

## Data and task

Read only `vistas_templates/templates.csv`, `crops/` and `masks/`. RGB is converted to 8-bit grayscale first. `mask` uses all supplied object-mask pixels; `crop` uses all original rectangular crop pixels, including background. No mask-complement classifier or matched pixel subsampling is used.

The task is recognition of supplied cone/sign object ROIs: cone versus sign, reported separately by class. Opposite-class objects are negatives. This does not measure rejection of pure background, object localization or whole-scene detection. Supplied masks define a precise computation region relative to annotation; masks are eroded and are not guaranteed perfect semantic truth.

Reference: 183 cone + 36 sign. Calibration: 60 cone + 11 sign. Held out: 77 cone + 35 sign. Source images are disjoint. Only template-role data supplies references and calibration; role=validation is independent evaluation.

Clean mask-only references are fixed. Thresholds and one winner per class are selected on full calibration crops using source-balanced accuracy, then frozen before held-out mask/crop evaluation. Threshold ties prefer smaller absolute threshold; feature ties use declared feature order. Histogram distance follows the slide's max-denominator formula; scalar distance is absolute difference. Per-class decisions can both accept or reject; scores are not probabilities.

## Four notebooks

1. `01_Feature_Comparison.ipynb`: formulas, every object's feature values, class summaries and histogram distributions.
2. `02_Validation_Run.ipynb`: all 12 calibration/held-out results, frozen best method per class, mask-free ROI inference helper.
3. `03_Validation_Summary.ipynb`: readable comparison and selected winners.
4. `04_Template_Diagnostics.ipynb`: paired background effects, input fractions, feature changes and exclusions.

Original filenames are retained for continuity. Each notebook includes reusable analysis functions and can run independently. Dependencies: numpy, pandas, pillow, matplotlib and Jupyter. Local/Colab PROJECT discovery is included.

## Results and inference

Output: `intensity_results/vistas_global/`, with separate `cone/` and `sign/` folders. `object_features.csv` stores all 12 feature values for every object/view; histogram vectors are also expanded into 256 columns in per-class `histograms.csv`. `statistics_summary.csv` contains scalar descriptive summaries. `all_method_scores.csv` gives every method's calibration/held-out metrics. `winners.json` stores the frozen selection, and `model.json` stores reference vectors and thresholds.

`predict_crop(path, cls)` consumes an object ROI, possibly containing background, without a mask. No NYC test images are read automatically. Input fingerprints detect changes to the manifest and crop/mask contents; set RUN_AGAIN=True in 02 after editing analysis code.

Spatial entropy retains mask positions and returns NaN if either vertical half contains no selected pixels. Invalid values are excluded and counts reported. Global entropy/histogram use 256 levels. Std uses population variance; trimming removes 10% per tail; mode ties choose the lowest gray level; fixed gray range is inclusive [64,192].

Selecting a method on held-out results would bias the evaluation. Crop background is only a preliminary robustness check, not a simulation of arbitrary background or a final scene test. More background does not necessarily lower accuracy; effects are measured rather than assumed.

A Chinese report covering all 12 methods and selected winners is saved at `intensity_results/vistas_global/RESULTS_中文.md`.
