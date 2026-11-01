# Feature matching

`feature_matching.py` is the shared matching layer for `features/color.py` and `features/intensity.py`. It receives already extracted reference/candidate arrays, selects the nearest reference class, and applies a supplied validation threshold. It performs no image preprocessing, extraction, reference/prototype fitting, validation split, threshold tuning, window generation, NMS, disk I/O or feature fusion.

## Inputs

- Reference features from **training GT**: `(R,D)` templates or externally computed prototypes; one label per reference. Several references may share a class.
- Candidate features from **S2 sliding windows**: `(N,D)`, aligned with externally maintained candidate boxes. This module does not enforce geometry or choose a window stage.
- Frozen thresholds determined on an independent **validation set**, for the same feature definition and distance metric. Use a shared scalar threshold or `{class: threshold}`. Do not derive thresholds from query windows.

Each row is one window/template, not a whole scene made by concatenating all sliding windows. Each call evaluates one feature type. RGB/HSV/Lab histogram vectors from `color.py` and grayscale histograms from `intensity.py` are accepted without fixed dimensions, provided references and candidates match. For Mean, Variance, Std or Entropy, `(N,)` means N scalar candidates (converted to `(N,1)`). A single histogram must be passed as `(1,D)`, not `(D,)`.

## Professor's distance formulas

| `metric` | Formula |
|---|---|
| `l1` | sum_i abs(a_i - b_i) |
| `squared_l2` | sum_i (a_i - b_i)^2; **no square root** |
| `chi_square_max` | sum_i (a_i - b_i)^2 / max(a_i,b_i); zero/zero contributes0 |
| `absolute` | abs(a-b), for a single statistic only |

The max-denominator formula is implemented exactly as displayed in the supplied slide. It is not silently replaced with a sum-denominator chi-square variant. Histogram normalization is handled by the feature extractor: grayscale sums1; `color.py` normalizes each channel separately and concatenates. No hidden global renormalization is applied here.

## Decision rule

1. Compute distances to references using one feature representation/metric.
2. Pick the nearest reference; its label is the nearest class. Equivalently, class distance is the minimum across that class's templates.
3. Compare **that class's** distance with its validation threshold.
4. Return the class if distance<=threshold; otherwise return `Background`.

Do not pick a farther class because its threshold is wider. Unlike the earlier independent-class exploratory filter, one candidate gets at most one output class. Ties follow the first supplied reference. Rejected candidates retain their nearest-class/index/distance diagnostic values.

## Usage

```python
from detection.feature_matching import match_features

# Arrays have already been produced by features/color.py or features/intensity.py.
result = match_features(
    reference_features=train_gt_histograms,  # (R,D)
    reference_labels=train_gt_labels,       # e.g. ['cone','roadworks',...]
    candidate_features=s2_histograms,       # (N,D), same representation/bins
    thresholds=validation_thresholds,       # previously frozen for THIS metric
    metric='chi_square_max',
)
labels = result['labels']                   # target class or 'Background'
accepted_boxes = s2_boxes[result['accepted']]
```

`intensity_features()` returns267 values for storage: slice `[:, :256]` for Histogram, or select a single statistic column using `STATISTIC_NAMES` before matching. Do not pass the raw267-dimensional combination as a histogram or add unscaled scalar/color distances.

For a single statistic:

```python
result = match_features(train_gt_means, train_gt_labels, s2_means,
                        thresholds=mean_validation_thresholds, metric='absolute')
```

`pairwise_distances(candidates, references, metric=...)` returns the full `(N,R)` matrix for small analyses. `match_features` streams candidate/reference blocks and stores only each candidate's best match; it does not allocate the full matrix. Batch sizes are configurable, and temporary memory also depends on feature dimensionality. Empty candidates return empty aligned arrays; empty references, nonfinite features, invalid thresholds and mismatched dimensions are rejected.

## Feature separation and existing results

Test intensity histogram, Mean/Variance/etc., RGB, HSV and Lab separately. Distance magnitudes depend on feature units, binning, dimensions and metric, so do not compare or add them directly. Thresholds calibrated for L1 cannot be reused for squared L2 or the max-denominator formula. Any later fusion must use training/validation-fitted scaling and separate fusion validation; none is implemented here.

The older `intensity_matching` folder remains an explicitly historical development experiment: FINAL refinement windows, independent per-class acceptance, positive-only calibration and optional ranking/NMS. Its historical scores/results are not results of this new nearest-class/Background decision or a new S2 validation run. Its distance arithmetic delegates to this shared module, while experiment-specific fitting/selection stays outside detection.

No new validation thresholds or recognition metrics are asserted by adding this module. Reference extraction and validation-based threshold selection remain the caller's responsibility. Existing common ROI/refinement, preprocessing, feature extraction and old result files are unchanged.
