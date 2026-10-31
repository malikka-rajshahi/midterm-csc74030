# Experiment A — Positive-group Detection Window Coverage

## Scope and interpretation

Only validation images in the original **cone_only**, **sign_only** and **both** groups are included (40, 40 and 16 images). Negative images and their annotations are excluded. Test is not read. Groups come from original `images.csv`, not corrected labels. `both` means individual cone and sign annotation boxes in those images, **not a single box enclosing both objects**. Results are annotation-weighted, not the fraction of images where both classes are successfully covered.

There are 228 valid boxes: 78 cone_only, 56 sign_only, 94 both. One zero-width sign box is excluded and recorded, without editing the dataset. All original duplicate annotations are retained. These annotations are uncorrected model predictions; coverage is a geometric reference, not detector accuracy or certified GT quality.

## Box size reference (pixels)

Width and height quantiles are calculated independently; the median width/height pair is a descriptive reference, not necessarily an actual box or the optimal window.

| Positive group | Valid boxes | Median width × height | Width P25–P75 | Height P25–P75 | Median height/width |
|---|---:|---:|---:|---:|---:|
| cone_only | 78 | 9 × 18 | 6–15 | 12.25–32 | 2.00 |
| sign_only | 56 | 24 × 29 | 12.75–46.5 | 13.75–58.25 | 1.06 |
| both | 94 | 8.5 × 19 | 6–19.75 | 14–31 | 2.07 |

## Coverage reference

12 geometries: widths 8, 16, 24, 32, 48, 64; width:height 1:1 and 1:2 (8×8 through 64×128). Images stay at original resolution. Half stride = (width/2, height/2); quarter stride = (width/4, height/4). Windows are fully inside the image, including appended right/bottom boundary positions. No padding or image pyramid.

A box is covered when any candidate reaches IoU >= 0.5 (primary). IoU measures overlap relative to the combined area of the two boxes. Coverage at 0.3 and 0.7 is also saved. Scanning cost is calculated **only on images of the corresponding group**, including images with no valid annotations. No class-specific window choice may depend on the unknown label at inference; these separate statistics diagnose the common window bank.

| Group | All 12, half stride | All 12, quarter stride | Windows/image, half | Windows/image, quarter |
|---|---:|---:|---:|---:|
| cone_only | 80.8% | 89.7% | 91,561 | 362,529 |
| sign_only | 76.8% | 87.5% | 93,578 | 370,566 |
| both | 78.7% | 88.3% | 96,267 | 381,283 |

## Single-window reference

Best means highest validation annotation coverage at IoU >= 0.5 among the 12 geometries and two stride policies; ties favor fewer windows. This is not the best detector and these values are not frozen final choices.

| Group | Best single window (width × height) | Stride (x, y) | Coverage @0.5 |
|---|---:|---:|---:|
| cone_only | 8 × 16 | (2, 4) | 42.3% |
| sign_only | 16 × 16 | (4, 4) | 26.8% |
| both | 8 × 16 | (2, 4) | 39.4% |

Use the full 12-window bank with half stride as the lower-cost starting reference. Quarter stride is the higher-coverage reference, at roughly four times the scan cost. A single window is insufficient; compare feature computation cost and validation detection accuracy before selecting the final configuration. For `both`, keep the multi-size bank because cones and signs vary independently.

## Implementation and outputs

`sliding_windows.py` is the reusable feature-independent generator. `experiment_a.py` first computes box statistics, then evaluates the two predeclared stride policies. Exact best grid IoU is found by maximizing overlap separately along x and y, without materializing millions of 2D windows. The method was checked against exhaustive scans on 200 seeded random cases.

Local histogram neighborhoods/cells belong separately inside each future feature extractor. This experiment does not run Intensity, HSV, Lab, HOG, PHOG, matching or NMS, and does not correct or generate GT annotations.

Run from repository root (standard library only):

```sh
python3 window_experiment_a/experiment_a.py
python3 window_experiment_a/make_report.py
```

Outputs in `results/`: `box_statistics.csv` (three groups), `coverage.csv` (12 sizes plus combined bank, each stride and group), `per_annotation.csv` (group, object class, best combined-bank IoU and coordinates), `invalid_annotations.csv`, `summary.json`, `reference_values.csv`. CSV annotation row indices are zero-based; coordinates are half-open xyxy.
