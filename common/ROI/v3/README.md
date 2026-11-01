# ROI Candidate Generation V3

V3 is an independent copy of V2 with two requested merge changes. Shared preprocessing, training-sized window bank, red/orange localization, working resolution and evaluation metrics are unchanged. See `../v2/README.md` for the underlying algorithm; that document describes V2's different merge settings.

## Merge changes

1. Intersection / smaller-box area must be **>=0.70**, previously 0.50.
2. The **4x area-scale restriction does not apply to fully nested boxes**. The larger box is retained. For partial overlaps, the 4x restriction still applies.

All other safeguards remain: fixed original seed, no repeated transitive merging pass, union area and dimensions bounded to 1.5x the seed, maximum ROI area 25% of the image, and background-gate recheck. Failed unions retain originals. V3 does not guarantee zero overlapping boxes.

## Independent V3 results

The saved V3 report follows the same per-version evaluation workflow as V2: no-background-filter, Green>40%, Green-or-Blue>50% unmerged, Green-or-Blue>50% merged, and the S>=0.25 ablation. It contains only V3 runs, not V2/V3 side-by-side figures.

The default V3 configuration produces 34.58 ROIs/image and fully contains 31/73 supplied GT boxes on the same 12-image training development sample. Median compact ROI/GT area ratio for strict containment is 19.28; at >=95% coverage the p90 ratio is 169.58. Only 7/73 GTs have >=95% coverage with an eligible area ratio <=4. Large enclosing ROIs replace many small nested alternatives, so fewer ROIs do not establish better quality. See `results/index.html`, `results/analysis.md` and the per-GT tables.

Background filtering rejects zero candidates in this sample. No test data, intensity or classifier is used. These in-sample results do not establish an optimum.

## Run

From the project root:

```sh
python -m common.ROI.v3.evaluate_rois --limit 12
# Refit the same global bank and preserve V3's 0.70 overlap setting:
python -m common.ROI.v3.evaluate_rois --fit-config --limit 12
```

Use `from common.ROI.v3 import generate_proposals`. The parent `from common.ROI import generate_proposals` also selects V3. Results are local to `v3/results`; V1 and V2 are not overwritten. There is no combined cross-version report.

The small, imbalanced in-sample evaluation is not a held-out benchmark. All V2 coordinate, API and conditional compactness definitions remain unchanged.
