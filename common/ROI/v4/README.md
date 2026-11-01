# Selected pipeline: V4 `green_blue_50`

**V4 `green_blue_50` is our final choice for the current ROI stage.** It prioritizes few non-overlapping regions and complete-target containment; one ROI can include several targets. No Cone/Sign classification, intensity or histogram matching is applied here.

The parent [ROI README](../README.md) is the main specification: it includes the brief V1–V3 change history, exact selected rules, measured results, unresolved issues and reproduction commands.

## Files

- `generate_rois.py`: prepare working image, one red/orange support mask, anchors, training-sized windows, green/blue filtering, grouping and coordinate mapping.
- `sliding_window.py`: anchored placement, integral color fractions and intersecting-group closure.
- `roi_config.json`: working resolution, template bank, thresholds and official-training provenance.
- `evaluate_rois.py`: training-only GT containment/partial-overlap evaluation and the selected output report.
- `results/`: selected pipeline only. Each image has three panels: supplied GT, generated ROIs, and GT coverage status.

## Key rules

Work at 1440x930 with shared denoising/HSV conversion. Red H=[0,15) or [345,360), orange H=[15,45), S>=0.35, V>=0.15. Hue is in degrees, converted from OpenCV H*2. Only the red/orange union seeds anchors; other colors remain inside crops. Blocks are 2/4/8 pixels, with >=2 support pixels, >=15% support and >=4 support pixels per region.

Use 17 globally training-derived templates plus a contextual anchor rectangle with 25% padding per side. Try up to three compatible templates and nearby offsets without cutting the anchor. Templates fit the contextual anchor and stay within 3x each dimension and 4x its area. Reject initial candidates >25% image area.

Reject the whole ROI when green H=[65,170) OR blue H=[190,260) accounts for >50% of ROI pixels with the same S/V thresholds. Keep white; do not require white or cut individual pixels out.

Join every positive-area intersecting group into its smallest enclosing rectangle, repeating if envelopes intersect. Touching edges alone do not merge. Recheck green/blue after grouping; remove failing groups as a whole. Final groups >25% image area are flagged, not split back into overlapping boxes. There are no retained V2/V3 merge thresholds or fixed-seed expansion limits, and no top-K cap.

## Saved results and selection rationale

Same 12 training images, 73 supplied GT boxes: **6.83 ROIs/image**, **36/73 complete >=95% coverage**, **36/73 strict containment**, **2 partial**, **35 no intersection**. No final ROI pairs intersect on the sample. Median/p90 eligible ROI/GT area ratios are **259.68/683.75**. Historical IoU>=0.5 matches are 1/73; tight localization is not this stage's objective.

We select this output because its grouping and containment behavior match the requested ROI purpose. A group can contain several targets, making its area large relative to each individual GT. This is a known tradeoff, not proof that the method is unsuitable. Downstream feature tests must determine whether excess background is harmful. Green/blue filtering itself has not demonstrated a benefit on this sample.

Unresolved: missing qualifying color support, incomplete coverage, non-target colored regions, oversized context groups, uncertain source annotations and lack of held-out validation. These are candidate regions, not recognized objects. The official test split remains unused.

## Run and API

```sh
python -m common.ROI.v4.evaluate_rois --limit 12
python -m common.ROI.v4.evaluate_rois --fit-config --limit 12
```

```python
from common.ROI.v4 import generate_proposals
result = generate_proposals(image_bgr)
```

Original-coordinate `boxes` align with original input pixels; `work_boxes` align with `work_image_bgr`. Both are half-open int32 xyxy arrays. `image_bgr` is an unchanged input copy; `evidence` uses working-image estimated color/support ratios, anchor IDs and the `oversized` flag. `counts` and `runtime_seconds` report generation costs; timing excludes file reads/plotting. `return_masks=True` provides estimated working-resolution masks. No unused experimental flags or tracing API remain in the selected V4 implementation.

The 17 templates are fitted on all official training GT with source labels used solely for global size-bank balancing; image-specific GT never reaches the generator. Shared preprocessing/dataset code and V1–V3 remain unchanged. Outputs are `summary.csv/json/png`, `by_label.csv`, `coverage_states.csv`, `per_gt.csv`, `per_image.csv`, `proposals.jsonl`, per-image figures and `index.html`.
