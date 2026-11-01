# Final ROI selection: V4 `green_blue_50`

**Our selected ROI candidate-generation pipeline is V4 `green_blue_50`.** It matches the current objective: use color to locate potential targets, combine intersecting regions, and provide a small set of non-overlapping ROIs that can contain complete targets. A single ROI may contain several targets. Cone/Sign classification, intensity features and histogram matching are deferred to downstream stages.

## Version history

| Version | Main change |
|---|---|
| V1 | Color-block proposals with geometric duplicate suppression; many intersecting candidates remained. |
| V2 | One red/orange support mask, training-derived anchored windows, and bounded merging at 50% intersection/min-area overlap. |
| V3 | Raised overlap threshold to 70%; fully nested boxes became exempt from the 4x area-scale restriction. |
| V4 — selected | Combines all positive-area intersecting candidates into connected groups, including intersections created by group envelopes. Each final group becomes one rectangle. GT diagnosis distinguishes complete, partial and no overlap. |

V1–V3 retain independent historical code and results. Their reports document development, not alternative defaults. No cross-version composite figures are generated.

## Selected generation and filtering rules

1. **Prepare the image.** Resize to 1440x930 (width x height), denoise with `common.preprocessing.denoise`, and convert BGR to HSV with the shared helper. Original pixels remain available for downstream crops. GT is scaled separately for evaluation only.
2. **Build one red/orange support mask.** Hue uses degrees: red [0,15) or [345,360), orange [15,45). Require normalized S>=0.35 and V>=0.15. OpenCV uint8 H is multiplied by two. White, yellow, green and blue never independently trigger a candidate; white may remain inside an ROI.
3. **Find anchors.** At 2/4/8-pixel block granularities, retain blocks with at least two support pixels and 15% support; group 8-connected blocks. Regions need at least four support pixels. Tighten anchors to their red/orange support, not final ROIs to the color-mask shape.
4. **Place local multi-scale windows.** The 17 global width/height templates are derived from official training GT distributions at working resolution. Each anchor gets 25% context per side plus up to three compatible templates, with centered and 15% left/right/up/down offsets that preserve the anchor. Template compatibility limits excessive expansion: template fits the contextual anchor, is <=3x its width/height and <=4x its area. No full-image sliding-window grid or class-specific rule is used.
5. **Filter initial candidates.** Reject initial windows covering >25% of image area. Reject an entire ROI if **green OR blue pixels together exceed 50% of all ROI pixels**. Green is H [65,170), blue [190,260), using the same S/V gates. Keep equality at 50%. Do not remove individual background pixels or require white.
6. **Merge intersecting regions.** Any positive-area intersection connects boxes; edge contact alone does not. Replace each connected group with its smallest enclosing rectangle. Repeat when the new envelopes intersect, until outputs do not intersect. Combine provenance and recheck the green/blue rule on final groups. Reject failing groups as a whole.
7. **Keep group size visible.** A final merged group >25% of image area is flagged `oversized`, not split into overlapping members. V4 does not use V2/V3's 50%/70% merging thresholds or fixed-seed growth limits. These would leave overlaps. There is no top-K output cap.

The window bank is fitted globally from 2,871 official training images and 14,738 canonical source boxes. Source labels balance template fitting only; per-image generation receives no GT or labels. Vistas crop averages are references, not fixed Roadwork window sizes. Shared preprocessing and dataset code are unchanged.

## Results of the selected pipeline

Saved development run: the same 12 camera-balanced official training images, with 73 canonical supplied GT boxes (7 cone, 3 roadworks, 63 vertical_pannel).

| Metric | V4 `green_blue_50` |
|---|---:|
| Mean final ROI count/image | 6.83 |
| GT coverage >=95% | 36/73 (49.3%) |
| Strict GT containment =100% | 36/73 (49.3%) |
| Partial overlap, below95% | 2/73 |
| No ROI intersection | 35/73 |
| Intersecting final ROI pairs on the saved sample | 0 |
| Median smallest eligible ROI/GT area ratio, coverage >=95% | 259.68 |
| P90 eligible area ratio | 683.75 |
| Historical IoU>=0.5 matches | 1/73 (1.4%) |

Containment means an ROI covers the GT rectangle; equal boundaries are not required. Partial overlap is reported with its percentage, not counted as a correct detection. Area ratio uses the smallest ROI satisfying the coverage condition for each GT; uncovered GT has no ratio. Independent per-GT matching allows one ROI to cover multiple GTs.

**Why this is selected:** few candidates, no intersecting outputs on this sample, and full-target containment fit our current ROI objective. Large area ratios are a tradeoff: multi-target groups are large relative to individual GT boxes. They remain a useful warning about included background, not a standalone reason to reject the selected approach. Historical IoU favors tighter boxes and is not the current selection objective.

The name `green_blue_50` identifies the retained filtering rule; it does not prove that rule caused the improvement. In this sample the selected output equals the no-background-filter output from the previous controlled run, so no filtering benefit has been established.

## Unresolved issues

- **Incomplete coverage.** Two GTs are only partially overlapped and 35 have no intersection. Targets without qualifying red/orange support cannot supply their own color anchor. Changing window sizes alone cannot create that missing evidence.
- **Large groups and excess background.** Connecting chains of regions can yield a broad ROI containing several objects and substantial background. Whether this interferes with downstream intensity/color features must be tested; it is not yet established.
- **Non-target colored regions.** Red/orange surfaces outside the target classes can also generate candidates. ROI generation is not recognition; unmatched proposals cannot automatically be certified as false positives with incomplete annotations.
- **Annotation quality and grouping.** Supplied GT is unverified and may contain context or groups. No silent semantic relabeling or GT edits are performed.
- **Limited evidence.** The small, imbalanced development sample and training-fitted templates do not establish held-out accuracy, optimal thresholds, precision/F1 or final dataset performance. Official test is reserved until the pipeline is frozen.

## Use and reproduce

From the project root:

```python
import cv2
from common.ROI import generate_proposals  # selected V4 green_blue_50

image = cv2.imread('/absolute/path/to/image.jpg')
result = generate_proposals(image)
for x0, y0, x1, y1 in result['boxes']:
    roi = image[y0:y1, x0:x1]
```

`boxes` are half-open original-coordinate int32 xyxy; `work_boxes` use working coordinates. `work_image_bgr` is the actual denoised working image. Evidence contains estimated whole-ROI color ratios and anchor provenance, not class probabilities or true segmentation masks. An empty result is shape (0,4).

```sh
python -m common.ROI.v4.evaluate_rois --limit 12
# Same selected pipeline via compatibility entry point:
python -m common.ROI.evaluate_rois --limit 12
# Refit global windows from official training GT before evaluation:
python -m common.ROI.v4.evaluate_rois --fit-config --limit 12
```

`--limit 0` evaluates all official training images; `--output`, `--config`, and `--no-denoise` remain available. Dependencies: numpy, OpenCV, pandas, pyarrow, Pillow, matplotlib.

[Selected report](v4/result_official/index.html) · [V4 implementation details](v4/README.md)

Historical reports: [V1](v1/results/index.html), [V2](v2/results/index.html), [V3](v3/results/index.html). Explicit imports such as `common.ROI.v2` retain historical behavior; the parent API selects V4. V4 code and reports contain only the selected pipeline. The supplied reference PDF remains at the ROI root.
