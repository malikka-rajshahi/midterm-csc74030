# Intensity multi-scale detection baseline

Run `intensity_detection.ipynb` to review saved results, threshold curves and predicted boxes. It includes a Colab path and a rerun switch. Required packages: numpy, pillow, pandas, matplotlib; notebook display requires Jupyter. Code uses no test files and does not modify datasets.

## Exact configuration

- All 120 NYC validation images, including 24 negatives; 228 valid annotation rows, one invalid zero-width box excluded. Original duplicate/source errors retained.
- 12 detection windows: widths 8,16,24,32,48,64; width:height 1:1 and 1:2. Half-size strides (x=width/2, y=height/2), including right/bottom boundary alignment. No image resizing, blur or pyramid.
- Template library: only `role=template`, 243 cone and 47 sign masks. Normalize each mask-only 32-bin grayscale histogram, then average within class. Each class is represented by **one average prototype**, not nearest individual template. Entropy reference is mean individual-template entropy. Held-out Vistas validation roles are not used.
- Histogram distance: sum((p-q)^2/max(p,q)), ignoring both-zero bins. Entropy distance: absolute difference in 32-bin entropy. Previous exploratory notebook used 256-level entropy, so scores are not directly comparable.
- Class-wise NMS IoU=0.3; pre-NMS top 1000 windows/class; post-NMS max 100/class. Limits are part of this baseline; proposal truncation can lose valid targets. Each class is processed separately; the same location can yield two class hypotheses.
- Distance thresholds: histogram 0..2 by .05; entropy 0..5 by .1. Select each class threshold by class F1, tie higher precision then smaller threshold. Thresholds are selected on these validation results and must not be described as final held-out performance. No parameter tuning on test.
- One-to-one greedy prediction-to-GT matching by class at IoU>=0.5; predictions ranked by smaller template distance. Additional predictions on an already matched GT are false positives. All surviving source GT rows count independently.

## Results

| Feature | Precision | Recall | F1 | Seconds/image |
|---|---:|---:|---:|---:|
| histogram | 0.000255 | 0.021930 | 0.000503 | 0.152 |
| entropy | 0.000000 | 0.000000 | 0.000000 | 0.157 |

Selected thresholds: `{"histogram": {"cone": 0.4, "sign": 0.25}, "entropy": {"cone": 0.0, "sign": 0.0}}`.

These scores are poor and this baseline should not be treated as a working detector. The GT-crop exploration did not test localization. Averaging heterogeneous masked templates can lose distinguishing information; comparison windows include backgrounds; brightness/entropy do not encode object shape or color. Pre-NMS candidate limits can also reduce recall. The current experiment does not isolate the contributions of these factors. Entropy's zero-F1 grid has no useful selected operating point; the tie rule returns zero thresholds. This is **not proof that all intensity/template methods fail**.

Recommended next controlled comparison: use a bank of individual or representative template histograms instead of one averaged prototype, measure proposal survival before/after truncation, and compare color/shape features on the same candidates. Keep this baseline as an experiment record. Do not claim entropy is best based on the earlier crop study or combine statistics without end-to-end evidence.

## Speed and files

Inference timing includes image reading, window generation, shared integral-histogram extraction and feature-specific matching/NMS. Both methods are charged the same extraction time; entropy's implementation is not separately optimized. Template setup, threshold sweep, metric evaluation, output writing and plotting are excluded. Single machine/run, no repeated benchmark. The pipeline builds one integral image at a time, avoiding a 256-channel integral volume.

Results: `metrics.csv` (all, three positive groups, negative, cone/sign), `threshold_curves.csv`, `timings.csv`, `predictions.json` (post-NMS pre-threshold), `invalid_annotations.csv`, `summary.json` (full reproducible configuration and template prototypes).

From repository root:

```sh
python3 intensity_detection/pipeline.py
python3 intensity_detection/checks.py
```

Optional quarter stride, saved separately:

```sh
python3 intensity_detection/pipeline.py --stride-fraction 0.25 --output intensity_detection/results_quarter
```

Verification: integral histograms compared to direct pixel histograms on every synthetic candidate; NMS duplicate suppression; one-to-one matching including duplicate predictions and empty prediction/GT cases. The results notebook is executed and saved with charts.
