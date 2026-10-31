# Intensity Notebooks

## 01 — Feature Comparison

Compare annotation regions with same-size unannotated patches, separately for `cone_only`, `sign_only`, and `both`.

- **Baseline:** normalized Histogram, Mean, Variance, Standard Deviation / RMS Contrast, Median, Mode, Trimmed Mean, Entropy, Gray-range Fraction.
- **Combinations:** Mean + Variance; Variance + Entropy; Mean + Std + Entropy.
- **Engineered:** Entropy × Contrast; Normalized Contrast (`std / (mean + 1)`); Spatial Entropy Difference (`|H(top) − H(bottom)|`).

Metrics: image-separated four-fold Balanced Accuracy; descriptive ROC-AUC and distribution overlap for scalar features. No universal winner; spatial entropy difference reaches **65.0% crop accuracy for sign_only**, but does not improve full-image detection.

## 02 — Validation Run

Multi-scale detection on all **120 validation images**, including negatives: template comparison → threshold → NMS. Compares Histogram, Entropy, and the three engineered features. Test is untouched.

## 03 — Validation Summary

| Feature | Precision | Recall | F1 |
|---|---:|---:|---:|
| Histogram | 0.025% | 2.19% | 0.050% |
| Entropy | 0% | 0% | 0% |
| Entropy × Contrast | 0% | 0% | 0% |
| Normalized Contrast | 0.042% | 0.88% | 0.080% |
| Spatial Entropy Difference | 0.020% | 0.44% | 0.038% |

**Conclusion:** none is an effective detector in the current average-template pipeline. Crop-level differences do not establish detection performance. Metrics use class-matched, one-to-one IoU ≥ 0.5 and validation-selected thresholds; they are development results on uncorrected source annotations.

## 04 — Template Diagnostics

Average-template path: **180 / 228** source boxes covered by sliding windows → **29** after top-1000 ranking → **16** after NMS → **6** after the max-100 output cap. Final one-to-one matching gives **5 true positives**; coverage and detection recall differ.

**Diagnosis:** most targets are lost during scoring/ranking, followed by NMS/output limits. Keeping top 10,000 candidates or using 8 representative actual templates per class did not improve final F1. Next: test target-versus-background scoring before adding more intensity formulas.

## Use

Start with **03** for results. Code and saved results are embedded; no external `.py` files are required. Recomputing needs the original `Test Dataset/validation/` and `vistas_templates/` folders. Mount Google Drive in Colab and check `PROJECT`. In 02/04, set `RUN_AGAIN=True` to recompute; outputs go to `intensity_results/` outside this folder. The saved snapshot in 03 does not update automatically.
