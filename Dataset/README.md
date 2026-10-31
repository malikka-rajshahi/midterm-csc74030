# NYC dataset — validation / test split

## Current version

This is the original 300-image NYC dataset split into **40% validation / 60% test**, stratified by the original `images.csv` group, with random seed **42**. No correction pass is applied: original image-level groups and all 547 source annotation rows are preserved. Images are copied unchanged from `nyc_dataset/`; only relative image paths in the split CSVs are updated.

| Original group | Folder | Validation | Test | Total |
|---|---|---:|---:|---:|
| cone_only | cone/ | 40 | 60 | 100 |
| sign_only | construction_sign/ | 40 | 60 | 100 |
| both | combination/ | 16 | 24 | 40 |
| negative | negative/ | 24 | 36 | 60 |
| **Total** | | **120** | **180** | **300** |

Validation contains 229 annotation rows; test contains 318. No images are withheld for review. The prior correction pass, filtered annotations, corrected labels and scene-cluster split are not part of this version.

## Folder contents

- `nyc_dataset/`: unchanged original source images, CSVs and source README.
- `validation/` and `test/`: category folders, `images.csv`, `annotations.csv`, a split README and `SOURCE_README.md` copied from the source.
- `split_summary.json`: seed, validation ratio, image/category counts and annotation counts.
- `split_dataset.py`: reproducible stratified splitting script. Its default ratio is 0.5; explicitly use `--validation-ratio 0.4` for this version.

Paths in each split's CSVs are relative to that split folder. Folder names map to the original group labels shown above; they must not be supplied as model features.

## Annotation status and limitations

Labels and boxes come from the original dataset and have not been corrected in this split. Source boxes are Mapillary segmentation-model predictions and may include incorrect classes, duplicates, missing targets or inaccurate extents. Image-level groups may also contain errors. The source README describes the original collection process; any reference there to manual checking does not mean this restored split includes the later correction pass. Do not describe these annotations as corrected or independently verified ground truth.

The split is stratified by the original group and uses distinct source image IDs and capture sequence IDs. It does not apply perceptual-hash clustering or establish geographic/scene independence.

## Development and final evaluation

Use **validation** for feature comparisons, template selection, debugging, threshold selection, parameter tuning and error analysis. If any fitting is introduced, use development data only and document it.

Use **test** only for final evaluation after the templates, preprocessing, feature extraction, thresholds and other settings are frozen. Extracting test features through the fixed inference pipeline is allowed. Do not fit, tune or select the method based on test images, labels, features or results. Keep validation/test feature caches separate.

The original `nyc_dataset/` contains both splits and must not be used as an extra development pool. See its README or each split's `SOURCE_README.md` for provenance, licensing and sampling details.

## Reproduce

Run from this folder, using a new output directory (the script refuses to overwrite existing splits):

```sh
python3 split_dataset.py --validation-ratio 0.4 --seed 42 --output /path/to/new/output
```
