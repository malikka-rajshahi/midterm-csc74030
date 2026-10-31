import csv
from pathlib import Path
p=Path(__file__).parent
read=lambda name:list(csv.DictReader((p/'results'/name).open()))
coverage=read('coverage.csv'); stats=read('box_statistics.csv')
references=[]
for group in ('cone_only','sign_only','both'):
 candidates=[r for r in coverage if r['group']==group and r['window']!='all_12']
 best=max(candidates,key=lambda r:(float(r['coverage_iou_050']),-float(r['windows_per_image'])) )
 references.append(dict(group=group,best_single_window=best['window'],stride_fraction=best['stride_fraction'],coverage_iou_050=best['coverage_iou_050'],windows_per_image=best['windows_per_image']))
with (p/'results/reference_values.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(references[0]));w.writeheader();w.writerows(references)
text='''# Experiment A — Positive-group Detection Window Coverage

## Scope and interpretation

Only validation images in the original **cone_only**, **sign_only** and **both** groups are included (40, 40 and 16 images). Negative images and their annotations are excluded. Test is not read. Groups come from original `images.csv`, not corrected labels. `both` means individual cone and sign annotation boxes in those images, **not a single box enclosing both objects**. Results are annotation-weighted, not the fraction of images where both classes are successfully covered.

There are 228 valid boxes: 78 cone_only, 56 sign_only, 94 both. One zero-width sign box is excluded and recorded, without editing the dataset. All original duplicate annotations are retained. These annotations are uncorrected model predictions; coverage is a geometric reference, not detector accuracy or certified GT quality.

## Box size reference (pixels)

Width and height quantiles are calculated independently; the median width/height pair is a descriptive reference, not necessarily an actual box or the optimal window.

| Positive group | Valid boxes | Median width × height | Width P25–P75 | Height P25–P75 | Median height/width |
|---|---:|---:|---:|---:|---:|
'''
for group in ('cone_only','sign_only','both'):
 s={r['metric']:r for r in stats if r['group']==group}
 w,h,a=s['width'],s['height'],s['height_over_width']
 text+=f"| {group} | {w['n']} | {float(w['median']):g} × {float(h['median']):g} | {float(w['p25']):g}–{float(w['p75']):g} | {float(h['p25']):g}–{float(h['p75']):g} | {float(a['median']):.2f} |\n"
text+='''
## Coverage reference

12 geometries: widths 8, 16, 24, 32, 48, 64; width:height 1:1 and 1:2 (8×8 through 64×128). Images stay at original resolution. Half stride = (width/2, height/2); quarter stride = (width/4, height/4). Windows are fully inside the image, including appended right/bottom boundary positions. No padding or image pyramid.

A box is covered when any candidate reaches IoU >= 0.5 (primary). IoU measures overlap relative to the combined area of the two boxes. Coverage at 0.3 and 0.7 is also saved. Scanning cost is calculated **only on images of the corresponding group**, including images with no valid annotations. No class-specific window choice may depend on the unknown label at inference; these separate statistics diagnose the common window bank.

| Group | All 12, half stride | All 12, quarter stride | Windows/image, half | Windows/image, quarter |
|---|---:|---:|---:|---:|
'''
for group in ('cone_only','sign_only','both'):
 a,b=[next(r for r in coverage if r['group']==group and r['window']=='all_12' and r['stride_fraction']==str(f)) for f in (.5,.25)]
 text+=f"| {group} | {float(a['coverage_iou_050']):.1%} | {float(b['coverage_iou_050']):.1%} | {float(a['windows_per_image']):,.0f} | {float(b['windows_per_image']):,.0f} |\n"
text+='''
## Single-window reference

Best means highest validation annotation coverage at IoU >= 0.5 among the 12 geometries and two stride policies; ties favor fewer windows. This is not the best detector and these values are not frozen final choices.

| Group | Best single window (width × height) | Stride (x, y) | Coverage @0.5 |
|---|---:|---:|---:|
'''
for r in references:
 w,h=map(int,r['best_single_window'].split('x'));f=float(r['stride_fraction'])
 text+=f"| {r['group']} | {w} × {h} | ({round(w*f)}, {round(h*f)}) | {float(r['coverage_iou_050']):.1%} |\n"
text+='''
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
'''
(p/'README.md').write_text(text)
print(text)
