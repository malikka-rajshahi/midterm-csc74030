"""Validation-only annotation coverage study; no feature extraction or matching."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from sliding_windows import WINDOWS, axis_positions, best_window


def read(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def write(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values, q):
    values = sorted(values)
    pos = (len(values)-1)*q
    lo = int(pos)
    hi = min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(pos-lo)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[1]/'Test Dataset'/'validation')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent/'results')
    args = parser.parse_args()
    groups = ('cone_only', 'sign_only', 'both')
    images = {r['image']: r for r in read(args.data/'images.csv')}
    images = {key: row for key, row in images.items() if row['group'] in groups}
    boxes, invalid = [], []
    for index, row in enumerate(read(args.data/'annotations.csv')):
        if row['image'] not in images:
            continue
        image = images[row['image']]
        box = tuple(float(row[k]) for k in ('x0','y0','x1','y1'))
        w,h = int(image['width']), int(image['height'])
        if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
            invalid.append(dict(row, group=image['group'], annotation_row=index, reason='degenerate_or_out_of_bounds'))
            continue
        boxes.append(dict(row, group=image['group'], annotation_row=index, box=box, iw=w, ih=h))
    if not boxes:
        raise ValueError('No valid annotations')
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output/'invalid_annotations.csv', invalid)
    stats=[]
    for group in groups:
        subset=[r for r in boxes if r['group']==group]
        for metric, values in [('width',[r['box'][2]-r['box'][0] for r in subset]),('height',[r['box'][3]-r['box'][1] for r in subset]),('height_over_width',[(r['box'][3]-r['box'][1])/(r['box'][2]-r['box'][0]) for r in subset])]:
            stats.append(dict(group=group,metric=metric,n=len(values),**{label:quantile(values,q) for label,q in [('min',0),('p10',.1),('p25',.25),('median',.5),('p75',.75),('p90',.9),('max',1)]}))
    write(args.output/'box_statistics.csv',stats)
    results=[]
    per_box=[]
    # Only two predeclared stride policies; all 12 requested geometries retained.
    for fraction in (.5,.25):
        scores={}
        for width,height in WINDOWS:
            sx,sy=max(1,round(width*fraction)),max(1,round(height*fraction))
            key=f'{width}x{height}'
            scores[key]=[best_window(r['box'],r['iw'],r['ih'],width,height,sx,sy) for r in boxes]
        for key in list(scores)+['all_12']:
            selected = scores[key] if key!='all_12' else [max((scores[k][i] for k in scores),key=lambda pair:pair[0]) for i in range(len(boxes))]
            # Count scans on the images belonging to each positive group.
            for group in groups:
                values=[v[0] for r,v in zip(boxes,selected) if r['group']==group]
                group_images=[im for im in images.values() if im['group']==group]
                geometries=WINDOWS if key=='all_12' else [tuple(map(int,key.split('x')))]
                cost=sum(len(axis_positions(int(im['width']),w,max(1,round(w*fraction))))*len(axis_positions(int(im['height']),h,max(1,round(h*fraction)))) for im in group_images for w,h in geometries)
                results.append(dict(window=key,stride_fraction=fraction,group=group,images=len(group_images),annotations=len(values),coverage_iou_030=sum(v>=.3 for v in values)/len(values),coverage_iou_050=sum(v>=.5 for v in values)/len(values),coverage_iou_070=sum(v>=.7 for v in values)/len(values),mean_best_iou=sum(values)/len(values),total_windows=cost,windows_per_image=cost/len(group_images)))
            if key=='all_12':
                for r,(iou,window) in zip(boxes,selected):
                    per_box.append(dict(annotation_row=r['annotation_row'],image=r['image'],group=r['group'],cls=r['class'],stride_fraction=fraction,best_iou=iou,best_window=json.dumps(window)))
    write(args.output/'coverage.csv',results)
    write(args.output/'per_annotation.csv',per_box)
    summary=dict(images=len(images),valid_annotations=len(boxes),excluded_annotations=len(invalid),image_groups=dict(Counter(r['group'] for r in images.values())),annotation_groups=dict(Counter(r['group'] for r in boxes)),class_counts=dict(Counter(r['class'] for r in boxes)),data=str(args.data.resolve()),note='Annotation proposal coverage only; not detector recall. Original predictions are uncorrected; duplicates retained.')
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))
    for row in results:
        if row['window']=='all_12': print(row)


if __name__=='__main__':
    main()
