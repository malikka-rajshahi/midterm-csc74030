"""Fit global training-sized windows and evaluate class-independent ROI containment."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from .generate_rois import generate_proposals, load_config


def read_training(data):
    table=pd.read_parquet(Path(data)/'annotations.parquet')
    table=table[table.image_path.str.startswith('train/')].copy()
    coords=np.stack(table.bbox_coords).astype(float)
    if not np.isfinite(coords).all() or (coords[:,2:]<=0).any():
        raise ValueError('Invalid training boxes.')
    table[['x','y','w','h']]=coords
    return table.drop_duplicates(['image_path','label','x','y','w','h'])


def select_images(table,limit):
    paths=sorted(table.image_path.unique())
    if limit<=0 or limit>=len(paths):
        return paths
    groups={}
    rng=np.random.default_rng(42)
    for path in paths:
        groups.setdefault(path.split('/')[1],[]).append(path)
    for group in groups.values():
        rng.shuffle(group)
    selected=[]
    while len(selected)<limit:
        for camera in sorted(groups):
            if groups[camera] and len(selected)<limit:
                selected.append(groups[camera].pop())
    return selected


def fit_config(data,output,work_size=(1440,930)):
    """Global shape priors from canonical official-train boxes, never test labels."""
    data=Path(data)
    table=read_training(data)
    ww,wh=work_size
    shapes={}
    for path in sorted(table.image_path.unique()):
        with Image.open(data/path) as image:
            shapes[path]=image.size
    original=np.array([shapes[path] for path in table.image_path])
    xy=table[['x','y']].to_numpy()
    sizes=table[['w','h']].to_numpy()
    if (xy<0).any() or (xy+sizes>original).any():
        raise ValueError('Out-of-image GT.')
    scaled=sizes*np.array([ww,wh])/original
    table[['work_w','work_h']]=scaled
    bank=set()
    statistics=[]
    for label,group in table.groupby('label'):
        values=group[['work_w','work_h']].to_numpy()
        area=values.prod(axis=1)
        cuts=np.quantile(area,[0,.1,.25,.5,.75,.9,1])
        for index,(low,high) in enumerate(zip(cuts[:-1],cuts[1:])):
            subset=values[(area>=low)&((area<=high) if index==5 else (area<high))]
            if len(subset):
                # Context allowance is global, not fitted to individual evaluation GT.
                w,h=np.maximum(4,np.ceil(np.median(subset,axis=0)*1.25)).astype(int)
                bank.add((min(ww,int(w)),min(wh,int(h))))
        statistics.append(dict(source_label=label,boxes=len(group),
            width_p10_p50_p90=np.percentile(values[:,0],[10,50,90]).round(2).tolist(),
            height_p10_p50_p90=np.percentile(values[:,1],[10,50,90]).round(2).tolist()))
    config=dict(version=4,work_size=list(work_size),window_sizes=sorted(bank,key=lambda wh:wh[0]*wh[1]),
        saturation_min=.35,value_min=.15,anchor_blocks=[2,4,8],min_anchor_pixels=4,
        anchor_padding=.25,nearest_windows=3,max_roi_area_fraction=.25,
        merge_strategy='positive_intersection_closure',window_anchor_area_ratio=4.,
        training_provenance=dict(images=len(shapes),canonical_boxes=len(table),
            annotation_sha256=hashlib.sha256((data/'annotations.parquet').read_bytes()).hexdigest(),
            fit='Per-label area-quantile bins; median width/height times 1.25. Labels used only for global bank balancing.',
            gt_statistics=statistics))
    Path(output).write_text(json.dumps(config,indent=2))
    return config


def measure_gt(gt,proposals):
    """Independent containment; compactness is conditional on satisfying coverage."""
    gt=np.asarray(gt,float)
    proposals=np.asarray(proposals,float).reshape(-1,4)
    gt_area=np.prod(gt[2:]-gt[:2])
    area=np.prod(proposals[:,2:]-proposals[:,:2],axis=1)
    intersection=np.maximum(0,np.minimum(gt[2:],proposals[:,2:])-np.maximum(gt[:2],proposals[:,:2])).prod(axis=1)
    coverage=intersection/gt_area
    contains=(proposals[:,:2]<=gt[:2]).all(axis=1)&(proposals[:,2:]>=gt[2:]).all(axis=1)
    iou=intersection/np.maximum(gt_area+area-intersection,1e-30)
    result=dict(best_gt_coverage=float(coverage.max(initial=0)),covered_95=bool((coverage>=.95).any()),
                covered_100=bool(contains.any()),best_iou=float(iou.max(initial=0)))
    result['coverage_state']='complete_95' if result['covered_95'] else 'partial' if result['best_gt_coverage']>0 else 'none'
    for suffix,valid in [('95',coverage>=.95),('100',contains)]:
        eligible=np.flatnonzero(valid)
        index=int(eligible[np.argmin(area[eligible])]) if len(eligible) else -1
        result[f'compact_roi_{suffix}']=index
        result[f'area_ratio_{suffix}']=float(area[index]/gt_area) if index>=0 else np.nan
    return result


def draw_boxes(ax,boxes,color):
    boxes=np.asarray(boxes).reshape(-1,4)
    if not len(boxes):
        return
    a,b,c,d=boxes.T
    vertices=np.stack([np.c_[a,b],np.c_[c,b],np.c_[c,d],np.c_[a,d],np.c_[a,b]],axis=1)
    ax.add_collection(LineCollection(vertices,colors=color,linewidths=.7))


def plot_image(image, truth, result, records, output, title):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), layout='constrained')
    for ax in axes:
        ax.imshow(rgb)
        ax.axis('off')
    draw_boxes(axes[0], truth, 'cyan')
    axes[0].set_title(f'Supplied GT: {len(truth)} (unverified)')
    draw_boxes(axes[1], result['work_boxes'], 'lime')
    axes[1].set_title(f'Selected V4 green_blue_50: {len(result["work_boxes"])} ROIs')
    states = np.array([row['coverage_state'] for row in records])
    for state, color in [('complete_95', 'cyan'), ('partial', 'gold'), ('none', 'red')]:
        draw_boxes(axes[2], truth[states == state], color)
    for box, record in zip(truth, records):
        if record['coverage_state'] == 'partial':
            axes[2].text(box[0], box[1], f'{record["best_gt_coverage"]:.0%}', color='gold',
                         fontsize=8, bbox=dict(facecolor='black', alpha=.5, pad=.5))
    axes[2].set_title(f'GT status: complete {sum(states=="complete_95")} | partial {sum(states=="partial")} | none {sum(states=="none")}')
    fig.suptitle(title + '\nGreen: candidate ROIs | Cyan: supplied/covered GT | Gold: partial GT overlap | Red: no overlap. No classification.')
    fig.savefig(output, dpi=120)
    plt.close(fig)


def summarize(group):
    return dict(gt_boxes=len(group), complete_gt_boxes=int(group.covered_95.sum()),
                partial_gt_boxes=int((group.coverage_state == 'partial').sum()),
                no_overlap_gt_boxes=int((group.coverage_state == 'none').sum()),
                gt_coverage_95=float(group.covered_95.mean()),
                gt_coverage_100=float(group.covered_100.mean()),
                iou_05=float((group.best_iou >= .5).mean()),
                area_ratio_95_median=float(group.area_ratio_95.median()),
                area_ratio_95_p90=float(group.area_ratio_95.quantile(.9)),
                compact_coverage_95_area_le_4=float((group.covered_95 & (group.area_ratio_95 <= 4)).mean()))


def evaluate(data, output, config=None, limit=12, apply_denoise=True):
    """Evaluate only the selected pipeline, with no unused experiment variants."""
    data, output = Path(data), Path(output)
    cfg = load_config(config)
    table = read_training(data)
    selected = select_images(table, limit)
    (output/'images').mkdir(parents=True, exist_ok=True)
    rows, image_rows, gallery = [], [], []
    with (output/'proposals.jsonl').open('w') as proposal_file:
        for index, path in enumerate(selected):
            image = cv2.imread(str(data/path))
            if image is None:
                raise ValueError(f'Cannot read training image: {path}')
            height, width = image.shape[:2]
            group = table[table.image_path == path]
            truth = group[['x', 'y', 'w', 'h']].to_numpy().copy()
            truth[:, 2:] += truth[:, :2]
            if (truth < 0).any() or (truth[:, [0,2]] > width).any() or (truth[:, [1,3]] > height).any():
                raise ValueError(f'Out-of-image GT: {path}')
            ww, wh = cfg['work_size']
            truth *= [ww/width, wh/height, ww/width, wh/height]
            result = generate_proposals(image, config=cfg, apply_denoise=apply_denoise)
            records = []
            for number, (label, box) in enumerate(zip(group.label, truth)):
                record = measure_gt(box, result['work_boxes'])
                records.append(record)
                rows.append(dict(image=path, object_index=number, source_label=label,
                                 method='green_blue_50', gt_work_xyxy=json.dumps(box.tolist()), **record))
            image_rows.append(dict(image=path, method='green_blue_50', **result['counts'],
                                   runtime_seconds=result['runtime_seconds']))
            proposal_file.write(json.dumps(dict(image=path, method='green_blue_50',
                work_boxes=result['work_boxes'].tolist(), original_boxes=result['boxes'].tolist()))+'\n')
            filename = f'{index:03d}_{Path(path).stem}.png'
            plot_image(result['work_image_bgr'], truth, result, records, output/'images'/filename, path)
            gallery.append((path, filename))
            print(f'{index+1}/{len(selected)}: {len(result["work_boxes"])} selected ROIs', flush=True)
    objects, counts = pd.DataFrame(rows), pd.DataFrame(image_rows)
    objects.to_csv(output/'per_gt.csv', index=False)
    counts.to_csv(output/'per_image.csv', index=False)
    summary = pd.DataFrame([dict(method='green_blue_50', **summarize(objects),
        mean_roi_count=float(counts.final.mean()), mean_runtime_seconds=float(counts.runtime_seconds.mean()))])
    summary.to_csv(output/'summary.csv', index=False)
    pd.DataFrame([dict(source_label=label, **summarize(group))
                  for label, group in objects.groupby('source_label')]).to_csv(output/'by_label.csv', index=False)
    objects.groupby('coverage_state').size().rename('gt_boxes').to_csv(output/'coverage_states.csv')
    report = dict(selected_method='V4 green_blue_50', images=len(selected), config=cfg,
                  scope='Official training development only; test not used; GT not supplied to image-specific generation',
                  denoised=apply_denoise, summary=summary.iloc[0].to_dict())
    (output/'summary.json').write_text(json.dumps(report, indent=2))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
    bars = axes[0].bar(['Complete >=95%', 'Partial', 'No overlap'],
                      [objects.covered_95.sum(), (objects.coverage_state=='partial').sum(),
                       (objects.coverage_state=='none').sum()], color=['steelblue','gold','indianred'])
    axes[0].bar_label(bars, fmt='%d')
    axes[0].set_title('Supplied GT coverage states')
    axes[0].set_ylim(0, max(1,len(objects)))
    axes[1].bar(range(len(counts)), counts.final)
    axes[1].set(xlabel='Sample image index', ylabel='Candidate ROIs', title='Final ROI count per image')
    fig.suptitle('Selected pipeline: V4 green_blue_50 — training development sample')
    fig.savefig(output/'summary.png', dpi=150)
    plt.close(fig)
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Final ROI: V4 green_blue_50</title>',
        '<style>body{font:16px system-ui;margin:24px}img{width:100%}td,th{padding:8px}</style>',
        '<h1>Selected ROI pipeline: V4 green_blue_50</h1>',
        '<p>Selected for few candidates, non-overlapping output and target containment. This stage does not classify objects. The supplied GT is unverified and may include context or groups.</p>',
        '<p>One ROI can contain multiple targets. Large ROI/GT ratios flag extra background relative to individual GT, but do not alone invalidate a multi-target ROI. Downstream feature performance has not been evaluated.</p>',
        summary.to_html(index=False, float_format=lambda v:f'{v:.3f}'), '<img src="summary.png">',
        '<p>Training development only, not a held-out benchmark. See <a href="../README.md">README</a> for rules and unresolved limitations.</p>']
    for path, filename in gallery:
        body.append(f'<h3>{html.escape(path)}</h3><img loading="lazy" src="images/{filename}">')
    (output/'index.html').write_text('\n'.join(body)+'</html>')
    print(summary.to_string(index=False), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path.home()/'Downloads/datasets/Roadwork_Cones_Dataset/roadwork_cones_dataset')
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('roi_config.json'))
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('results'))
    parser.add_argument('--limit', type=int, default=12, help='0 means all official training images.')
    parser.add_argument('--fit-config', action='store_true')
    parser.add_argument('--no-denoise', action='store_true')
    args = parser.parse_args()
    if args.fit_config:
        fit_config(args.data, args.config)
    evaluate(args.data, args.output, args.config, args.limit, not args.no_denoise)
