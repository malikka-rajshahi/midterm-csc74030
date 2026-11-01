"""Training-only proposal coverage and visual diagnostics for local Roadwork data."""
from pathlib import Path
import argparse
import hashlib
import html
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .generate_rois import generate_proposals

def read_training(data):
    """Canonical view only: no source edits, split changes or semantic relabeling."""
    table = pd.read_parquet(data / 'annotations.parquet')
    table = table[table.image_path.str.startswith('train/')].copy()
    coords = np.stack(table.bbox_coords).astype(float)
    if not np.isfinite(coords).all() or (coords[:, 2:] <= 0).any():
        raise ValueError('Invalid training bounding-box coordinates.')
    for i, column in enumerate(['x', 'y', 'w', 'h']):
        table[column] = coords[:, i]
    before = len(table)
    table = table.drop_duplicates(['image_path', 'label', 'x', 'y', 'w', 'h'])
    return table, before - len(table)


def select_images(table, limit):
    """Fixed camera-balanced training sample, selected without proposal scores."""
    paths = sorted(table.image_path.unique())
    if limit <= 0 or limit >= len(paths):
        return paths
    groups = {}
    rng = np.random.default_rng(42)
    for path in paths:
        groups.setdefault(path.split('/')[1], []).append(path)
    for group in groups.values():
        rng.shuffle(group)
    selected = []
    while len(selected) < limit:
        for camera in sorted(groups):
            if groups[camera] and len(selected) < limit:
                selected.append(groups[camera].pop())
    return selected


def best_iou(box, proposals):
    if not len(proposals):
        return 0.0
    intersection = np.maximum(0, np.minimum(box[2:], proposals[:, 2:]) -
                              np.maximum(box[:2], proposals[:, :2])).prod(axis=1)
    area = (box[2] - box[0]) * (box[3] - box[1])
    other = (proposals[:, 2] - proposals[:, 0]) * (proposals[:, 3] - proposals[:, 1])
    return float((intersection / np.maximum(area + other - intersection, 1e-30)).max())


def draw_boxes(ax, boxes, color, dashed=False):
    """Draw the complete proposal set efficiently, without a display cap."""
    from matplotlib.collections import LineCollection
    boxes = np.asarray(boxes).reshape(-1, 4)
    if not len(boxes):
        return
    x0, y0, x1, y1 = boxes.T
    corners = np.stack([np.column_stack((x0,y0)), np.column_stack((x1,y0)),
                        np.column_stack((x1,y1)), np.column_stack((x0,y1)),
                        np.column_stack((x0,y0))], axis=1)
    ax.add_collection(LineCollection(corners, colors=color, linewidths=.6,
                                     linestyles='--' if dashed else '-'))


def plot_scene(image, result, source_boxes, scores, output, name):
    rgb = cv2.cvtColor(result['image_bgr'], cv2.COLOR_BGR2RGB)
    canvas = np.zeros_like(rgb)
    for color, display_color in [('red', (255, 0, 0)), ('orange', (255, 128, 0)),
                                 ('yellow', (255, 255, 0)), ('blue', (0, 100, 255))]:
        canvas[result['color_masks'][color]] = display_color
    fig, axes = plt.subplots(1, 4, figsize=(22, 5.5), layout='constrained')
    axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw_boxes(axes[0], source_boxes, 'cyan', True)
    axes[0].set_title(f'Source boxes (unverified): {len(source_boxes)}')
    axes[1].imshow(canvas)
    axes[1].set_title('Estimated color evidence; not object masks')
    axes[2].imshow(rgb)
    draw_boxes(axes[2], result['boxes'], 'lime')
    axes[2].set_title(f'All candidate proposals: {len(result["boxes"])}')
    axes[3].imshow(rgb)
    missed = source_boxes[np.asarray(scores) < 0.5]
    draw_boxes(axes[3], missed, 'orange', True)
    axes[3].set_title(f'Not covered at IoU 0.5: {len(missed)}')
    for ax in axes:
        ax.axis('off')
    fig.suptitle(name + '\nCyan: source boxes | Green: all proposals, not detections | Orange: source boxes not covered at IoU 0.5.')
    fig.savefig(output, dpi=120)
    plt.close(fig)


def evaluate(data, output, limit=12, apply_denoise=True):
    data, output = Path(data), Path(output)
    table, removed = read_training(data)
    selected = select_images(table, limit)
    (output / 'images').mkdir(parents=True, exist_ok=True)
    rows, image_rows = [], []
    for index, path in enumerate(selected):
        image = cv2.imread(str(data / path))
        if image is None:
            raise ValueError(f'Cannot read training image: {path}')
        annotations = table[table.image_path == path]
        boxes = annotations[['x', 'y', 'w', 'h']].to_numpy().copy()
        boxes[:, 2:] += boxes[:, :2]  # source xywh -> image xyxy
        height, width = image.shape[:2]
        if ((boxes < 0).any() or (boxes[:, [0, 2]] > width).any() or
                (boxes[:, [1, 3]] > height).any()):
            raise ValueError(f'Out-of-bounds training annotation: {path}')
        result = generate_proposals(image, apply_denoise=apply_denoise, return_masks=True)
        scores = [best_iou(box, result['boxes']) for box in boxes]
        for label, score in zip(annotations.label, scores):
            rows.append(dict(source_label=label, best_iou=score))
        filename = f'{index:03d}_{Path(path).stem}.png'
        plot_scene(image, result, boxes, scores, output / 'images' / filename, path)
        image_rows.append(dict(image=path, proposals=len(result['boxes']), annotations=len(boxes),
                               covered_iou_05=sum(s >= 0.5 for s in scores), plot='images/' + filename))
        print(f'{index + 1}/{len(selected)} training images: {len(result["boxes"])} proposals', flush=True)
    objects = pd.DataFrame(rows)
    summary = []
    for label, group in objects.groupby('source_label'):
        summary.append(dict(source_label=label, annotation_boxes=len(group),
                            coverage_iou_03=float((group.best_iou >= 0.3).mean()),
                            coverage_iou_05=float((group.best_iou >= 0.5).mean()),
                            coverage_iou_07=float((group.best_iou >= 0.7).mean())))
    summary = pd.DataFrame(summary)
    summary.to_csv(output / 'coverage.csv', index=False)
    pd.DataFrame(image_rows).to_csv(output / 'per_image.csv', index=False)
    report = dict(scope='official training sample only; no test images read', images=len(selected),
                  training_exact_duplicate_rows_removed=removed, denoised=apply_denoise,
                  mean_proposals_per_image=float(np.mean([r['proposals'] for r in image_rows])),
                  labels_used_for_generation=False, source_labels_verified=False,
                  annotation_sha256=hashlib.sha256((data / 'annotations.parquet').read_bytes()).hexdigest())
    (output / 'summary.json').write_text(json.dumps(report, indent=2))
    body = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>ROI proposal diagnosis</title>',
            '<style>body{font:16px system-ui;margin:24px}img{width:100%}article{margin:24px 0}</style>',
            '<h1>Class-independent ROI proposals</h1>',
            '<p>Training sample only. Supplied labels are unverified. Coverage is independent per box, not detection recall. No class decisions or precision/F1 are reported.</p>',
            summary.to_html(index=False, float_format=lambda x: f'{x:.3f}'),
            f'<p>Mean candidates/image: {report["mean_proposals_per_image"]:.1f}; common denoise: {apply_denoise}.</p>']
    for row in image_rows:
        body.append(f'<article><h3>{html.escape(row["image"])}</h3><img loading="lazy" src="{row["plot"]}"></article>')
    body.append('</html>')
    (output / 'index.html').write_text('\n'.join(body))
    print(summary.to_string(index=False))
    return report, summary


def plot_comparison(summary, output):
    """Aligned count and coverage bars; no overlapping method labels."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout='constrained', sharey=True)
    y = np.arange(len(summary))
    axes[0].barh(y, summary.mean_candidates, color='steelblue')
    axes[0].set_yticks(y, summary.method)
    axes[0].invert_yaxis()
    axes[0].set(xscale='log', xlabel='Mean candidates / image (log scale)')
    axes[1].barh(y, summary.coverage_iou_05 * 100, color='seagreen')
    axes[1].set(xlim=(0, 100), xlabel='Source-box coverage at IoU 0.5 (%)')
    for i, row in enumerate(summary.itertuples()):
        axes[0].text(row.mean_candidates, i, f' {row.mean_candidates:,.0f}', va='center', fontsize=9)
        axes[1].text(row.coverage_iou_05*100, i,
                     f' {row.coverage_iou_05:.1%}; lost {row.lost_baseline_boxes}', va='center', fontsize=9)
    fig.suptitle('Training development sample: fewer candidates versus retained coverage\nLost = baseline-covered source boxes no longer covered; not detection accuracy')
    fig.savefig(output, dpi=150)
    plt.close(fig)


def compare_windows(data, output, limit=12, apply_denoise=True, legacy_colors=False):
    """Compare counts and source-box coverage without using labels to make ROIs."""
    from .sliding_window import WINDOW_PRESETS, sliding_windows, filter_color_windows, deduplicate_boxes
    data, output = Path(data), Path(output)
    table, removed = read_training(data)
    (output / 'images').mkdir(parents=True, exist_ok=True)
    color_options = dict(proposal_colors=('red','orange','yellow','blue'), yellow_max=75, max_green_fraction=None) if legacy_colors else {}
    object_rows, image_rows, gallery = [], [], []
    for index, path in enumerate(select_images(table, limit)):
        image = cv2.imread(str(data / path))
        if image is None:
            raise ValueError(f'Cannot read training image: {path}')
        base = generate_proposals(image, apply_denoise=apply_denoise,
                                  return_masks=True, dedup_iou=None, **color_options)
        annotations = table[table.image_path == path]
        truth = annotations[['x', 'y', 'w', 'h']].to_numpy().copy()
        truth[:, 2:] += truth[:, :2]
        height, width = image.shape[:2]
        if ((truth < 0).any() or (truth[:, [0, 2]] > width).any() or
                (truth[:, [1, 3]] > height).any()):
            raise ValueError(f'Out-of-bounds training annotation: {path}')
        raw = base['boxes']
        variants = {'blocks_raw': (raw, len(raw), len(raw))}
        for threshold in (.9, .8):
            keep, _ = deduplicate_boxes(raw, threshold)
            variants[f'blocks_dedup_{threshold}'] = (raw[keep], len(raw), len(raw))
        for name, sizes in [('medium', (8,16,32)), ('coarse', (16,32,64))]:
            result = generate_proposals(base['image_bgr'], apply_denoise=False, block_sizes=sizes, **color_options)
            variants[f'blocks_{name}'] = (result['boxes'], result['raw_count'], result['raw_count'])
        for preset, stride in [('small',.5), ('balanced',.5), ('balanced',.75), ('context',.75)]:
            windows = sliding_windows(image.shape[:2], WINDOW_PRESETS[preset], stride_fraction=stride)
            gated = filter_color_windows(windows, base['color_masks'],
                proposal_colors=color_options.get('proposal_colors', ('red','orange','yellow')),
                max_green_fraction=color_options.get('max_green_fraction', .5))
            keep, _ = deduplicate_boxes(gated, .9)
            variants[f'windows_{preset}_{stride}'] = (gated[keep], len(windows), len(gated))
        scores = {name: np.array([best_iou(box, value[0]) for box in truth])
                  for name, value in variants.items()}
        for name, (boxes, generated, gated) in variants.items():
            image_rows.append(dict(image=path, method=name, generated=generated,
                                   color_pass=gated, candidates=len(boxes)))
            for obj, (label, box, score) in enumerate(zip(annotations.label, truth, scores[name])):
                size = np.sqrt(np.prod(box[2:] - box[:2]))
                object_rows.append(dict(image=path, object_index=obj, method=name, source_label=label,
                    size_group='small' if size < 32 else 'medium' if size < 128 else 'large',
                    best_iou=score, baseline_covered=bool(scores['blocks_raw'][obj] >= .5)))
        names = ['blocks_raw', 'blocks_dedup_0.9', 'windows_balanced_0.75', 'windows_context_0.75']
        fig, axes = plt.subplots(1, 5, figsize=(25, 6), layout='constrained')
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        axes[0].imshow(rgb); draw_boxes(axes[0], truth, 'cyan', True)
        axes[0].set_title(f'Source boxes (unverified): {len(truth)}')
        for ax, name in zip(axes[1:], names):
            ax.imshow(rgb); draw_boxes(ax, variants[name][0], 'lime')
            draw_boxes(ax, truth[scores[name] < .5], 'orange', True)
            ax.set_title(f'{name}\nCandidates: {len(variants[name][0])}; covered: {sum(scores[name]>=.5)}/{len(truth)}')
        for ax in axes: ax.axis('off')
        fig.suptitle(path + '\nGreen: all candidates | Cyan: source boxes | Orange: source boxes not covered at IoU 0.5. No class predictions.')
        filename = f'{index:03d}_{Path(path).stem}.png'
        fig.savefig(output / 'images' / filename, dpi=100); plt.close(fig)
        gallery.append((path, filename))
        print(f'{index+1}: window comparison complete', flush=True)
    objects, counts = pd.DataFrame(object_rows), pd.DataFrame(image_rows)
    rows = []
    for name, group in objects.groupby('method', sort=False):
        rows.append(dict(method=name, mean_candidates=float(counts[counts.method==name].candidates.mean()),
            coverage_iou_03=float((group.best_iou>=.3).mean()),
            coverage_iou_05=float((group.best_iou>=.5).mean()),
            coverage_iou_07=float((group.best_iou>=.7).mean()),
            lost_baseline_boxes=int((group.baseline_covered & (group.best_iou<.5)).sum())))
    summary = pd.DataFrame(rows)
    detailed = []
    for columns in [('source_label',), ('size_group',)]:
        for (method, category), group in objects.groupby(['method', columns[0]]):
            detailed.append(dict(method=method, grouping=columns[0], category=category,
                                 boxes=len(group), coverage_iou_05=float((group.best_iou>=.5).mean())))
    summary.to_csv(output/'comparison.csv', index=False)
    pd.DataFrame(detailed).to_csv(output/'coverage_by_group.csv', index=False)
    counts.to_csv(output/'per_image.csv', index=False)
    objects.to_csv(output/'per_box.csv', index=False)
    eligible=summary[summary.lost_baseline_boxes==0].sort_values('mean_candidates')
    recommendation=eligible.iloc[0].method
    report=dict(scope='12-image training development comparison; official test untouched' if limit==12 else 'Training development comparison; official test untouched',
                images=len(gallery), duplicate_annotation_rows_removed=removed, legacy_colors=legacy_colors,
                annotation_sha256=hashlib.sha256((data/'annotations.parquet').read_bytes()).hexdigest(),
                window_widths=WINDOW_PRESETS, height_width_ratios=[.5,1.,2.,4.],
                denoised=apply_denoise, lowest_count_preserving_baseline= recommendation,
                selection_rule='Fewest candidates with zero lost baseline-covered boxes at IoU 0.5; preliminary sample only.')
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    plot_comparison(summary, output/'summary.png')
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><title>ROI window comparison</title>',
          '<style>body{font:16px system-ui;margin:24px}img{width:100%}table{border-collapse:collapse}td,th{padding:8px}</style>',
          '<h1>ROI window and duplicate comparison</h1>',
          '<p>Official training sample only. Source boxes are unverified and may contain context or object groups. Coverage is not classification accuracy. Green boxes are proposals, not detections. No output cap.</p>',
          summary.to_html(index=False,float_format=lambda x:f'{x:.3f}'),
          f'<p>Fewest candidates preserving every baseline-covered box in this sample: {recommendation}. This is not a universal optimal window size.</p>',
          '<p>Block activity uses local color fraction; sliding-window gating uses whole-window color fraction. Thus generator comparisons change more than size alone. Within sliding-window methods the same gate and aspect bank are used.</p>',
          '<p>Full-image sliding windows use a broad color-only gate: colored background can pass. They are an ablation, not the production default. Larger blocks can reduce counts while losing useful detail. Inspect the count and lost-coverage columns together; see README for the measured diagnosis.</p>',
          '<img src="summary.png">']
    body += [f'<h3>{html.escape(path)}</h3><img loading="lazy" src="images/{filename}">' for path,filename in gallery]
    (output/'index.html').write_text('\n'.join(body)+ '</html>')
    print(summary.to_string(index=False),flush=True)
    return report, summary


def compare_colors(data, output, limit=12, apply_denoise=True):
    """Isolate seed-color changes on the same training images and dedup threshold."""
    data, output = Path(data), Path(output)
    table, _ = read_training(data)
    (output/'images').mkdir(parents=True, exist_ok=True)
    rows, counts, gallery = [], [], []
    for index, path in enumerate(select_images(table, limit)):
        image = cv2.imread(str(data/path))
        if image is None: raise ValueError(f'Cannot read {path}')
        old = generate_proposals(image, apply_denoise=apply_denoise,
                                 proposal_colors=('red','orange','yellow','blue'), yellow_max=75, max_green_fraction=None)
        new = generate_proposals(old['image_bgr'], apply_denoise=False, max_green_fraction=None)
        annotations = table[table.image_path==path]
        truth = annotations[['x','y','w','h']].to_numpy().copy()
        truth[:,2:] += truth[:,:2]
        scores = {}
        variants = {'previous_blue_yellow75': old, 'warm_only_yellow65': new}
        for threshold in (.4,.5):
            variants[f'warm_green_max_{threshold}'] = generate_proposals(
                old['image_bgr'], apply_denoise=False, max_green_fraction=threshold)
        for method, result in variants.items():
            scores[method] = np.array([best_iou(box,result['boxes']) for box in truth])
            counts.append(dict(image=path,method=method,candidates=len(result['boxes'])))
            for obj,(label,score) in enumerate(zip(annotations.label,scores[method])):
                rows.append(dict(image=path,object_index=obj,method=method,source_label=label,best_iou=score))
        fig, axes = plt.subplots(1,5,figsize=(25,6),layout='constrained')
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        for ax in axes: ax.imshow(rgb); ax.axis('off')
        draw_boxes(axes[0],truth,'cyan',True); axes[0].set_title(f'Source boxes: {len(truth)}')
        for ax,(method,result) in zip(axes[1:],variants.items()):
            draw_boxes(ax,result['boxes'],'lime')
            draw_boxes(ax, truth[scores[method]<.5], 'orange', True)
            ax.set_title(f'{method}\nCandidates {len(result["boxes"])}; covered {sum(scores[method]>=.5)}/{len(truth)}')
        fig.suptitle(path+'\nGreen: proposals, not detections | Blue/green pixels do not seed new proposals; background inside boxes is allowed.')
        filename=f'{index:03d}_{Path(path).stem}.png'
        fig.savefig(output/'images'/filename,dpi=110);plt.close(fig)
        gallery.append((path,filename));print(f'{index+1}: color comparison complete',flush=True)
    objects=pd.DataFrame(rows);counts=pd.DataFrame(counts)
    objects.to_csv(output/'per_box.csv',index=False);counts.to_csv(output/'per_image.csv',index=False)
    summary=objects.groupby(['method','source_label']).best_iou.agg(boxes='size',coverage_iou_05=lambda v:(v>=.5).mean()).reset_index()
    summary.to_csv(output/'coverage.csv',index=False)
    totals=objects.groupby('method').best_iou.agg(boxes='size',covered=lambda v:int((v>=.5).sum())).reset_index()
    totals['mean_candidates']=totals.method.map(counts.groupby('method').candidates.mean())
    pivot=objects.pivot(index=['image','object_index'],columns='method',values='best_iou')
    chart=totals.rename(columns={'covered':'covered_boxes'}).copy()
    chart['coverage_iou_05']=chart.covered_boxes/chart.boxes
    chart['lost_baseline_boxes']=[int(((pivot.previous_blue_yellow75>=.5)&(pivot[name]<.5)).sum()) for name in chart.method]
    plot_comparison(chart,output/'summary.png')
    report=dict(training_images=len(gallery),test_images_read=0,dedup_iou=.9,
        baseline_covered_lost=int(((pivot.previous_blue_yellow75>=.5)&(pivot.warm_only_yellow65<.5)).sum()),
        newly_covered=int(((pivot.previous_blue_yellow75<.5)&(pivot.warm_only_yellow65>=.5)).sum()),
        seeds=['red','orange','yellow'],yellow_degrees=[45,65],green_blue_evidence_only=True)
    report['green_gate_vs_warm'] = {name: dict(
        lost=int(((pivot.warm_only_yellow65>=.5)&(pivot[name]<.5)).sum()),
        gained=int(((pivot.warm_only_yellow65<.5)&(pivot[name]>=.5)).sum()))
        for name in ['warm_green_max_0.4','warm_green_max_0.5']}
    (output/'summary.json').write_text(json.dumps(report,indent=2))
    body=['<!doctype html><html lang="en"><meta charset="utf-8"><title>ROI color comparison</title><style>body{font:16px system-ui;margin:24px}img{width:100%}td,th{padding:8px}</style>',
          '<h1>Warm-color seeds and optional green-background rejection</h1><p>Same training images and IoU 0.90 deduplication. Green/blue are evidence only; their presence inside a box does not veto a real object with background.</p>',totals.to_html(index=False),summary.to_html(index=False),'<img src="summary.png">',
          f'<p>Previously covered source boxes lost: {report["baseline_covered_lost"]}; newly covered: {report["newly_covered"]}. Coverage is not detection accuracy. Narrowing colors may remove legitimate objects.</p>']
    body += [f'<h3>{html.escape(path)}</h3><img loading="lazy" src="images/{filename}">' for path,filename in gallery]
    (output/'index.html').write_text('\n'.join(body)+'</html>')
    print(totals.to_string(index=False),flush=True);print(json.dumps(report),flush=True)
    return report,totals


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path.home() / 'Downloads/datasets/Roadwork_Cones_Dataset/roadwork_cones_dataset')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
    parser.add_argument('--limit', type=int, default=12, help='Number of training images; 0 means all.')
    parser.add_argument('--no-denoise', action='store_true', help='Diagnostic ablation only.')
    parser.add_argument('--legacy-colors', action='store_true', help='Reproduce the blue-inclusive original window comparison.')
    parser.add_argument('--compare-colors', action='store_true', help='Compare old blue-inclusive seeds with the warm-only defaults.')
    parser.add_argument('--compare-windows', action='store_true', help='Compare window banks, strides and geometric deduplication.')
    args = parser.parse_args()
    if args.compare_windows and args.compare_colors:
        parser.error("Choose one comparison mode.")
    runner = compare_colors if args.compare_colors else compare_windows if args.compare_windows else evaluate
    if args.legacy_colors and not args.compare_windows:
        parser.error('--legacy-colors requires --compare-windows.')
    options = dict(legacy_colors=args.legacy_colors) if args.compare_windows else {}
    runner(args.data, args.output, args.limit, not args.no_denoise, **options)
