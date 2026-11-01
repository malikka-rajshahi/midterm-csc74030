"""Independent detection evaluation; original-image xywh boxes throughout.

No feature extraction, prediction, threshold fitting or test-driven calibration.
Matching is classwise greedy in ascending distance order (not confidence/AP).
"""
from pathlib import Path
import argparse
import csv
import json
import numpy as np


def _box(box):
    values = np.asarray(box, dtype=float)
    if values.shape != (4,) or not np.isfinite(values).all() or (values[:2] < 0).any() or (values[2:] <= 0).any():
        raise ValueError('Expected finite nonnegative x/y and positive width/height.')
    return values


def _records(data):
    if isinstance(data, dict) and 'images' in data:
        if data.get('bbox_format', 'xywh') != 'xywh' or data.get('coordinate_space', 'original') != 'original':
            raise ValueError('Evaluation requires original-image xywh.')
        data = data['images']
    elif isinstance(data, dict) and 'image_id' in data:
        data = [data]
    elif isinstance(data, dict):
        data = [dict(image_id=identity, detections=items) for identity, items in data.items()]
    records = {}
    for record in data:
        identity = str(record['image_id']).replace('\\', '/')
        if identity in records:
            raise ValueError(f'Duplicate image ID: {identity}')
        if record.get('bbox_format', 'xywh') != 'xywh' or record.get('coordinate_space', 'original') != 'original':
            raise ValueError('Evaluation requires original-image xywh.')
        items = record.get('detections', record.get('annotations', []))
        checked = []
        for item in items:
            if item.get('accepted', True) is False:
                continue
            label = item['class']
            if not isinstance(label, str) or not label:
                raise ValueError('Class labels must be nonempty strings.')
            box = _box(item['bbox'])
            if 'image_size' in record and (box[:2]+box[2:] > np.asarray(record['image_size'])+1e-6).any():
                raise ValueError('Box exceeds original image_size.')
            checked.append(dict(item, bbox=box.tolist()))
        records[identity] = checked
    return records


def load_predictions(path):
    """Read detection.py JSON/CSV including images with zero predictions."""
    path = Path(path)
    if path.suffix.lower() == '.json':
        return _records(json.loads(path.read_text()))
    if path.suffix.lower() != '.csv':
        raise ValueError('Predictions must be .json or .csv.')
    records = {}
    with path.open(newline='') as stream:
        for row in csv.DictReader(stream):
            identity = row['image_id']
            if identity not in records:
                metadata = json.loads(row.get('metadata_json') or '{}')
                records[identity] = dict(metadata, image_id=identity, detections=[])
            if row['x'] == '':
                continue
            records[identity]['detections'].append(dict(
                bbox=[float(row[k]) for k in ('x', 'y', 'width', 'height')],
                **{'class': row['class']}, distance=float(row['distance']),
                accepted=row['accepted'].lower() == 'true'))
    return _records(list(records.values()))


def load_ground_truth(path, *, split=None, image_ids=None):
    """Roadwork annotations.parquet, or original xywh JSON/flat CSV.

    Roadwork columns: image_path, label, bbox_coords=[x,y,w,h]. Exact duplicate
    annotations are removed, consistent with existing common dataset handling.
    split filters a path prefix ('train/' or 'test/'); never changes class names.
    image_ids selects an explicit subset and may include known empty images.
    JSON accepts the same images schema with annotations instead of detections.
    CSV columns: image_id,x,y,width,height,class; blank x denotes an empty image.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == '.json':
        records = _records(json.loads(path.read_text()))
    elif suffix == '.parquet':
        import pandas as pd
        table = pd.read_parquet(path)
        required = {'image_path', 'label', 'bbox_coords'}
        if not required.issubset(table.columns):
            raise ValueError(f'Roadwork annotations require {required}.')
        records = {}
        seen = set()
        for row in table.itertuples():
            identity = str(row.image_path).replace('\\', '/')
            if split is not None and not identity.startswith(split.rstrip('/')+'/'):
                continue
            box = _box(row.bbox_coords).tolist()
            key = (identity, row.label, *box)
            if key not in seen:
                records.setdefault(identity, []).append(dict(bbox=box, **{'class': row.label}))
                seen.add(key)
        records = _records(records)
    elif suffix == '.csv':
        records = {}
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                items = records.setdefault(row['image_id'], [])
                if row['x'] != '':
                    items.append(dict(bbox=[float(row[k]) for k in ('x', 'y', 'width', 'height')], **{'class': row['class']}))
        records = _records(records)
    else:
        raise ValueError('Ground truth must be .parquet, .json or .csv.')
    if split is not None:
        records = {k: v for k, v in records.items() if k.startswith(split.rstrip('/')+'/')}
    if image_ids is not None:
        scope = {str(identity).replace('\\', '/') for identity in image_ids}
        # Explicit subset evaluation; caller selects a complete list, not only TP images.
        records = {identity: records.get(identity, []) for identity in sorted(scope)}
    return records


def calculate_iou(box_a, box_b):
    """Scalar IoU for half-open original xywh; touching edges have zero area."""
    a, b = _box(box_a), _box(box_b)
    extent = np.maximum(0, np.minimum(a[:2]+a[2:], b[:2]+b[2:])-np.maximum(a[:2], b[:2]))
    intersection = float(np.prod(extent))
    return intersection/(float(np.prod(a[2:])+np.prod(b[2:]))-intersection)


def match_detections_to_gt(predictions, ground_truth, iou_threshold=.5, class_mapping=None):
    """Same-class one-to-one matching; lower distance first, input order for ties.

    Each prediction takes the unmatched same-class GT with greatest qualifying
    IoU. Distance is only a within-configuration ranking, never a probability.
    Unmatched prediction reasons are diagnostic, not extra TP/FP categories.
    """
    if not np.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError('iou_threshold must be in (0,1].')
    mapping = {} if class_mapping is None else class_mapping
    def label(item):
        result = mapping.get(item['class'], item['class'])
        if not isinstance(result, str) or not result:
            raise ValueError('Mapped classes must be nonempty strings.')
        return result
    predictions = _records({'image_id': 'image', 'detections': predictions})['image']
    ground_truth = _records({'image_id': 'image', 'annotations': ground_truth})['image']
    distances = np.array([p.get('distance', 0.) for p in predictions], float)
    if not np.isfinite(distances).all() or (distances < 0).any():
        raise ValueError('Prediction distances must be finite and nonnegative.')
    truth_labels = [label(g) for g in ground_truth]
    used, matched, unmatched_predictions = set(), [], []
    for index in np.argsort(distances, kind='stable'):
        index = int(index)
        prediction = predictions[index]
        current = label(prediction)
        overlaps = [calculate_iou(prediction['bbox'], g['bbox']) for g in ground_truth]
        available = [j for j, value in enumerate(overlaps)
                     if j not in used and truth_labels[j] == current and value >= iou_threshold]
        if available:
            target = max(available, key=lambda j: overlaps[j])
            used.add(target)
            matched.append(dict(prediction_index=index, gt_index=target, **{'class': current}, iou=overlaps[target]))
        else:
            same = [value for j, value in enumerate(overlaps) if truth_labels[j] == current]
            reason = ('duplicate' if any(value >= iou_threshold for value in same) else
                      'wrong_class' if any(value >= iou_threshold for value in overlaps) else
                      'localization' if any(value > 0 for value in same) else 'background')
            unmatched_predictions.append(dict(prediction_index=index, **{'class': current}, reason=reason))
    missing = [dict(gt_index=j, **{'class': truth_labels[j]}) for j in range(len(ground_truth)) if j not in used]
    return dict(matches=matched, unmatched_predictions=unmatched_predictions, unmatched_gt=missing)


def calculate_metrics(tp, fp, fn):
    """Zero denominator -> 0.0, including an entirely empty evaluation."""
    if any(not isinstance(v, (int, np.integer)) or v < 0 for v in (tp, fp, fn)):
        raise ValueError('Counts must be nonnegative integers.')
    return dict(TP=int(tp), FP=int(fp), FN=int(fn),
                precision=tp/(tp+fp) if tp+fp else 0.,
                recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def evaluate_dataset(predictions, ground_truth, iou_threshold=.5, class_mapping=None):
    """Evaluate union of image IDs; absent predictions are empty, never dropped.

    Supply only the intended split/subset on both sides. IDs must be identical
    relative paths, not bare basenames. Unknown prediction-only images count FP.
    """
    if not np.isfinite(iou_threshold) or not 0 < iou_threshold <= 1:
        raise ValueError('iou_threshold must be in (0,1].')
    predictions = load_predictions(predictions) if isinstance(predictions, (str, Path)) else _records(predictions)
    ground_truth = load_ground_truth(ground_truth) if isinstance(ground_truth, (str, Path)) else _records(ground_truth)
    per_image, counts, failure_counts = {}, {}, {}
    mapping = {} if class_mapping is None else class_mapping
    for identity in sorted(set(predictions) | set(ground_truth)):
        pred, gt = predictions.get(identity, []), ground_truth.get(identity, [])
        result = match_detections_to_gt(pred, gt, iou_threshold, mapping)
        for item in pred+gt:
            counts.setdefault(mapping.get(item['class'], item['class']), [0, 0, 0])
        for item in result['matches']:
            counts[item['class']][0] += 1
        for item in result['unmatched_predictions']:
            counts[item['class']][1] += 1
            failure_counts[item['reason']] = failure_counts.get(item['reason'], 0)+1
        for item in result['unmatched_gt']:
            counts[item['class']][2] += 1
        per_image[identity] = dict(calculate_metrics(len(result['matches']), len(result['unmatched_predictions']), len(result['unmatched_gt'])), **result)
    totals = np.array(list(counts.values()), dtype=int).reshape(-1, 3).sum(axis=0)
    return dict(iou_threshold=float(iou_threshold), bbox_format='xywh', coordinate_space='original',
        zero_denominator=0., matching_order='ascending distance, stable ties, same-class greedy max IoU',
        class_mapping=mapping, image_count=len(per_image), overall=calculate_metrics(*totals),
        per_class={k: calculate_metrics(*v) for k, v in sorted(counts.items())},
        failure_counts=failure_counts, per_image=per_image)


def generate_report(report, output_dir):
    """Save full report JSON plus small overall/per-class/per-image CSV tables."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    for name, rows in (
        ('overall', [report['overall']]),
        ('per_class', [dict(label=k, **v) for k, v in report['per_class'].items()]),
        ('per_image', [dict(image_id=k, **{m: v[m] for m in ('TP', 'FP', 'FN', 'precision', 'recall', 'f1')}) for k, v in report['per_image'].items()]),
        ('failure_counts', [dict(reason=k, count=v) for k, v in report['failure_counts'].items()])):
        keys = list(rows[0]) if rows else {'per_class': ['label','TP','FP','FN','precision','recall','f1'],
            'per_image': ['image_id','TP','FP','FN','precision','recall','f1'],
            'failure_counts': ['reason','count']}[name]
        with (output/f'{name}.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
    return output/'report.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', required=True)
    parser.add_argument('--gt', required=True)
    parser.add_argument('--split', help='Annotation path prefix, e.g. test. Retains test/ in IDs.')
    parser.add_argument('--image-prefix', default='', help='Prefix prediction IDs, e.g. test/ when input directory was test/.')
    parser.add_argument('--class-mapping', help='Optional JSON mapping; applied to both GT and predictions.')
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    predictions = load_predictions(args.predictions)
    if args.image_prefix:
        predictions = {args.image_prefix.rstrip('/')+'/'+k: v for k, v in predictions.items()}
    truth = load_ground_truth(args.gt, split=args.split)
    mapping = json.loads(Path(args.class_mapping).read_text()) if args.class_mapping else None
    report = evaluate_dataset(predictions, truth, args.iou, mapping)
    generate_report(report, args.output)
    print(json.dumps(report['overall'], indent=2))


if __name__ == '__main__':
    main()
