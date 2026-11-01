"""GT-free V4 -> local windows -> existing features/matching -> detections.

Public/exported boxes are original-image xywh. Internal boxes are half-open xyxy.
Run from the repository root: python -m detection.detection --help.
"""
from pathlib import Path
import argparse
import csv
import json
import time
import cv2
import numpy as np
from common.ROI.v4.generate_rois import generate_proposals, load_config as load_roi_config
from common.ROI_refinement.generate_windows import generate_windows, load_config as load_window_config
from .feature_extract import extract_features
from .feature_matching import match_features, postprocess_detections


def load_image(image):
    """Read a path, or copy an explicitly uint8 BGR array (not RGB)."""
    if isinstance(image, (str, Path)):
        path = Path(image)
        if not path.is_file():
            raise FileNotFoundError(path)
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) == 0:
        raise ValueError('Expected a readable image or nonempty uint8 BGR array.')
    return image.copy()


def generate_candidates(image_bgr, roi_config=None):
    """Reuse selected V4 green_blue_50 and S2; FINAL fallback is opt-in."""
    options = {} if roi_config is None else roi_config
    coarse_config = options.get('v4', load_roi_config())
    window_config = options.get('windows', load_window_config())
    if list(coarse_config['work_size']) != list(window_config['work_size']):
        raise ValueError('V4 and sliding windows must use the same work_size.')
    method = options.get('method', 'S2')
    if method not in ('S2', 'FINAL'):
        raise ValueError('Use S2 or the previously selected FINAL S2+S1 fallback.')
    coarse = generate_proposals(image_bgr, config=coarse_config,
                                apply_denoise=options.get('apply_denoise', True))
    windows = generate_windows(coarse['work_boxes'], config=window_config, method=method)
    return dict(boxes=windows['boxes'], work_image_bgr=coarse['work_image_bgr'],
                coarse_boxes=coarse['work_boxes'], work_size=coarse['work_size'],
                method=method, roi_counts=coarse['counts'])


def _feature_matrix(extracted, config):
    """Concatenate only explicitly scaled blocks; no fitted parameters here.

    Transform: (feature - offset) / scale * weight, applied identically to train,
    validation and candidates. For a single feature all defaults are identity.
    Combined requires explicit scales/weights plus its own calibrated threshold.
    """
    names = config['names']
    names = [names] if isinstance(names, str) else list(names)
    if not names or len(set(names)) != len(names):
        raise ValueError('Choose nonempty distinct feature names.')
    combined = len(names) > 1
    if combined and any(name not in config.get('scales', {}) or name not in config.get('weights', {}) for name in names):
        raise ValueError('Combined needs explicit, training-fitted scales and weights for every block.')
    blocks = []
    for name in names:
        values = extracted['features'][name]
        scale = np.asarray(config.get('scales', {}).get(name, 1.), float)
        offset = np.asarray(config.get('offsets', {}).get(name, 0.), float)
        weight = config.get('weights', {}).get(name, 1.)
        if scale.ndim > 1 or offset.ndim > 1 or scale.size not in (1, values.shape[1]) or offset.size not in (1, values.shape[1]):
            raise ValueError('Scale/offset must be scalar or match the feature dimension.')
        if not np.isfinite(scale).all() or (scale <= 0).any() or not np.isfinite(offset).all() or not np.isscalar(weight) or not np.isfinite(weight) or weight <= 0:
            raise ValueError('Scales/weights must be finite positive; offsets finite.')
        blocks.append((values-offset)/scale*weight)
    matrix = np.concatenate(blocks, axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError('Undefined candidate features; select features valid for every window.')
    return matrix


def extract_candidate_features(candidates, feature_config):
    """Reuse the already resized/denoised V4 image; do not preprocess twice."""
    extracted = extract_features(candidates['work_image_bgr'], candidates['boxes'],
        feature_names=feature_config['names'], apply_denoise=False,
        histogram_bins=feature_config.get('histogram_bins'))
    return _feature_matrix(extracted, feature_config)


def match_candidates(candidate_features, references, thresholds, feature_config):
    """References must already use the same representation/transforms/metric."""
    if 'feature_config' not in references or references['feature_config'] != feature_config:
        raise ValueError('References must include the identical feature_config used for calibration.')
    return match_features(references['features'], references['labels'], candidate_features,
                          thresholds, metric=feature_config.get('metric', 'l1'))


def detect_image(image, references, thresholds, feature_config, roi_config=None, *,
                 image_id=None, nms_iou=.5):
    """Return serializable detections and stage counts/times, without consulting GT."""
    started = time.perf_counter()
    original = load_image(image)
    loaded = time.perf_counter()
    candidates = generate_candidates(original, roi_config)
    generated = time.perf_counter()
    matrix = extract_candidate_features(candidates, feature_config)
    extracted = time.perf_counter()
    matches = match_candidates(matrix, references, thresholds, feature_config)
    matched = time.perf_counter()
    # NMS uses float original coordinates, avoiding quantization-induced suppression.
    height, width = original.shape[:2]
    ww, wh = candidates['work_size']
    boxes = candidates['boxes'].astype(float) * [width/ww, height/wh, width/ww, height/wh]
    final = postprocess_detections(boxes, matches, iou_threshold=nms_iou)
    detections = []
    for box, label, distance, source in zip(final['boxes'], final['labels'], final['distances'], final['source_indices']):
        x0, y0, x1, y1 = box
        detections.append(dict(bbox=[float(x0), float(y0), float(x1-x0), float(y1-y0)],
            **{'class': str(label)}, distance=float(distance), accepted=True, candidate_index=int(source)))
    ended = time.perf_counter()
    identity = image_id if image_id is not None else (str(Path(image).as_posix()) if isinstance(image, (str, Path)) else 'image')
    return dict(image_id=str(identity), image_size=[width, height], bbox_format='xywh',
        coordinate_space='original', detections=detections,
        counts=dict(coarse_rois=len(candidates['coarse_boxes']), candidates=len(boxes),
                    accepted_before_nms=int(matches['accepted'].sum()), final=len(detections)),
        runtime_seconds=dict(load=loaded-started, candidates=generated-loaded,
            extraction=extracted-generated, matching=matched-extracted,
            postprocessing=ended-matched, total=ended-started),
        configuration=dict(features=feature_config, roi_method=candidates['method'],
            work_size=candidates['work_size'], nms_iou=nms_iou,
            roi=roi_config if roi_config is not None else dict(v4=load_roi_config(), windows=load_window_config(), method='S2')))


def visualize_detections(image, result, output_path=None):
    """Draw only final predictions on the original image; no GT needed."""
    canvas = load_image(image)
    if list(result['image_size']) != [canvas.shape[1], canvas.shape[0]]:
        raise ValueError('Result/image size mismatch.')
    for detection in result['detections']:
        x, y, w, h = detection['bbox']
        p0, p1 = (round(x), round(y)), (round(x+w), round(y+h))
        cv2.rectangle(canvas, p0, p1, (0, 220, 0), 2)
        text = f"{detection['class']} d={detection['distance']:.4g}"
        cv2.putText(canvas, text, (p0[0], max(15, p0[1]-5)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 220, 0), 1, cv2.LINE_AA)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), canvas):
            raise OSError(f'Could not write {output_path}')
    return canvas


def save_detections(results, output_path):
    """Save JSON or CSV; CSV includes empty-image rows so false negatives survive."""
    results = [results] if isinstance(results, dict) else list(results)
    ids = [r['image_id'] for r in results]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate image IDs.')
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == '.json':
        path.write_text(json.dumps(dict(bbox_format='xywh', coordinate_space='original', images=results), indent=2, allow_nan=False))
    elif path.suffix.lower() == '.csv':
        with path.open('w', newline='') as stream:
            fields = ['image_id', 'x', 'y', 'width', 'height', 'class', 'distance', 'accepted', 'metadata_json']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for result in results:
                metadata = json.dumps({k: v for k, v in result.items() if k not in ('detections', 'image_id')}, allow_nan=False)
                for detection in result['detections'] or [None]:
                    row = dict(image_id=result['image_id'], metadata_json=metadata)
                    if detection is not None:
                        row.update(zip(('x', 'y', 'width', 'height'), detection['bbox']))
                        row.update({k: detection[k] for k in ('class', 'distance', 'accepted')})
                    writer.writerow(row)
    else:
        raise ValueError('Output must be .json or .csv.')


def detect_dataset(images, references, thresholds, feature_config, roi_config=None, *,
                   output_path=None, visualization_dir=None, nms_iou=.5):
    """Directory -> recursive relative-path IDs; mapping -> explicit ID:path/array.

    A directory should be the dataset root or split root used by evaluation IDs.
    No annotations, data splits or calibration are loaded here.
    """
    if isinstance(images, (str, Path)):
        root = Path(images)
        if not root.is_dir():
            raise ValueError('Pass an image directory or an ID-to-image mapping.')
        items = {p.relative_to(root).as_posix(): p for p in sorted(root.rglob('*'))
                 if p.is_file() and p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')}
    else:
        items = dict(images)
    results = []
    for index, (identity, image) in enumerate(items.items()):
        result = detect_image(image, references, thresholds, feature_config, roi_config,
                              image_id=identity, nms_iou=nms_iou)
        results.append(result)
        if visualization_dir is not None:
            # Index prevents basename collisions and untrusted ID path traversal.
            visualize_detections(image, result, Path(visualization_dir)/f'{index:06d}.png')
    if output_path is not None:
        save_detections(results, output_path)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', required=True, help='Directory to process; IDs are relative paths.')
    parser.add_argument('--model', required=True, help='JSON with references, thresholds, feature_config and roi_config.')
    parser.add_argument('--output', required=True, help='Predictions .json or .csv.')
    parser.add_argument('--visualizations')
    parser.add_argument('--nms-iou', type=float, default=.5)
    args = parser.parse_args()
    model = json.loads(Path(args.model).read_text())
    if 'roi_config' not in model:
        raise ValueError('Saved model must specify the ROI/preprocessing configuration used in validation.')
    detect_dataset(args.images, model['references'], model['thresholds'], model['feature_config'],
        model['roi_config'], output_path=args.output, visualization_dir=args.visualizations, nms_iou=args.nms_iou)


if __name__ == '__main__':
    main()
