"""Separate array APIs for references, validation calibration, matching and NMS.

Feature extraction/preprocessing, dataset splits, I/O and fusion remain external.
Each call compares one consistent feature representation with one distance metric.
"""
from collections.abc import Mapping
import numpy as np

METRICS = ('absolute', 'l1', 'squared_l2', 'chi_square_max')


def _matrix(features, name):
    """An (N,) scalar-statistic array becomes (N,1), never one long vector."""
    values = np.asarray(features, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError(f'{name} must be finite (N,D), or (N,) for scalar statistics.')
    return values


def _validate_metric(candidates, references, metric):
    if metric not in METRICS:
        raise ValueError(f'metric must be one of {METRICS}.')
    if candidates.shape[1] != references.shape[1]:
        raise ValueError('Candidate and reference feature dimensions must match.')
    if metric == 'absolute' and candidates.shape[1] != 1:
        raise ValueError('absolute is for a single statistic; use l1 for a vector.')
    if metric == 'chi_square_max' and ((candidates < 0).any() or (references < 0).any()):
        raise ValueError('chi_square_max requires nonnegative histogram features.')


def _distance_block(candidates, references, metric):
    try:
        with np.errstate(over='raise', invalid='raise'):
            difference = candidates[:, None, :] - references[None, :, :]
            if metric in ('absolute', 'l1'):
                return np.abs(difference).sum(axis=2)
            if metric == 'squared_l2':
                return np.square(difference).sum(axis=2)
            denominator = np.maximum(candidates[:, None, :], references[None, :, :])
            return np.divide(np.square(difference), denominator,
                             out=np.zeros_like(difference), where=denominator > 0).sum(axis=2)
    except FloatingPointError as error:
        raise ValueError('Feature magnitudes overflow this distance; check feature scaling.') from error


def pairwise_distances(candidate_features, reference_features, metric='l1',
                       candidate_batch_size=256, reference_batch_size=64):
    """Return (N,R) distances; the three vector formulas follow the supplied slide.

    l1 = sum |a-b|; squared_l2 = sum (a-b)^2, WITHOUT sqrt;
    chi_square_max = sum (a-b)^2/max(a,b), with 0/0 contributing zero.
    The last formula is the slide's variant, not a substituted sum-denominator
    chi-square. absolute = |a-b| for Mean/Variance/other scalar statistics.

    Three-dimensional working arrays are batched. The returned N*R matrix itself
    still costs memory; use match_features for large candidate/reference sets.
    Histogram normalization is the extractor's responsibility; this function does
    not rescale color channels, statistics or different feature representations.
    """
    candidates = _matrix(candidate_features, 'candidate_features')
    references = _matrix(reference_features, 'reference_features')
    _validate_metric(candidates, references, metric)
    if len(references)==0:
        raise ValueError('Supply at least one reference.')
    _batch_sizes(candidate_batch_size, reference_batch_size)
    result = np.empty((len(candidates), len(references)), dtype=np.float64)
    for start in range(0, len(candidates), candidate_batch_size):
        for other in range(0, len(references), reference_batch_size):
            result[start:start+candidate_batch_size, other:other+reference_batch_size] = _distance_block(
                candidates[start:start+candidate_batch_size],
                references[other:other+reference_batch_size], metric)
    return result


def _batch_sizes(candidate_size, reference_size):
    for value in (candidate_size, reference_size):
        if not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError('Batch sizes must be positive integers.')


def _thresholds(thresholds, labels):
    if isinstance(thresholds, Mapping):
        missing = set(labels) - set(thresholds)
        if missing:
            raise ValueError(f'Missing validation thresholds for: {sorted(missing)}')
        values = np.array([thresholds[label] for label in labels], dtype=float)
    elif np.asarray(thresholds).ndim == 0:
        values = np.full(len(labels), float(thresholds))
    else:
        raise ValueError('thresholds must be a scalar or a mapping from class to threshold.')
    if values.ndim != 1:
        raise ValueError('Each class threshold must be a scalar.')
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError('Validation thresholds must be finite and nonnegative.')
    return values


def _nearest_matches(reference_features, reference_labels, candidate_features,
                     metric, background_label, candidate_batch_size, reference_batch_size):
    candidates = _matrix(candidate_features, 'candidate_features')
    references = _matrix(reference_features, 'reference_features')
    labels = np.asarray(reference_labels, dtype=object)
    if labels.ndim != 1 or len(labels) != len(references) or len(references) == 0:
        raise ValueError('Supply at least one reference and exactly one label per reference.')
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError('Reference labels must be nonempty strings.')
    if not isinstance(background_label, str) or not background_label or background_label in labels:
        raise ValueError('background_label must be a nonempty string distinct from reference classes.')
    _validate_metric(candidates, references, metric)
    _batch_sizes(candidate_batch_size, reference_batch_size)
    closest = np.full(len(candidates), -1, dtype=np.int64)
    best = np.full(len(candidates), np.inf)
    # Keep only the best reference, not the full potentially huge N*R matrix.
    for start in range(0, len(candidates), candidate_batch_size):
        stop = min(len(candidates), start+candidate_batch_size)
        local_best = np.full(stop-start, np.inf)
        local_index = np.full(stop-start, -1, dtype=np.int64)
        for other in range(0, len(references), reference_batch_size):
            distance = _distance_block(candidates[start:stop], references[other:other+reference_batch_size], metric)
            index = np.argmin(distance, axis=1)
            value = distance[np.arange(len(index)), index]
            improve = value < local_best  # strict update preserves first-reference ties
            local_best[improve] = value[improve]
            local_index[improve] = other+index[improve]
        best[start:stop] = local_best
        closest[start:stop] = local_index
    return labels, closest, best


def match_features(reference_features, reference_labels, candidate_features, thresholds,
                   metric='l1', background_label='Background',
                   candidate_batch_size=256, reference_batch_size=64):
    """Nearest class first, then its validation threshold; otherwise Background.

    Each call compares one feature/metric. Templates may repeat classes; the class
    distance is its minimum reference distance. Ties keep the first reference.
    Never fall back to a farther class with a wider threshold. No refitting/NMS.
    """
    labels, closest, best = _nearest_matches(reference_features, reference_labels,
        candidate_features, metric, background_label, candidate_batch_size, reference_batch_size)
    limits = _thresholds(thresholds, labels)
    accepted = best <= limits[closest]
    nearest = labels[closest]
    prediction = np.full(len(best), background_label, dtype=object)
    prediction[accepted] = nearest[accepted]
    return dict(labels=prediction, accepted=accepted, nearest_classes=nearest,
                distances=best, reference_indices=closest)


def build_references(training_features, training_labels, strategy='mean',
                     background_label='Background'):
    """Prepare train-GT references only; validation/candidate arrays do not belong here.

    mean: one class mean (preserves per-channel histogram normalization).
    median: one class median for a SCALAR statistic only.
    templates: preserve every input reference and its order.
    Mean/median prototypes are class-name sorted; templates retain input order.
    Return features/labels/strategy; no dataset splitting, sampling or fusion.
    """
    features = _matrix(training_features, 'training_features')
    labels = np.asarray(training_labels, dtype=object)
    if labels.ndim != 1 or len(labels) != len(features) or len(features) == 0:
        raise ValueError('Supply nonempty training features and one label per row.')
    if any(not isinstance(label, str) or not label for label in labels):
        raise ValueError('Training labels must be nonempty strings.')
    if not isinstance(background_label, str) or not background_label or background_label in labels:
        raise ValueError('Only target-class GT may build references, not Background.')
    if strategy not in ('mean', 'median', 'templates'):
        raise ValueError('strategy must be mean, median or templates.')
    if strategy == 'median' and features.shape[1] != 1:
        raise ValueError('median is scalar-only; use mean/templates for histogram vectors.')
    if strategy == 'templates':
        reference_features, reference_labels = features.copy(), labels.copy()
    else:
        reference_labels = np.unique(labels)
        reduction = np.mean if strategy == 'mean' else np.median
        reference_features = np.vstack([reduction(features[labels == label], axis=0)
                                        for label in reference_labels])
    return dict(features=reference_features, labels=reference_labels, strategy=strategy)


def calibrate_thresholds(reference_features, reference_labels, validation_features,
                         validation_labels, metric='l1', mode='balanced_accuracy',
                         background_label='Background', positive_quantile=.95,
                         candidate_batch_size=256, reference_batch_size=64):
    """Fit per-class thresholds on separately supplied labeled validation regions.

    balanced_accuracy (default) requires Background examples and each target class.
    Search observed nonnegative distances plus zero, maximizing one-vs-rest
    (TPR+TNR)/2 under the actual nearest-class-first rule. Ties prefer lower limits.
    Misclassified nearest classes remain false negatives, not hidden exclusions.

    positive_quantile is an explicit target-retention-only alternative. Quantiles
    use correctly-nearest positives; no background-rejection ability is claimed
    without Background examples. A class with no such positive cannot calibrate.

    Train/validation/query identity separation is the caller's responsibility.
    Return thresholds, per-class validation diagnostics and feature/metric metadata.
    """
    if mode not in ('balanced_accuracy', 'positive_quantile'):
        raise ValueError('Unknown calibration mode.')
    if not 0 < positive_quantile <= 1:
        raise ValueError('positive_quantile must be in (0,1].')
    ref_labels, indices, best = _nearest_matches(reference_features, reference_labels,
        validation_features, metric, background_label, candidate_batch_size, reference_batch_size)
    truth = np.asarray(validation_labels, dtype=object)
    if truth.ndim != 1 or len(truth) != len(best) or len(truth) == 0:
        raise ValueError('Supply nonempty validation features and aligned labels.')
    if any(not isinstance(label, str) or not label for label in truth):
        raise ValueError('Validation labels must be nonempty strings.')
    classes = np.unique(ref_labels)
    if set(truth) - (set(classes) | {background_label}):
        raise ValueError('Validation labels must be reference classes or Background.')
    background_count = int((truth == background_label).sum())
    if mode == 'balanced_accuracy' and background_count == 0:
        raise ValueError('Background validation examples required; otherwise explicitly use positive_quantile.')
    nearest = ref_labels[indices]
    thresholds, diagnostics = {}, {}
    for label in classes:
        positive = truth == label
        n_positive, n_negative = int(positive.sum()), int((~positive).sum())
        if n_positive == 0:
            raise ValueError(f'No validation target examples for {label}.')
        predicted_class = nearest == label
        if mode == 'balanced_accuracy':
            values = best[predicted_class]
            correct = positive[predicted_class]
            order = np.argsort(values, kind='stable')
            values, correct = values[order], correct[order]
            candidates = np.unique(np.r_[0., values])
            counts = np.searchsorted(values, candidates, side='right')
            tp = np.r_[0, np.cumsum(correct)][counts]
            fp = np.r_[0, np.cumsum(~correct)][counts]
            balanced = .5 * (tp/n_positive + 1 - fp/n_negative)
            threshold = float(candidates[np.argmax(balanced)])
        else:
            values = best[positive & predicted_class]
            if not len(values):
                raise ValueError(f'No correctly-nearest validation positives for {label}.')
            threshold = float(np.quantile(values, positive_quantile, method='higher'))
        accepted = predicted_class & (best <= threshold)
        tp = int((accepted & positive).sum())
        fp = int((accepted & ~positive).sum())
        thresholds[label] = threshold
        diagnostics[label] = dict(target_count=n_positive,negative_count=n_negative,
            correctly_nearest_positive_count=int((positive & predicted_class).sum()),
            true_positive=tp,false_positive=fp,target_retention=tp/n_positive,
            false_positive_rate=fp/n_negative if n_negative else None,
            background_acceptance=float(accepted[truth == background_label].mean()) if background_count else None)
    return dict(thresholds=thresholds,metric=metric,mode=mode,
                feature_dimension=_matrix(reference_features, 'reference_features').shape[1],
                validation_count=len(truth),background_count=background_count,
                positive_quantile=positive_quantile if mode == 'positive_quantile' else None,
                per_class=diagnostics)


def postprocess_detections(boxes, matches, iou_threshold=.5):
    """Same-class NMS AFTER matching, ranking lower distances first; no box growth.

    boxes align with full match_features outputs. Only accepted candidates enter
    NMS; Background is excluded. Different classes are not suppressed together.
    None disables NMS. No Top-K cap or cross-feature distance mixing is applied.
    Return source_indices and aligned boxes/labels/distances/reference_indices.
    """
    original = np.asarray(boxes)
    if original.size == 0:
        original = original.reshape(0, 4)
    coordinates = np.asarray(original, dtype=float)
    if coordinates.ndim != 2 or coordinates.shape[1] != 4 or not np.isfinite(coordinates).all() or (coordinates[:, 2:] <= coordinates[:, :2]).any():
        raise ValueError('boxes must be finite positive-area (N,4) xyxy.')
    labels = np.asarray(matches['labels'], dtype=object)
    distance = np.asarray(matches['distances'], dtype=float)
    accepted = np.asarray(matches['accepted'])
    references = np.asarray(matches['reference_indices'])
    if any(array.shape != (len(coordinates),) for array in (labels, distance, accepted, references)) or accepted.dtype != bool or not np.isfinite(distance).all():
        raise ValueError('Matching arrays must align with boxes and contain finite distances.')
    if iou_threshold is not None and not 0 < iou_threshold <= 1:
        raise ValueError('iou_threshold must be in (0,1] or None.')
    kept = []
    area = np.prod(coordinates[:, 2:] - coordinates[:, :2], axis=1)
    for label in np.unique(labels[accepted]):
        indices = np.flatnonzero(accepted & (labels == label))
        indices = indices[np.argsort(distance[indices], kind='stable')]
        while len(indices):
            first = int(indices[0])
            kept.append(first)
            remaining = indices[1:]
            if iou_threshold is None:
                kept.extend(remaining.tolist())
                break
            intersection = np.maximum(0, np.minimum(coordinates[first, 2:], coordinates[remaining, 2:]) -
                                      np.maximum(coordinates[first, :2], coordinates[remaining, :2])).prod(axis=1)
            iou = intersection / (area[first] + area[remaining] - intersection)
            indices = remaining[iou < iou_threshold]
    kept = np.array(sorted(kept), dtype=np.int64)
    return dict(source_indices=kept, boxes=original[kept].copy(), labels=labels[kept],
                distances=distance[kept], reference_indices=references[kept])
