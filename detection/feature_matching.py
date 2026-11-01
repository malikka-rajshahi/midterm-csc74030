"""Feature-only nearest-reference matching and validation-threshold rejection.

Inputs are already extracted, consistently normalized features. No preprocessing,
reference fitting, threshold calibration, window generation, NMS, I/O or fusion.
One call compares one feature representation with one distance metric.
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


def match_features(reference_features, reference_labels, candidate_features, thresholds,
                   metric='l1', background_label='Background',
                   candidate_batch_size=256, reference_batch_size=64):
    """Choose nearest reference CLASS, then reject by its supplied validation threshold.

    reference_features: (R,D) train-GT templates/prototypes, or (R,) scalar values.
    reference_labels: R nonempty class strings; multiple templates/class are allowed.
    candidate_features: (N,D) extracted S2-window features, or (N,) scalar values.
    thresholds: validation-selected scalar, or {class: threshold}, for THIS feature
        and metric. Thresholds are never selected from candidate/query features.

    A class distance is the minimum over its references, so the globally nearest
    reference gives the nearest class. Equal distances retain the first supplied
    reference. Accept iff nearest_distance <= its class threshold; otherwise label
    Background. Do not switch to a farther class merely because it has a wider
    threshold. One candidate receives at most one class label.

    Return aligned labels, accepted, nearest_classes, distances, reference_indices.
    Distances are not confidence probabilities. No feature fusion is performed.
    """
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
    limits = _thresholds(thresholds, labels)
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
    accepted = best <= limits[closest]
    nearest = labels[closest]
    prediction = np.full(len(candidates), background_label, dtype=object)
    prediction[accepted] = nearest[accepted]
    return dict(labels=prediction, accepted=accepted, nearest_classes=nearest,
                distances=best, reference_indices=closest)
