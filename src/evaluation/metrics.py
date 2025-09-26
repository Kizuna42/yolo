from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np

try:
    from numpy.typing import NDArray
except ImportError:  # pragma: no cover - fallback for older numpy
    NDArray = np.ndarray  # type: ignore


def _validate_non_negative(values: Sequence[int]) -> None:
    if any(v < 0 for v in values):
        raise ValueError("Values must be non-negative")


def _as_numpy(array: Sequence[float] | NDArray[np.float64]) -> NDArray[np.float64]:
    if isinstance(array, np.ndarray):
        return array.astype(np.float64)
    return np.asarray(array, dtype=np.float64)


def precision(tp: Sequence[int], fp: Sequence[int]) -> float:
    _validate_non_negative(tp)
    _validate_non_negative(fp)
    tp_sum = np.sum(tp)
    fp_sum = np.sum(fp)
    denom = tp_sum + fp_sum
    if denom == 0:
        return 0.0
    return float(tp_sum / denom)


def recall(tp: Sequence[int], fn: Sequence[int]) -> float:
    _validate_non_negative(tp)
    _validate_non_negative(fn)
    tp_sum = np.sum(tp)
    fn_sum = np.sum(fn)
    denom = tp_sum + fn_sum
    if denom == 0:
        return 0.0
    return float(tp_sum / denom)


def f1_score(tp: Sequence[int], fp: Sequence[int], fn: Sequence[int]) -> float:
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    if prec + rec == 0:
        return 0.0
    return float(2 * prec * rec / (prec + rec))


def confusion_matrix(tp: Sequence[int], fp: Sequence[int], fn: Sequence[int], tn: Sequence[int]) -> np.ndarray:
    _validate_non_negative(tp)
    _validate_non_negative(fp)
    _validate_non_negative(fn)
    _validate_non_negative(tn)
    return np.array(
        [
            [np.sum(tp), np.sum(fn)],
            [np.sum(fp), np.sum(tn)],
        ],
        dtype=np.int64,
    )


def average_precision(recalls: Sequence[float], precisions: Sequence[float]) -> float:
    r = _as_numpy(recalls)
    p = _as_numpy(precisions)
    if r.size == 0 or p.size == 0:
        return 0.0
    order = np.argsort(r)
    r_sorted = r[order]
    p_sorted = p[order]
    r_unique, indices = np.unique(r_sorted, return_index=True)
    p_max = np.maximum.accumulate(p_sorted[::-1])[::-1]
    p_unique = p_max[indices]
    area = np.trapezoid(p_unique, r_unique)
    return float(area)


def mean_average_precision(recalls_per_class: Iterable[Sequence[float]], precisions_per_class: Iterable[Sequence[float]]) -> float:
    aps: List[float] = []
    for recalls, precisions in zip(recalls_per_class, precisions_per_class):
        aps.append(average_precision(recalls, precisions))
    if not aps:
        return 0.0
    return float(np.mean(aps))


@dataclass
class EvaluationMetrics:
    precision: float
    recall: float
    f1: float
    average_precision: float


def evaluate_detections(
    true_positives: Sequence[int],
    false_positives: Sequence[int],
    false_negatives: Sequence[int],
    recalls_curve: Sequence[float],
    precisions_curve: Sequence[float],
) -> EvaluationMetrics:
    prec = precision(true_positives, false_positives)
    rec = recall(true_positives, false_negatives)
    f1 = f1_score(true_positives, false_positives, false_negatives)
    ap = average_precision(recalls_curve, precisions_curve)
    return EvaluationMetrics(precision=prec, recall=rec, f1=f1, average_precision=ap)

