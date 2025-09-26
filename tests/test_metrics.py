from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    EvaluationMetrics,
    average_precision,
    confusion_matrix,
    evaluate_detections,
    f1_score,
    mean_average_precision,
    precision,
    recall,
)


def test_precision() -> None:
    tp = [1, 1, 0, 1]
    fp = [0, 1, 0, 0]
    assert precision(tp, fp) == 0.75


def test_recall() -> None:
    tp = [1, 1, 0, 1]
    fn = [0, 0, 1, 0]
    assert recall(tp, fn) == 0.75


def test_f1_score() -> None:
    tp = [1, 1, 0, 1]
    fp = [0, 1, 0, 0]
    fn = [0, 0, 1, 0]
    assert np.isclose(f1_score(tp, fp, fn), 0.75)


def test_confusion_matrix() -> None:
    tp = [1, 0, 1]
    fp = [0, 1, 0]
    fn = [0, 1, 0]
    tn = [1, 1, 1]
    matrix = confusion_matrix(tp, fp, fn, tn)
    expected = np.array([[2, 1], [1, 3]])
    assert np.array_equal(matrix, expected)


def test_average_precision() -> None:
    recalls = [0.0, 0.5, 1.0]
    precisions = [1.0, 0.8, 0.6]
    ap = average_precision(recalls, precisions)
    assert ap > 0.0


def test_mean_average_precision() -> None:
    recalls = [[0.0, 0.5, 1.0], [0.0, 0.4, 0.9]]
    precisions = [[1.0, 0.8, 0.6], [1.0, 0.7, 0.5]]
    map_value = mean_average_precision(recalls, precisions)
    assert 0.0 < map_value <= 1.0


def test_evaluate_detections() -> None:
    tp = [1, 1, 0, 1]
    fp = [0, 1, 0, 0]
    fn = [0, 0, 1, 0]
    recalls = [0.0, 0.5, 1.0]
    precisions = [1.0, 0.8, 0.6]
    metrics = evaluate_detections(tp, fp, fn, recalls, precisions)
    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.precision == 0.75
    assert metrics.recall == 0.75
    assert metrics.f1 > 0.0
    assert metrics.average_precision > 0.0

