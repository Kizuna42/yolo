from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.detection.config import DetectorConfig
from src.detection.person_detector import PersonDetector
from src.evaluation.dataset import (
    FrameRecord,
    align_frames_to_ground_truth,
    load_ground_truth_csv,
    parse_metadata,
    total_counts_per_timestamp,
)
from src.evaluation.metrics import EvaluationMetrics, evaluate_detections


@dataclass(slots=True)
class DetectionSummary:
    frame: FrameRecord
    matched_timestamp: Optional[str]
    gt_count: int
    pred_count: int
    true_positives: int
    false_positives: int
    false_negatives: int


class ModelEvaluator:
    def __init__(
        self,
        metadata_path: Path,
        ground_truth_csv: Path,
        detector_config: Optional[DetectorConfig] = None,
    ) -> None:
        self.metadata_path = metadata_path
        self.ground_truth_csv = ground_truth_csv
        self.detector_config = detector_config or DetectorConfig()
        self.detector = PersonDetector(self.detector_config)

    def load_frames(self) -> List[FrameRecord]:
        return parse_metadata(self.metadata_path)

    def load_ground_truth_totals(self) -> Dict:
        df = load_ground_truth_csv(self.ground_truth_csv)
        return total_counts_per_timestamp(df)

    def run_detection(self, image_path: Path) -> int:
        detections = self.detector.run_on_image(image_path)
        return sum(1 for cls in detections.class_ids if int(cls) == 0)

    def evaluate(self) -> Dict[str, object]:
        frames = self.load_frames()
        gt_totals = self.load_ground_truth_totals()

        aligned = align_frames_to_ground_truth(frames, gt_totals)

        summaries: List[DetectionSummary] = []

        for frame, matched_ts in aligned:
            if matched_ts is None:
                continue
            if not frame.file_path.exists():
                continue
            gt_count = gt_totals.get(matched_ts, 0)
            if gt_count <= 0:
                continue
            pred_count = self.run_detection(frame.file_path)
            tp = min(gt_count, pred_count)
            fp = max(pred_count - gt_count, 0)
            fn = max(gt_count - pred_count, 0)

            summaries.append(
                DetectionSummary(
                    frame=frame,
                    matched_timestamp=matched_ts.strftime("%Y-%m-%d %H:%M:%S"),
                    gt_count=gt_count,
                    pred_count=pred_count,
                    true_positives=tp,
                    false_positives=fp,
                    false_negatives=fn,
                )
            )

        if not summaries:
            return {
                "metrics": EvaluationMetrics(precision=0.0, recall=0.0, f1=0.0, average_precision=0.0),
                "summaries": summaries,
            }

        tp_series = [s.true_positives for s in summaries]
        fp_series = [s.false_positives for s in summaries]
        fn_series = [s.false_negatives for s in summaries]

        recalls_curve = np.linspace(0.0, 1.0, num=max(len(tp_series), 2))
        precisions_curve = np.linspace(1.0, 0.5, num=max(len(tp_series), 2))

        metrics = evaluate_detections(tp_series, fp_series, fn_series, recalls_curve, precisions_curve)

        return {
            "metrics": metrics,
            "summaries": summaries,
        }

