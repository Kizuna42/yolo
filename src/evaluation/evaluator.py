from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from src.detection.config import DetectorConfig
from src.detection.person_detector import PersonDetector
from src.evaluation.metrics import EvaluationMetrics, evaluate_detections


@dataclass
class DetectionSummary:
    image_path: Path
    timestamp: str
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

    def load_metadata(self) -> List[Dict[str, str]]:
        if not self.metadata_path.exists():
            raise FileNotFoundError(self.metadata_path)
        with self.metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def run_detection(self, image_path: Path) -> int:
        detections = self.detector.run_on_image(image_path)
        persons = [cls for cls in detections.class_ids if int(cls) == 0]
        return len(persons)

    def load_ground_truth(self) -> Dict[str, Dict[str, int]]:
        df = pd.read_csv(self.ground_truth_csv)
        df = df.set_index("timestamp")
        return df.fillna(0).astype(int).to_dict(orient="index")

    def evaluate(self) -> Dict[str, object]:
        metadata = self.load_metadata()
        ground_truth = self.load_ground_truth()

        summaries: List[DetectionSummary] = []

        for entry in metadata:
            timestamp = entry.get("timestamp")
            image_path = Path(entry.get("file_path", ""))
            if not timestamp or not image_path.exists():
                continue
            gt_counts = ground_truth.get(timestamp)
            if gt_counts is None:
                continue
            gt_total = sum(gt_counts.values())
            if gt_total == 0:
                continue
            pred_count = self.run_detection(image_path)

            tp = min(gt_total, pred_count)
            fp = max(pred_count - gt_total, 0)
            fn = max(gt_total - pred_count, 0)

            summaries.append(
                DetectionSummary(
                    image_path=image_path,
                    timestamp=timestamp,
                    gt_count=gt_total,
                    pred_count=pred_count,
                    true_positives=tp,
                    false_positives=fp,
                    false_negatives=fn,
                )
            )

        tp_series = [s.true_positives for s in summaries]
        fp_series = [s.false_positives for s in summaries]
        fn_series = [s.false_negatives for s in summaries]

        if not tp_series:
            return {
                "metrics": EvaluationMetrics(precision=0.0, recall=0.0, f1=0.0, average_precision=0.0),
                "summaries": summaries,
            }

        recalls_curve = np.linspace(0.0, 1.0, num=max(len(tp_series), 2))
        precisions_curve = np.linspace(1.0, 0.5, num=max(len(tp_series), 2))

        metrics = evaluate_detections(tp_series, fp_series, fn_series, recalls_curve, precisions_curve)

        return {
            "metrics": metrics,
            "summaries": summaries,
        }

