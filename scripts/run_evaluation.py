from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.detection.config import DetectorConfig
from src.evaluation.evaluator import ModelEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run detection evaluation against ground truth")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("output/frames/samples/metadata.json"),
        help="Path to metadata JSON listing frame file paths",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("output/results/label_data.csv"),
        help="CSV file containing ground truth counts",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("yolo11m.pt"),
        help="YOLO model path",
    )
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=832, help="Input image size")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/results/evaluation_summary.json"),
        help="Where to save evaluation metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = DetectorConfig(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        image_size=args.imgsz,
    )

    evaluator = ModelEvaluator(
        metadata_path=args.metadata,
        ground_truth_csv=args.ground_truth,
        detector_config=config,
    )

    result = evaluator.evaluate()

    metrics = result["metrics"]
    summaries = result["summaries"]

    print("Evaluation metrics:")
    print(f"  Precision: {metrics.precision:.3f}")
    print(f"  Recall:    {metrics.recall:.3f}")
    print(f"  F1 score:  {metrics.f1:.3f}")
    print(f"  AP:        {metrics.average_precision:.3f}")
    print(f"Summaries: {len(summaries)} frames evaluated")

    output = {
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "average_precision": metrics.average_precision,
        "frames": len(summaries),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

