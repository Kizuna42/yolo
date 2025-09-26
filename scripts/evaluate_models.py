from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median, stdev
from time import perf_counter
from typing import Dict, List, Sequence

import cv2

from src.detection.config import DetectorConfig
from src.detection.person_detector import PersonDetector


def load_image_paths(metadata_path: Path, limit: int | None = None) -> List[Path]:
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    image_paths = [Path(entry["file_path"]) for entry in metadata]
    if limit is not None:
        image_paths = image_paths[:limit]
    return image_paths


def evaluate_model(model_path: Path, image_paths: Sequence[Path]) -> Dict[str, float]:
    config = DetectorConfig(model_path=model_path, conf_threshold=0.3)
    detector = PersonDetector(config)

    inference_times: List[float] = []
    detection_counts: List[int] = []

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        start = perf_counter()
        detections = detector.detect(image)
        elapsed = perf_counter() - start
        inference_times.append(elapsed)
        detection_counts.append(len(detections))

    if not inference_times:
        return {
            "frames": 0,
            "avg_time": 0.0,
            "median_time": 0.0,
            "fps": 0.0,
            "avg_count": 0.0,
            "count_std": 0.0,
        }

    avg_time = mean(inference_times)
    return {
        "frames": len(inference_times),
        "avg_time": avg_time,
        "median_time": median(inference_times),
        "fps": (1.0 / avg_time) if avg_time > 0 else 0.0,
        "avg_count": mean(detection_counts),
        "count_std": stdev(detection_counts) if len(detection_counts) > 1 else 0.0,
    }


def save_results(results: Dict[str, Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate YOLO models on sample frames")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("output/frames/samples/metadata.json"),
        help="Path to metadata JSON containing frame file paths",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        default=["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"],
        help="YOLO model names or paths to evaluate",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of frames")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/results/model_comparison.json"),
        help="Path to save evaluation results",
    )
    args = parser.parse_args()

    image_paths = load_image_paths(args.metadata, args.limit)
    if not image_paths:
        raise RuntimeError("No image paths found for evaluation")

    results: Dict[str, Dict[str, float]] = {}
    for model in args.models:
        stats = evaluate_model(Path(model), image_paths)
        results[str(model)] = stats
        print(f"Model: {model}")
        print(
            f"  Frames: {stats['frames']} | Avg time: {stats['avg_time'] * 1000:.2f} ms | "
            f"Median time: {stats['median_time'] * 1000:.2f} ms | FPS: {stats['fps']:.2f}"
        )
        print(
            f"  Avg detections/frame: {stats['avg_count']:.2f} | Count std: {stats['count_std']:.2f}"
        )

    save_results(results, args.output)


if __name__ == "__main__":
    main()

