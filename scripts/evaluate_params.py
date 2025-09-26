from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2

from src.detection.config import DetectorConfig
from src.detection.person_detector import PersonDetector


@dataclass
class ParameterSet:
    conf: float
    iou: float
    imgsz: int

    def as_tuple(self) -> Tuple[float, float, int]:
        return self.conf, self.iou, self.imgsz


def parse_parameter_sets(
    conf_values: Sequence[float],
    iou_values: Sequence[float],
    imgsz_values: Sequence[int],
) -> List[ParameterSet]:
    params: List[ParameterSet] = []
    for conf in conf_values:
        for iou in iou_values:
            for imgsz in imgsz_values:
                params.append(ParameterSet(conf, iou, imgsz))
    return params


def load_image_paths(metadata_path: Path, limit: int | None = None) -> List[Path]:
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    image_paths = [Path(entry["file_path"]) for entry in metadata]
    if limit is not None:
        image_paths = image_paths[:limit]
    return image_paths


def evaluate_parameters(
    model_path: Path,
    image_paths: Sequence[Path],
    params: Iterable[ParameterSet],
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}

    for param in params:
        config = DetectorConfig(
            model_path=model_path,
            conf_threshold=param.conf,
            iou_threshold=param.iou,
            image_size=param.imgsz,
        )
        detector = PersonDetector(config)

        total_time = 0.0
        total_detections = 0
        frames = 0

        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            start = perf_counter()
            detections = detector.detect(image)
            total_time += perf_counter() - start
            total_detections += len(detections)
            frames += 1

        key = f"conf_{param.conf}_iou_{param.iou}_imgsz_{param.imgsz}"
        if frames == 0:
            results[key] = {
                "frames": 0,
                "avg_time": 0.0,
                "fps": 0.0,
                "avg_detections": 0.0,
            }
        else:
            avg_time = total_time / frames
            results[key] = {
                "frames": frames,
                "avg_time": avg_time,
                "fps": (1.0 / avg_time) if avg_time > 0 else 0.0,
                "avg_detections": total_detections / frames,
            }

    return results


def save_results(results: Dict[str, Dict[str, float]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate detection parameters")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("output/frames/samples/metadata.json"),
        help="Path to frame metadata JSON",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("yolo11n.pt"),
        help="Model path for evaluation",
    )
    parser.add_argument("--limit", type=int, default=20, help="Limit number of frames")
    parser.add_argument(
        "--conf",
        type=float,
        nargs="*",
        default=[0.3, 0.5, 0.7],
        help="Confidence thresholds to evaluate",
    )
    parser.add_argument(
        "--iou",
        type=float,
        nargs="*",
        default=[0.3, 0.45, 0.6],
        help="IoU thresholds to evaluate",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="*",
        default=[416, 640, 832],
        help="Image sizes to evaluate",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/results/parameter_search.json"),
        help="Output path for results",
    )
    args = parser.parse_args()

    params = parse_parameter_sets(args.conf, args.iou, args.imgsz)
    image_paths = load_image_paths(args.metadata, args.limit)
    if not image_paths:
        raise RuntimeError("No image paths found for evaluation")

    results = evaluate_parameters(args.model, image_paths, params)

    for key, stats in results.items():
        print(key)
        print(
            f"  Frames: {stats['frames']} | Avg time: {stats['avg_time'] * 1000:.2f} ms | "
            f"FPS: {stats['fps']:.2f} | Avg detections/frame: {stats['avg_detections']:.2f}"
        )

    save_results(results, args.output)


if __name__ == "__main__":
    main()

