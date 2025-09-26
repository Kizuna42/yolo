from __future__ import annotations

import argparse
from pathlib import Path

from src.detection.config import DetectorConfig
from src.detection.person_detector import PersonDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv11 person detection on images")
    parser.add_argument("inputs", nargs="+", type=Path, help="Image files or directories")
    parser.add_argument("--model", type=Path, default=Path("yolo11n.pt"), help="Path to YOLO model")
    parser.add_argument("--output", type=Path, default=Path("output/detections/batch"), help="Output directory for annotated images")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"], help="Inference device")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--half", action="store_true", help="Use half precision")
    parser.add_argument("--classes", type=int, nargs="*", default=None, help="Filter classes (IDs)")
    parser.add_argument("--no-annotate", action="store_true", help="Skip saving annotated images")
    return parser.parse_args()


def iter_images(paths: list[Path]):
    for path in paths:
        if path.is_dir():
            for image_path in sorted(path.glob("**/*")):
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    yield image_path
        elif path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            yield path


def main() -> None:
    args = parse_args()
    output_dir = args.output
    if not args.no_annotate:
        output_dir.mkdir(parents=True, exist_ok=True)

    config = DetectorConfig(
        model_path=args.model,
        device=args.device,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        image_size=args.imgsz,
        half_precision=args.half,
        classes=args.classes,
    )

    detector = PersonDetector(config)

    for image_path in iter_images([Path(p) for p in args.inputs]):
        save_path = None if args.no_annotate else output_dir / f"annotated_{image_path.name}"
        detections = detector.run_on_image(image_path, save_path)
        print(image_path, len(detections))


if __name__ == "__main__":
    main()
