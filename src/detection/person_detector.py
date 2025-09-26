from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

from src.detection.config import DetectorConfig, VisualizationConfig


@dataclass
class DetectionResult:
    boxes: np.ndarray  # (N, 4) xyxy
    scores: np.ndarray  # (N,)
    class_ids: np.ndarray  # (N,)
    names: Sequence[str]

    def __len__(self) -> int:  # pragma: no cover - trivial proxy
        return len(self.boxes)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "boxes": self.boxes.tolist(),
            "scores": self.scores.tolist(),
            "class_ids": self.class_ids.tolist(),
        }


class PersonDetector:
    def __init__(
        self,
        config: DetectorConfig = DetectorConfig(),
        viz_config: VisualizationConfig = VisualizationConfig(),
    ) -> None:
        self.config = config
        self.viz_config = viz_config
        self.model = self._load_model(config.model_path)
        self.names = self.model.names

    def _load_model(self, model_path: Path) -> YOLO:
        if not Path(model_path).exists():
            return YOLO(str(model_path))
        return YOLO(str(model_path))

    def _select_device(self) -> str:
        device = self.config.device
        if device != "auto":
            return device
        if YOLO.is_available("cuda"):
            return "cuda"
        if YOLO.is_available("mps"):
            return "mps"
        return "cpu"

    def detect(self, image: np.ndarray) -> DetectionResult:
        results = self.model.predict(
            source=image,
            device=self._select_device(),
            conf=self.config.conf_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            half=self.config.half_precision,
            classes=self.config.classes,
            verbose=False,
        )

        result = results[0]
        detections = self._parse_result(result)
        return detections

    def _parse_result(self, result: Results) -> DetectionResult:
        boxes_data: Boxes | None = result.boxes
        if boxes_data is None:
            return DetectionResult(
                boxes=np.empty((0, 4)),
                scores=np.empty((0,)),
                class_ids=np.empty((0,)),
                names=self.names,
            )
        boxes = boxes_data.xyxy.cpu().numpy()
        scores = boxes_data.conf.cpu().numpy()
        class_ids = boxes_data.cls.cpu().numpy()
        return DetectionResult(boxes=boxes, scores=scores, class_ids=class_ids, names=self.names)

    def visualize(self, image: np.ndarray, detections: DetectionResult) -> np.ndarray:
        if not self.viz_config.draw_boxes or len(detections) == 0:
            return image
        annotated = image.copy()
        for bbox, score, cls_id in zip(detections.boxes, detections.scores, detections.class_ids):
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if self.viz_config.draw_labels:
                names = detections.names
                class_name = names[int(cls_id)] if isinstance(names, (list, tuple)) and int(cls_id) < len(names) else str(cls_id)
                label = f"{class_name}: {score:.{self.viz_config.score_precision}f}"
                (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated, (x1, y1 - h - baseline), (x1 + w, y1), (0, 255, 0), cv2.FILLED)
                cv2.putText(annotated, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return annotated

    def run_on_image(self, image_path: Path, save_path: Optional[Path] = None) -> DetectionResult:
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        detections = self.detect(image)
        if save_path:
            annotated = self.visualize(image, detections)
            cv2.imwrite(str(save_path), annotated)
        return detections
