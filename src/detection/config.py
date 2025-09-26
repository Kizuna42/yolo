from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


DeviceType = Literal["cpu", "cuda", "mps", "auto"]


@dataclass(slots=True)
class DetectorConfig:
    model_path: Path = Path("yolo11n.pt")
    device: DeviceType = "auto"
    conf_threshold: float = 0.5
    iou_threshold: float = 0.45
    image_size: int = 640
    half_precision: bool = False
    classes: Optional[list[int]] = None


@dataclass(slots=True)
class VisualizationConfig:
    draw_boxes: bool = True
    draw_labels: bool = True
    score_precision: int = 2
