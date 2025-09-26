from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2


@dataclass
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_seconds: float


def get_video_info(video_path: str | Path) -> VideoInfo:
    video_path = str(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    cap.release()
    return VideoInfo(width, height, fps, frame_count, duration)


def extract_frame(video_path: str | Path, frame_index: int) -> Optional[cv2.Mat]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame
