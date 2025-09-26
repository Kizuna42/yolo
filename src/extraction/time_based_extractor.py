from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from tqdm.auto import tqdm

from src.ocr.timestamp_ocr import OcrConfig, TimestampOCR


@dataclass
class FrameMetadata:
    frame_index: int
    timestamp: datetime
    file_path: Path

    def to_dict(self) -> Dict[str, str]:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp.isoformat(),
            "file_path": str(self.file_path),
        }


class TimeBasedFrameExtractor:
    def __init__(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        interval_minutes: int = 5,
        buffer_seconds: int = 10,
        ocr_config: Optional[OcrConfig] = None,
    ) -> None:
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval = timedelta(minutes=interval_minutes)
        self.buffer = timedelta(seconds=buffer_seconds)
        self.ocr = TimestampOCR(ocr_config or OcrConfig(self._default_roi()))

    def _default_roi(self) -> Tuple[int, int, int, int]:
        # ROI covering timestamp near top-right of 1280x720 frame
        return (900, 30, 350, 50)

    def scan_timestamps(self) -> List[Tuple[int, datetime]]:
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        frame_timestamps: List[Tuple[int, datetime]] = []

        frame_index = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None
        progress = tqdm(
            total=total_frames,
            desc="Scanning timestamps",
            unit="frame",
            leave=False,
        )
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = self.ocr.detect(frame)
            if timestamp:
                frame_timestamps.append((frame_index, timestamp))
            frame_index += 1
            progress.update(1)

        cap.release()
        progress.close()
        return frame_timestamps

    def _generate_target_times(self, start_time: datetime, end_time: datetime) -> Iterable[datetime]:
        current = start_time
        while current <= end_time:
            yield current
            current += self.interval

    def _find_best_frame(
        self,
        frame_timestamps: List[Tuple[int, datetime]],
        target: datetime,
    ) -> Optional[Tuple[int, datetime]]:
        best_frame = None
        min_diff = self.buffer
        for index, ts in frame_timestamps:
            diff = abs(ts - target)
            if diff <= min_diff:
                min_diff = diff
                best_frame = (index, ts)
        return best_frame

    def _save_frame(self, frame: np.ndarray, timestamp: datetime) -> Path:
        filename = f"frame_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
        out_path = self.output_dir / filename
        cv2.imwrite(str(out_path), frame)
        return out_path

    def extract(self) -> List[FrameMetadata]:
        frame_timestamps = self.scan_timestamps()
        if not frame_timestamps:
            raise RuntimeError("No timestamps detected in video")

        start_time = frame_timestamps[0][1]
        end_time = frame_timestamps[-1][1]

        targets = list(self._generate_target_times(start_time, end_time))

        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        metadata: List[FrameMetadata] = []
        for target in tqdm(targets, desc="Extracting target frames", unit="frame", leave=False):
            best = self._find_best_frame(frame_timestamps, target)
            if not best:
                continue
            frame_index, actual_timestamp = best
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue
            out_path = self._save_frame(frame, actual_timestamp)
            metadata.append(FrameMetadata(frame_index, actual_timestamp, out_path))

        cap.release()
        self._save_metadata(metadata)
        return metadata

    def _save_metadata(self, metadata: List[FrameMetadata]) -> None:
        meta_path = self.output_dir / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump([m.to_dict() for m in metadata], f, ensure_ascii=False, indent=2)
    
    def scan_timestamps_partial(self, max_frames: int = 1000, start_frame: int = 0) -> List[Tuple[int, datetime]]:
        """部分的なタイムスタンプスキャン（テスト用）"""
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        # 開始フレームに移動
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_timestamps: List[Tuple[int, datetime]] = []
        frame_index = start_frame
        max_frame = start_frame + max_frames
        
        progress = tqdm(
            total=max_frames,
            desc=f"Scanning timestamps (frames {start_frame}-{max_frame})",
            unit="frame",
            leave=False,
        )
        
        while frame_index < max_frame:
            ret, frame = cap.read()
            if not ret:
                break
            timestamp = self.ocr.detect(frame)
            if timestamp:
                frame_timestamps.append((frame_index, timestamp))
            frame_index += 1
            progress.update(1)

        cap.release()
        progress.close()
        return frame_timestamps