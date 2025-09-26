from __future__ import annotations

import json
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from tqdm.auto import tqdm

from src.ocr.timestamp_ocr import OcrConfig, TimestampOCR
from src.utils.video_utils import get_video_info


@dataclass
class SampledFrame:
    frame_index: int
    timestamp: Optional[datetime]
    file_path: Path

    def to_dict(self) -> dict:
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "file_path": str(self.file_path),
        }


class RandomFrameSampler:
    def __init__(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        sample_count: int = 100,
        seed: int = 42,
        ocr_config: Optional[OcrConfig] = None,
        overwrite: bool = False,
    ) -> None:
        self.video_path = Path(video_path)
        self.output_dir = Path(output_dir)
        self.sample_dir = self.output_dir / "samples"
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        self.sample_count = sample_count
        self.seed = seed
        self.overwrite = overwrite
        self.ocr = TimestampOCR(ocr_config or OcrConfig((900, 30, 350, 50), threshold=-1, invert=False))

    def _choose_frame_indices(self) -> List[int]:
        info = get_video_info(self.video_path)
        total_frames = info.frame_count
        if self.sample_count >= total_frames:
            return list(range(total_frames))
        rng = random.Random(self.seed)
        return sorted(rng.sample(range(total_frames), self.sample_count))

    def _format_filename(self, frame_index: int, timestamp: Optional[datetime]) -> str:
        if timestamp:
            ts_str = timestamp.strftime("%Y-%m-%d_%H-%M-%S")
            return f"sample_{frame_index:06d}_{ts_str}.jpg"
        return f"sample_{frame_index:06d}.jpg"

    def _save_frame(self, frame: np.ndarray, filename: str) -> Path:
        out_path = self.sample_dir / filename
        if out_path.exists() and not self.overwrite:
            return out_path
        cv2.imwrite(str(out_path), frame)
        return out_path

    def sample(self) -> List[SampledFrame]:
        indices = self._choose_frame_indices()
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        samples: List[SampledFrame] = []
        for frame_index in tqdm(indices, desc="Sampling frames", unit="frame"):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue
            timestamp = self.ocr.detect(frame)
            filename = self._format_filename(frame_index, timestamp)
            out_path = self._save_frame(frame, filename)
            samples.append(SampledFrame(frame_index, timestamp, out_path))

        cap.release()
        self._save_metadata(samples)
        return samples

    def _save_metadata(self, samples: List[SampledFrame]) -> None:
        meta_path = self.sample_dir / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump([s.to_dict() for s in samples], f, ensure_ascii=False, indent=2)
