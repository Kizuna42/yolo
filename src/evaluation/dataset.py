from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from pandas import DataFrame


@dataclass(slots=True)
class FrameRecord:
    frame_index: int
    timestamp: datetime
    file_path: Path


def parse_metadata(metadata_path: Path) -> List[FrameRecord]:
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)
    records: List[FrameRecord] = []
    for entry in entries:
        ts_str = entry.get("timestamp")
        if not ts_str:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        records.append(
            FrameRecord(
                frame_index=int(entry.get("frame_index", -1)),
                timestamp=ts,
                file_path=Path(entry.get("file_path", "")),
            )
        )
    return records


def load_ground_truth_csv(csv_path: Path) -> DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    df = df.rename(columns={df.columns[0]: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True)
    df = df.fillna(0).astype({col: int for col in df.columns if col != "timestamp"})
    return df.set_index("timestamp")


def total_counts_per_timestamp(df: DataFrame) -> Dict[datetime, int]:
    totals = df.sum(axis=1)
    return {idx.to_pydatetime(): int(value) for idx, value in totals.items()}


def align_frames_to_ground_truth(
    frames: Sequence[FrameRecord],
    gt_totals: Dict[datetime, int],
    tolerance_minutes: int = 5,
) -> List[Tuple[FrameRecord, Optional[datetime]]]:
    if not frames:
        return []
    sorted_gt = sorted(gt_totals.keys())
    results: List[Tuple[FrameRecord, Optional[datetime]]] = []
    tolerance = pd.Timedelta(minutes=tolerance_minutes)
    for frame in frames:
        matches = [ts for ts in sorted_gt if abs(pd.Timestamp(ts) - frame.timestamp) <= tolerance]
        matched_ts = min(matches, key=lambda ts: abs(pd.Timestamp(ts) - frame.timestamp)) if matches else None
        results.append((frame, matched_ts))
    return results


def align_frames_to_ground_truth_df(
    frames: Sequence[FrameRecord],
    gt_totals: Dict[datetime, int],
    tolerance_minutes: int = 5,
) -> DataFrame:
    rows = []
    aligned = align_frames_to_ground_truth(frames, gt_totals, tolerance_minutes)
    for frame, matched_ts in aligned:
        total_gt = gt_totals.get(matched_ts, 0) if matched_ts else None
        rows.append(
            {
                "frame_index": frame.frame_index,
                "frame_timestamp": frame.timestamp,
                "file_path": str(frame.file_path),
                "matched_timestamp": matched_ts,
                "ground_truth_total": total_gt,
            }
        )
    return pd.DataFrame(rows)
