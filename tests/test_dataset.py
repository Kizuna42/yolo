from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.evaluation.dataset import align_frames_to_ground_truth, align_frames_to_ground_truth_df, load_ground_truth_csv, parse_metadata, total_counts_per_timestamp


def create_metadata(tmp_path: Path) -> Path:
    data = [
        {
            "frame_index": 0,
            "timestamp": "2025-08-27T08:05:00",
            "file_path": "output/frames/sample_0.jpg",
        },
        {
            "frame_index": 1,
            "timestamp": "2025-08-27T08:10:00",
            "file_path": "output/frames/sample_1.jpg",
        },
    ]
    path = tmp_path / "metadata.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def create_ground_truth(tmp_path: Path) -> Path:
    csv_path = tmp_path / "label_data.csv"
    csv_path.write_text(
        "timestamp,Z1,Z2\n2025-08-27 08:05:00,1,2\n2025-08-27 08:10:00,0,1\n",
        encoding="utf-8",
    )
    return csv_path


def test_parse_metadata(tmp_path: Path) -> None:
    metadata_path = create_metadata(tmp_path)
    records = parse_metadata(metadata_path)
    assert len(records) == 2
    assert records[0].timestamp == datetime.fromisoformat("2025-08-27T08:05:00")


def test_load_ground_truth_csv(tmp_path: Path) -> None:
    csv_path = create_ground_truth(tmp_path)
    df = load_ground_truth_csv(csv_path)
    assert isinstance(df, pd.DataFrame)
    assert df.loc["2025-08-27 08:05:00"].sum() == 3


def test_total_counts_per_timestamp(tmp_path: Path) -> None:
    csv_path = create_ground_truth(tmp_path)
    df = load_ground_truth_csv(csv_path)
    totals = total_counts_per_timestamp(df)
    assert totals[datetime(2025, 8, 27, 8, 5, 0)] == 3


def test_align_frames_to_ground_truth(tmp_path: Path) -> None:
    metadata_path = create_metadata(tmp_path)
    csv_path = create_ground_truth(tmp_path)
    frames = parse_metadata(metadata_path)
    df = load_ground_truth_csv(csv_path)
    totals = total_counts_per_timestamp(df)

    aligned = align_frames_to_ground_truth(frames, totals, tolerance_minutes=5)
    assert len(aligned) == 2
    _, ts = aligned[0]
    assert ts == datetime(2025, 8, 27, 8, 5, 0)


def test_align_frames_to_ground_truth_df(tmp_path: Path) -> None:
    metadata_path = create_metadata(tmp_path)
    csv_path = create_ground_truth(tmp_path)
    frames = parse_metadata(metadata_path)
    df = load_ground_truth_csv(csv_path)
    totals = total_counts_per_timestamp(df)

    df_aligned = align_frames_to_ground_truth_df(frames, totals)
    assert len(df_aligned) == 2
    assert df_aligned.iloc[0]["matched_timestamp"] == datetime(2025, 8, 27, 8, 5, 0)

