from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.evaluation.evaluator import ModelEvaluator


def create_metadata(tmp_path: Path) -> Path:
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    entries = []
    for i in range(3):
        image_path = images_dir / f"frame_{i}.jpg"
        image_path.touch()
        entries.append(
            {
                "frame_index": i,
                "timestamp": f"2025-08-27T08:0{i}:00",
                "file_path": str(image_path),
            }
        )
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(entries), encoding="utf-8")
    return metadata_path


def create_ground_truth(tmp_path: Path) -> Path:
    csv_path = tmp_path / "ground_truth.csv"
    csv_path.write_text(
        "timestamp,Z13-09,Z13-08,Z13-07\n"
        "2025-08-27T08:00:00,1,2,1\n"
        "2025-08-27T08:01:00,0,1,0\n"
        "2025-08-27T08:02:00,1,0,0\n",
        encoding="utf-8",
    )
    return csv_path


@patch("src.evaluation.evaluator.PersonDetector")
def test_model_evaluator(mock_detector_cls: MagicMock, tmp_path: Path) -> None:
    metadata_path = create_metadata(tmp_path)
    gt_path = create_ground_truth(tmp_path)

    mock_detector = MagicMock()
    mock_detector.run_on_image.side_effect = [MagicMock(class_ids=[0, 0, 1]), MagicMock(class_ids=[0]), MagicMock(class_ids=[])]
    mock_detector_cls.return_value = mock_detector

    evaluator = ModelEvaluator(metadata_path=metadata_path, ground_truth_csv=gt_path)

    def fake_run_detection(path: Path) -> int:
        if path.name == "frame_0.jpg":
            return 2
        if path.name == "frame_1.jpg":
            return 1
        return 0

    with patch.object(evaluator, "run_detection", side_effect=fake_run_detection):
        result = evaluator.evaluate()

    metrics = result["metrics"]
    summaries = result["summaries"]

    assert metrics.precision >= 0.0
    assert metrics.recall >= 0.0
    assert len(summaries) == 3

