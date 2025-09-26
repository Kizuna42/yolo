from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

from scripts.evaluate_models import evaluate_model, load_image_paths, save_results


def test_load_image_paths(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    data = [
        {"file_path": str(tmp_path / "frame_1.jpg")},
        {"file_path": str(tmp_path / "frame_2.jpg")},
    ]
    metadata_path.write_text(json.dumps(data), encoding="utf-8")

    paths = load_image_paths(metadata_path)
    assert len(paths) == 2
    assert paths[0].name == "frame_1.jpg"


def test_load_image_paths_with_limit(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    data = [{"file_path": f"frame_{i}.jpg"} for i in range(5)]
    metadata_path.write_text(json.dumps(data), encoding="utf-8")

    paths = load_image_paths(metadata_path, limit=2)
    assert len(paths) == 2


@patch("scripts.evaluate_models.PersonDetector")
def test_evaluate_model(mock_detector_cls: MagicMock, tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    image_paths = []
    for i in range(3):
        image_path = image_dir / f"img_{i}.jpg"
        image_path.touch()
        image_paths.append(image_path)

    detection_mocks = []
    for count in (1, 2, 3):
        detection = MagicMock()
        detection.__len__.return_value = count
        detection_mocks.append(detection)

    mock_detector = MagicMock()
    mock_detector.detect.side_effect = detection_mocks
    mock_detector_cls.return_value = mock_detector

    perf_values = [i * 0.1 for i in range(len(image_paths) * 2)]

    with patch("scripts.evaluate_models.cv2.imread", return_value=np.zeros((10, 10, 3), dtype=np.uint8)):
        with patch("scripts.evaluate_models.perf_counter", side_effect=perf_values):
            stats = evaluate_model(Path("model.pt"), image_paths)

    assert stats["frames"] == len(image_paths)
    assert stats["avg_count"] > 0


def test_save_results(tmp_path: Path) -> None:
    output_path = tmp_path / "results" / "model.json"
    data = {"model": {"frames": 5}}
    save_results(data, output_path)

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == data

