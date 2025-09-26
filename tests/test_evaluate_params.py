from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts.evaluate_params import (
    ParameterSet,
    evaluate_parameters,
    load_image_paths,
    parse_parameter_sets,
    save_results,
)


def test_parse_parameter_sets() -> None:
    params = parse_parameter_sets([0.3, 0.5], [0.4], [416, 640])
    assert len(params) == 4
    assert params[0] == ParameterSet(conf=0.3, iou=0.4, imgsz=416)


def test_load_image_paths(tmp_path: Path) -> None:
    metadata_path = tmp_path / "metadata.json"
    data = [{"file_path": f"frame_{i}.jpg"} for i in range(3)]
    metadata_path.write_text(json.dumps(data), encoding="utf-8")

    paths = load_image_paths(metadata_path, limit=2)
    assert len(paths) == 2
    assert paths[0].name == "frame_0.jpg"


@patch("scripts.evaluate_params.PersonDetector")
@patch("scripts.evaluate_params.cv2.imread", return_value=MagicMock())
def test_evaluate_parameters(
    mock_imread: MagicMock, mock_detector_cls: MagicMock, tmp_path: Path
) -> None:
    image_paths = [tmp_path / f"img_{i}.jpg" for i in range(3)]

    detection_mocks = []
    for count in (1, 2, 3):
        detection = MagicMock()
        detection.__len__.return_value = count
        detection_mocks.append(detection)

    mock_detector = MagicMock()
    mock_detector.detect.side_effect = detection_mocks
    mock_detector_cls.return_value = mock_detector

    params = [ParameterSet(conf=0.5, iou=0.45, imgsz=640)]

    with patch("scripts.evaluate_params.perf_counter", side_effect=[0.0, 0.1, 0.2, 0.4, 0.5, 0.9]):
        results = evaluate_parameters(Path("model.pt"), image_paths, params)

    key = "conf_0.5_iou_0.45_imgsz_640"
    assert key in results
    assert results[key]["frames"] == len(image_paths)
    assert results[key]["avg_detections"] > 0


def test_save_results(tmp_path: Path) -> None:
    output_path = tmp_path / "out" / "params.json"
    data = {"param": {"frames": 5}}
    save_results(data, output_path)

    assert output_path.exists()
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == data

