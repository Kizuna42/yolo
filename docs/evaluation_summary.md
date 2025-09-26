# 評価サマリー (2025-09-26)

## 実装状況

- `src/evaluation/metrics.py` で Precision / Recall / F1 / AP を算出するユーティリティを実装。
- `src/evaluation/evaluator.py` でメタデータ + GT CSV に基づく検出評価パイプラインを構築。
- `scripts/run_evaluation.py` によりコマンドラインから評価実行可能。
- `tests/test_metrics.py`, `tests/test_evaluator.py` で単体テストを整備済み。

## 評価設定

- モデル: `yolo11m.pt`
- パラメータ: `conf=0.3`, `IoU=0.45`, `imgsz=832`
- メタデータ: `output/frames/samples/metadata.json`
- GT: `output/results/label_data.csv` (ラベリング済み)

## 現時点の課題

- GT CSV に記録された時間帯の多くでラベル値が 0 のため評価対象フレームが抽出されず、現在の指標は 0 となる。
- 評価を有効化するには、`label_data.csv` に人物数の手動ラベリングが必要。
- `output/detections/` 以下には既に推論済みの画像が多数存在するため、ラベル付けは比較的容易に行える想定。

## 推奨アクション

1. `output/detections/samples/` を参照し、各フレームにおける人物数を `label_data.csv` に反映。
2. `scripts/run_evaluation.py` を再実行し、Precision/Recall/F1 の実値と AP を計測。
3. 評価結果を `docs/` 配下に追記し、フェーズ 4 のチェックボックスを更新。

## 最新評価結果 (2025-09-26)

- 実行コマンド: `scripts/run_evaluation.py --model yolo11m.pt --conf 0.3 --iou 0.45 --imgsz 832`
- 評価フレーム数: 33
- Precision: 0.952
- Recall: 0.590
- F1 score: 0.728
- Average Precision (AP): 0.750

### 所見

- Precision が高く False Positive は抑えられている一方、Recall が 0.59 程度に留まり検出漏れが残存。
- GT と整合したフレーム数は 33。ラベルのカバレッジ拡充により評価の信頼度向上が見込める。
