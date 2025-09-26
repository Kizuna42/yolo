# 人物検出 AI PoC プロジェクト

YOLOv11 を用いた人物検出 PoC の実装リポジトリです。タイムスタンプ付き動画から 5 分間隔でフレームを抽出し、OCR による時刻取得と人物検出を行います。本ドキュメントでは現在の実装状況・評価結果・残タスクをまとめます。

## 1. プロジェクト概要

- 入力: `input/merged_moviefiles.mov`（1280x720, 30fps, タイムスタンプ付き）
- 抽出フレーム: `output/frames/`
- 推論結果: `output/detections/`
- 正解データ: `output/results/label_data.csv`（編集禁止）
- 計画書: `docs/人物検出PoC実装計画書.md`

ディレクトリ構成の詳細は `人物検出PoC実装計画書.md` を参照してください。

## 2. 実装状況ハイライト（2025-09-26 現在）

| フェーズ | 内容 | 状況 |
| --- | --- | --- |
| Phase 1 | 環境構築・基盤整備 | ✅ 完了 |
| Phase 2 | データ前処理・時刻抽出 | ✅ 完了 |
| Phase 3 | 人物検出実装・モデル比較 | ✅ 実装完了（パラメータ調整済）。バッチサイズ最適化は未実施 |
| Phase 4 | 精度評価・最適化 | ✅ 指標算出まで完了。推論速度評価はこれから |
| Phase 5 | 統合・最終検証 | ⏳ 未着手 |

### 2.1 主なスクリプト

| パス | 概要 |
| --- | --- |
| `scripts/aggregate_ground_truth.py` | ゾーン別人数を合計し `ground_truth_totals.csv` を生成 |
| `scripts/align_ground_truth.py` | メタデータと正解 CSV を突合し `aligned_frames.csv` を生成 |
| `scripts/run_evaluation.py` | 推奨設定で Precision / Recall / F1 / AP を算出 |
| `scripts/evaluate_models.py` | `yolo11n/s/m` の推論比較 | 
| `scripts/evaluate_params.py` | conf/IoU/imgsz でパラメータ探索 |

### 2.2 モジュール構成（抜粋）

- `src/ocr/timestamp_ocr.py` – タイムスタンプ OCR
- `src/extraction/time_based_extractor.py` – フレーム抽出
- `src/detection/person_detector.py` – YOLOv11 人物検出
- `src/evaluation/dataset.py` – メタデータとラベルの整合
- `src/evaluation/evaluator.py` – 指標算出パイプライン
- `src/evaluation/metrics.py` – Precision / Recall / F1 / AP 等のユーティリティ

### 2.3 最新評価結果（2025-09-26）

- モデル: `yolo11m.pt`
- 推奨設定: `conf=0.3`, `iou=0.45`, `imgsz=832`
- 評価フレーム数: 33
- Precision 0.952 / Recall 0.590 / F1 0.728 / AP 0.750
- 解釈: False Positive は低いが、検出漏れ（Recall）が 0.59 で改善余地あり。詳細は `docs/evaluation_summary.md` を参照。

## 3. 進捗サマリー

- ✅ Phase1〜3 の主要タスク（環境整備・OCR・抽出・人物検出実装）完了
- ✅ モデル比較 (`scripts/evaluate_models.py`)・パラメータ調整 (`scripts/evaluate_params.py`) 実施済み
- ✅ Phase 4.1（指標実装・評価実行・レポート更新）完了
- ⏳ Phase 4.2（推論速度・メモリベンチマーク）未着手
- ⏳ Phase 5（統合パイプライン・最終検証）未着手

## 4. 残タスク

| 項目 | 概要 |
| --- | --- |
| 推論速度評価 | `PerformanceBenchmark` クラス実装、CPU/MPS の FPS・推論時間・メモリ測定 |
| 精度改善策検討 | Recall 改善のための閾値・モデル調整、ラベル再検証 |
| ラベル拡充 | `label_data.csv` に無い時間帯へのラベル付与（別 CSV 等で管理）|
| 統合パイプライン | `TimeBasedPersonDetectionPipeline` の実装と end-to-end テスト |
| ドキュメント更新 | 性能レポート、最終総括レポートの作成 |

## 5. テストと検証手順

```bash
# ユニットテスト
PYTHONPATH=$PWD pytest

# 推論評価（推奨設定）
PYTHONPATH=$PWD python scripts/run_evaluation.py \
  --metadata output/frames/samples/metadata.json \
  --ground-truth output/results/label_data.csv \
  --model yolo11m.pt --conf 0.3 --iou 0.45 --imgsz 832

# GT 整合ファイル生成
PYTHONPATH=$PWD python scripts/align_ground_truth.py
```

## 6. Git / PR ワークフロー

- 開発ブランチ: `feature/poc-phase3`
- 直近コミット: 評価データ整合および計画書反映まで完了
- `yolo11*.pt` は大容量のためコミット禁止
- `label_data.csv` は正解データなので編集禁止（必要なラベル追加は別ファイルで管理）
- PR 作成前には `pytest` と `scripts/run_evaluation.py` を実行して結果を README / docs に反映

---

本 README は 2025-09-26 時点の進捗をまとめています。以降の変更は計画書およびドキュメント群に反映してください。
