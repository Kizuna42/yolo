# パラメータチューニング概要 (2025-09-26)

## 概要
- 使用モデル: YOLOv11n (事前学習モデル)
- 評価フレーム数: 20 (metadata.json より抽出)
- 指標: 平均推論時間 (ms), FPS, 平均検出数/フレーム

## 評価レンジ
- conf: [0.3, 0.5, 0.7]
- IoU: [0.3, 0.45, 0.6]
- 画像サイズ: [416, 640, 832]

## 結果概要

### conf = 0.3
| IoU | 416px | 640px | 832px |
| --- | ----- | ----- | ----- |
| 0.3 | Avg 131ms / 3.20 det | Avg 144ms / 6.35 det | Avg 171ms / 9.50 det |
| 0.45 | Avg 33.9ms / 3.20 det | Avg 49.0ms / 6.40 det | Avg 54.7ms / 9.60 det |
| 0.6 | Avg 33.1ms / 3.50 det | Avg 46.7ms / 6.70 det | Avg 60.7ms / 10.10 det |

### conf = 0.5
| IoU | 416px | 640px | 832px |
| --- | ----- | ----- | ----- |
| 0.3 | Avg 29.3ms / 1.25 det | Avg 48.9ms / 3.25 det | Avg 58.1ms / 5.00 det |
| 0.45 | Avg 33.2ms / 1.25 det | Avg 41.6ms / 3.25 det | Avg 47.7ms / 5.00 det |
| 0.6 | Avg 27.6ms / 1.30 det | Avg 32.3ms / 3.25 det | Avg 41.5ms / 5.35 det |

### conf = 0.7
| IoU | 416px | 640px | 832px |
| --- | ----- | ----- | ----- |
| 0.3 | Avg 21.2ms / 0.15 det | Avg 27.5ms / 0.85 det | Avg 32.9ms / 1.95 det |
| 0.45 | Avg 20.6ms / 0.15 det | Avg 24.0ms / 0.85 det | Avg 29.4ms / 1.95 det |
| 0.6 | Avg 19.7ms / 0.15 det | Avg 24.7ms / 0.85 det | Avg 29.5ms / 1.95 det |

## 結論
- conf=0.3, IoU=0.45, 画像サイズ832pxが検出数最大かつ許容推論時間。
- conf=0.5 を超えると検出漏れが顕著に増加。
- conf=0.3, imgsz=640px は高速性重視の妥協案 (Avg 49ms, 6.4 detections)。
- 今後、バッチ処理とメモリ使用量の検証が必要。

