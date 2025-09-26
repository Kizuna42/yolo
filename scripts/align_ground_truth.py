from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.dataset import (
    align_frames_to_ground_truth_df,
    load_ground_truth_csv,
    parse_metadata,
    total_counts_per_timestamp,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align metadata frames with ground truth counts")
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("output/frames/samples/metadata.json"),
        help="Path to metadata JSON",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        default=Path("output/results/label_data.csv"),
        help="Ground truth label CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/results/aligned_frames.csv"),
        help="Output CSV for aligned results",
    )
    parser.add_argument("--tolerance", type=int, default=5, help="Tolerance in minutes for alignment")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frames = parse_metadata(args.metadata)
    gt_df = load_ground_truth_csv(args.ground_truth)
    totals = total_counts_per_timestamp(gt_df)
    aligned_df = align_frames_to_ground_truth_df(frames, totals, tolerance_minutes=args.tolerance)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    aligned_df.to_csv(args.output, index=False)
    print(f"Aligned {len(aligned_df)} frames. Saved to {args.output}")


if __name__ == "__main__":
    main()

