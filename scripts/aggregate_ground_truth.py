from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate ground truth counts across zones")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/results/label_data.csv"),
        help="Ground truth CSV with zone counts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/results/ground_truth_totals.csv"),
        help="Output CSV with total counts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    df = df.rename(columns={df.columns[0]: "timestamp"})
    zone_cols = [col for col in df.columns if col != "timestamp"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], infer_datetime_format=True)
    df["total_count"] = df[zone_cols].fillna(0).astype(int).sum(axis=1)
    df[["timestamp", "total_count"]].to_csv(args.output, index=False)
    print(f"Saved aggregated totals to {args.output}")


if __name__ == "__main__":
    main()

