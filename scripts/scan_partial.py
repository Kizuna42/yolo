from __future__ import annotations

import argparse
from pathlib import Path

from src.extraction.time_based_extractor import TimeBasedFrameExtractor
from src.ocr.timestamp_ocr import OcrConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Partial timestamp scan for debugging")
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("--output", type=Path, default=Path("output/frames"), help="Output directory")
    parser.add_argument("--start", type=int, default=0, help="Start frame index")
    parser.add_argument("--frames", type=int, default=500, help="Number of frames to scan")
    parser.add_argument("--roi", type=int, nargs=4, default=[900, 30, 350, 50], help="ROI (x y w h)")
    args = parser.parse_args()

    ocr_config = OcrConfig(tuple(args.roi), threshold=-1, invert=False)
    extractor = TimeBasedFrameExtractor(
        video_path=args.video,
        output_dir=args.output,
        ocr_config=ocr_config,
    )
    frames = extractor.scan_timestamps_partial(max_frames=args.frames, start_frame=args.start)
    print(f"detected {len(frames)} timestamps")
    if frames:
        print("first", frames[0])
        print("last", frames[-1])


if __name__ == "__main__":
    main()
