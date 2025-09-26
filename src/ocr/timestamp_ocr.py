from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

import cv2
import numpy as np
import pytesseract


TIMESTAMP_PATTERN = re.compile(r"(\d{4}/\d{2}/\d{2})\s*(\d{2}:\d{2}:\d{2})")


@dataclass
class OcrConfig:
    roi: Tuple[int, int, int, int]
    threshold: int = 180
    blur_kernel: Tuple[int, int] = (3, 3)
    invert: bool = False
    lang: str = "eng"


class TimestampOCR:
    def __init__(self, config: OcrConfig):
        self.config = config

    def extract_rois(self, frame: np.ndarray) -> np.ndarray:
        x, y, w, h = self.config.roi
        h_frame, w_frame = frame.shape[:2]
        x = max(0, min(x, w_frame - 1))
        y = max(0, min(y, h_frame - 1))
        w = min(w, w_frame - x)
        h = min(h, h_frame - y)
        return frame[y : y + h, x : x + w]

    def preprocess(self, roi: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.config.blur_kernel:
            gray = cv2.GaussianBlur(gray, self.config.blur_kernel, 0)

        # Apply contrast enhancement for better OCR results
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        if self.config.threshold < 0:
            thresh_type = cv2.THRESH_BINARY
            if self.config.invert:
                thresh_type = cv2.THRESH_BINARY_INV
            _, thresh = cv2.threshold(
                gray,
                0,
                255,
                thresh_type | cv2.THRESH_OTSU,
            )
        else:
            thresh_type = cv2.THRESH_BINARY
            if self.config.invert:
                thresh_type = cv2.THRESH_BINARY_INV
            _, thresh = cv2.threshold(
                gray,
                self.config.threshold,
                255,
                thresh_type,
            )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        if self.config.invert:
            binary = cv2.bitwise_not(binary)
        return binary

    def recognize(self, processed: np.ndarray) -> Optional[str]:
        tess_config = "--psm 7 -c tessedit_char_whitelist=0123456789/:"
        text = pytesseract.image_to_string(processed, lang=self.config.lang, config=tess_config)
        match = TIMESTAMP_PATTERN.search(text)
        if match:
            date_part, time_part = match.groups()
            return f"{date_part} {time_part}"

        inverted = cv2.bitwise_not(processed)
        text_inv = pytesseract.image_to_string(inverted, lang=self.config.lang, config=tess_config)
        match = TIMESTAMP_PATTERN.search(text_inv)
        if match:
            date_part, time_part = match.groups()
            return f"{date_part} {time_part}"

        return None

    def parse_datetime(self, timestamp: str) -> Optional[datetime]:
        try:
            return datetime.strptime(timestamp, "%Y/%m/%d %H:%M:%S")
        except ValueError:
            return None

    def detect(self, frame: np.ndarray) -> Optional[datetime]:
        roi = self.extract_rois(frame)
        processed = self.preprocess(roi)
        timestamp = self.recognize(processed)
        if not timestamp:
            return None
        return self.parse_datetime(timestamp)
