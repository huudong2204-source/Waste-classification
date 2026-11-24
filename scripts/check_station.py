"""Phát chuỗi ảnh cố định qua đúng bộ quét của ứng dụng, không cần webcam."""
import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np

from material import BEST, DATA, LABELS, PROJECT, Predictor, write_json
from scanner import PalmDetector, Scanner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hand-image", type=Path, help="Ảnh bàn tay thật, nền sáng")
    args = parser.parse_args()
    predictor = Predictor(BEST)
    cv2.setNumThreads(min(8, os.cpu_count() or 4))
    scanner = Scanner(predictor, PalmDetector())
    blank = np.full((360, 360, 3), 245, np.uint8)
    results = []

    def run(case, roi, frames, expected):
        states, times = [], []
        for _ in range(frames):
            start = time.perf_counter()
            reading = scanner.step(roi)
            times.append((time.perf_counter() - start) * 1000)
            states.append(reading.label)
        results.append(dict(case=case, expected=expected, final_bin=states[-1], bins=states,
                            message=reading.message, matched=(states[-1] == expected),
                            p50_processing_ms=round(float(np.median(times)), 2)))
        print(f"{case}: {reading.message} | thùng={states[-1]} | mong đợi={expected}")

    def mounted(path):
        photo = cv2.imread(str(path))
        if photo is None:
            raise ValueError(f"Không đọc được {path}")
        h, w = photo.shape[:2]
        scale = 276 / max(h, w)
        resized = cv2.resize(photo, (round(w*scale), round(h*scale)), interpolation=cv2.INTER_AREA)
        a, b = resized.shape[:2]
        y, x = (360-a)//2, (360-b)//2
        roi = blank.copy()
        roi[y:y+a, x:x+b] = resized
        return roi

    scanner.calibrate()
    run("Ghi nền và kiểm tra bàn trống", blank, 14, None)
    rows = predictor.info["splits"]["validation"]
    for name in LABELS:
        example = next(row["path"] for row in rows if row["path"].startswith(name + "/"))
        run("Đặt hoặc đổi vật · " + example, mounted(DATA / example), 7, name)
        run("Lấy vật ra · " + name, blank, 1, None)
    if args.hand_image:
        run("Tay không · " + args.hand_image.name, mounted(args.hand_image), 6, None)
    write_json(PROJECT / "outputs" / "model" / "station_check.json", results)


if __name__ == "__main__":
    main()
