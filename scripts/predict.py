"""Thử một ảnh mới, sử dụng đúng thứ tự nhãn và tiền xử lý trong checkpoint."""
import argparse
from pathlib import Path

from material import BEST, LABELS, Predictor, open_rgb


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=BEST)
    args = parser.parse_args()
    try:
        label, score, accepted = Predictor(args.checkpoint).predict(open_rgb(args.image))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Không thể dự đoán: {exc}\n")
    print(f"Nhãn có điểm cao nhất: {LABELS[label]} ({label}) · {score:.1%}")
    print("Hướng dẫn:", f"Thùng {LABELS[label].lower()}" if accepted else "Chưa rõ vật liệu. Hãy thử góc khác.")
