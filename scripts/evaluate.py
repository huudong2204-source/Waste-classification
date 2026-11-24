"""Đánh giá checkpoint trên đúng tập test đã giữ lại."""
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from material import BEST, DATA, device_for, file_hash, load_checkpoint, transform, write_json
from train import Pictures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=BEST)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    device = device_for(args.device)
    model, info = load_checkpoint(args.checkpoint, device)
    rows = info["splits"]["test"]
    for row in rows:
        if file_hash(args.data_root / row["path"]) != row["sha256"]:
            raise ValueError(f"Ảnh test đã thay đổi: {row['path']}")
    labels = info["labels"]
    n = len(labels)
    matrix = torch.zeros((n, n), dtype=torch.long)
    with torch.inference_mode():
        for images, targets in DataLoader(Pictures(args.data_root, rows, transform(info["preprocess"])), batch_size=24):
            pred = model(images.to(device)).argmax(1).cpu()
            matrix += torch.bincount(targets * n + pred, minlength=n*n).reshape(n, n)
    f1 = 2 * matrix.diag() / (matrix.sum(0) + matrix.sum(1)).clamp_min(1)
    report = dict(checkpoint_sha256=file_hash(args.checkpoint), epoch=info["epoch"], seed=info["seed"],
                  labels=labels, n=len(rows), correct=int(matrix.diag().sum()),
                  accuracy=int(matrix.diag().sum()) / len(rows), f1=dict(zip(labels, f1.tolist())),
                  macro_f1=float(f1.mean()), confusion_true_rows_predicted_columns=matrix.tolist())
    write_json(args.checkpoint.parent / "test_metrics.json", report)
    print(f"Test: {report['correct']}/{len(rows)}, Accuracy {report['accuracy']:.2%}")
    print("F1:", report["f1"])
    print("Hàng = thực, cột = dự đoán:", labels, "\n", matrix)


if __name__ == "__main__":
    main()
