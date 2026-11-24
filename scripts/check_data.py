"""Kiểm tra 6 lớp TrashNet và ghép ảnh mẫu từ các tệp đã đọc được."""

import argparse
import hashlib
import json
import random
import struct
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "outputs" / "data_check"
LABELS = {
    "plastic": "Nhựa",
    "metal": "Kim loại",
    "cardboard": "Bìa carton",
    "glass": "Thủy tinh",
    "paper": "Giấy",
    "trash": "Rác khác",
}
SAMPLES = 5


def scan(root):
    classes = {}
    readable = {}
    errors = []
    hashes = {"file_sha256": defaultdict(list), "rgb_sha256": defaultdict(list)}
    for label, name in LABELS.items():
        folder = root / label
        readable[label] = []
        candidates = sorted(p for p in folder.rglob("*") if p.is_file()
                            and p.name not in {".DS_Store", "Thumbs.db", "desktop.ini"}
                            and not p.name.startswith("._"))
        for path in candidates:
            relative = path.relative_to(root).as_posix()
            try:
                with Image.open(path) as img:
                    img.load()  # Buộc giải mã toàn bộ ảnh trước khi đếm.
                    rgb = ImageOps.exif_transpose(img).convert("RGB")
                digest = hashlib.sha256(struct.pack(">II", *rgb.size))
                digest.update(rgb.tobytes())
                file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                errors.append({"file": relative, "error": str(exc)})
                continue
            hashes["file_sha256"][file_digest].append(relative)
            hashes["rgb_sha256"][digest.hexdigest()].append(relative)
            readable[label].append(path)
        count = len(readable[label])
        classes[label] = {"name": name, "readable": count, "errors": len(candidates) - count}

    groups = {method: {h: files for h, files in index.items() if len(files) > 1}
              for method, index in hashes.items()}
    return readable, {"dataset_root": str(root), "classes": classes,
                      "errors": errors, "duplicate_groups": groups}


def make_preview(readable, output):
    canvas = Image.new("RGB", (1024, 58 + len(LABELS) * 216), "#f6f7f5")
    draw = ImageDraw.Draw(canvas)
    try:
        title = ImageFont.truetype("segoeuib.ttf", 28)
        heading = ImageFont.truetype("segoeuib.ttf", 21)
        caption = ImageFont.truetype("segoeui.ttf", 16)
    except OSError:
        title, heading, caption = (ImageFont.load_default(size=s) for s in (28, 21, 16))
    draw.text((20, 10), "TrashNet  /  Kiểm tra ảnh theo vật liệu", font=title, fill="#173e32")
    rng = random.Random(42)
    for row, (label, name) in enumerate(LABELS.items()):
        top = 58 + row * 216
        paths = readable[label]
        draw.text((20, top), f"{name}  ·  {label}  ·  {len(paths)} ảnh đọc được",
                  font=heading, fill="#173e32")
        for col, path in enumerate(rng.sample(paths, min(SAMPLES, len(paths)))):
            x, y = 20 + col * 199, top + 36
            with Image.open(path) as img:
                thumb = ImageOps.exif_transpose(img).convert("RGB")
                thumb.thumbnail((186, 140), Image.Resampling.LANCZOS)
                canvas.paste(thumb, (x + (186 - thumb.width) // 2, y + (140 - thumb.height) // 2))
            draw.text((x, y + 142), path.name, font=caption, fill="#31463d")
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=PROJECT / "dataset-resized",
                        help="Thư mục chứa trực tiếp các thư mục lớp TrashNet")
    root = parser.parse_args().data_root.resolve()
    missing = [str(root / label) for label in LABELS if not (root / label).is_dir()]
    if missing:
        parser.error("Thiếu thư mục: " + ", ".join(missing))

    readable, report = scan(root)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    make_preview(readable, OUTPUT / "preview.jpg")
    (OUTPUT / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    for label, counts in report["classes"].items():
        print(f"{label}: {counts['readable']} ảnh đọc được, {counts['errors']} tệp lỗi")
    for method, groups in report["duplicate_groups"].items():
        print(f"{method}: {len(groups)} nhóm trùng")
    print(f"Kết quả: {OUTPUT}")


if __name__ == "__main__":
    if sys.stdout.encoding and not sys.stdout.isatty():
        sys.stdout.reconfigure(encoding="utf-8")
    main()
