"""Trạm phân loại rác – giao diện Tkinter (tự vẽ bằng Canvas + Pillow) và webcam OpenCV."""
import argparse
import csv
import math
import os
import queue
import re
import threading
import time
import tkinter as tk
from collections import deque
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from tkinter import font as tkfont

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps, ImageTk

from material import BEST, LABELS, PROJECT, Predictor
from scanner import PalmDetector, Scanner


# ───────────────────────────── Bảng màu "Eco-Tech Deep Slate" ─────────────────────────────
BG, CARD, CARD2, BORDER, TRACK = "#0B131E", "#131E2E", "#0E1726", "#233349", "#1C2A40"
INK, MUTED, DIM = "#F1F5F9", "#94A3B8", "#64748B"
EMERALD, ROSE, AMBER, CYAN = "#10B981", "#F43F5E", "#F59E0B", "#06B6D4"
COLORS = {
    "plastic": "#06B6D4",
    "metal": "#818CF8",
    "cardboard": "#F59E0B",
    "glass": "#34D399",
    "paper": "#E2E8F0",
    "trash": "#FB7185",
}
FONT = "Segoe UI"

# state của Reading -> (nhãn ngắn trên banner, màu, biểu tượng)
STATES = {
    "off": ("NGOẠI TUYẾN", DIM, "camera_off"),
    "loading": ("ĐANG KHỞI ĐỘNG", AMBER, "scan"),
    "error": ("CẦN CHÚ Ý", ROSE, "alert"),
    "uncalibrated": ("CHƯA CÓ NỀN", AMBER, "target"),
    "calibrating": ("ĐANG GHI NỀN TRỐNG", AMBER, "target"),
    "idle": ("BÀN TRỐNG · SẴN SÀNG", CYAN, "scan"),
    "moving": ("VẬT ĐANG DI CHUYỂN", AMBER, "pause"),
    "blocked": ("NỀN BỊ THAY ĐỔI", ROSE, "alert"),
    "hand": ("PHÁT HIỆN BÀN TAY", ROSE, "hand"),
    "scanning": ("ĐANG XÁC NHẬN VẬT LIỆU", AMBER, "scan"),
    "unknown": ("CHƯA RÕ VẬT LIỆU", AMBER, "question"),
    "locked": ("ĐÃ CHỐT NHÃN", EMERALD, "check"),
}
STEP_OF = {"uncalibrated": 0, "calibrating": 0, "idle": 1, "moving": 1, "hand": 1, "blocked": 1,
           "scanning": 1, "unknown": 1, "locked": 2}
STEPS = ("Ghi nền", "Đặt vật", "Bỏ thùng")


# ───────────────────────────── Tiện ích vẽ bằng Pillow (siêu lấy mẫu → mịn viền) ─────────────────────────────
SS = 4


def rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def mix(a, b, t):
    return "#%02X%02X%02X" % tuple(round(x + (y - x) * t) for x, y in zip(rgb(a), rgb(b)))


@lru_cache(maxsize=512)
def panel_img(w, h, r, fill, outline, ow=1, glow=None, gpx=0):
    """Thẻ bo góc (kèm viền, quầng sáng tùy chọn). Kích thước ảnh = (w+2*gpx, h+2*gpx)."""
    W, H = (w + 2 * gpx) * SS, (h + 2 * gpx) * SS
    base = rgb(fill)
    img = Image.new("RGBA", (W, H), base + (0,))
    box = (gpx * SS, gpx * SS, (gpx + w) * SS - 1, (gpx + h) * SS - 1)
    if glow:
        layer = Image.new("RGBA", (W, H), rgb(glow) + (0,))
        ImageDraw.Draw(layer).rounded_rectangle(box, r * SS, fill=rgb(glow) + (190,))
        layer = layer.filter(ImageFilter.GaussianBlur(max(1, gpx * SS * .5)))
        img = Image.alpha_composite(img, layer)
    ImageDraw.Draw(img).rounded_rectangle(box, r * SS, fill=base + (255,),
                                          outline=(rgb(outline) + (255,)) if outline else None, width=ow * SS)
    return img.resize((w + 2 * gpx, h + 2 * gpx), Image.LANCZOS)


@lru_cache(maxsize=512)
def bar_img(w, h, frac, fill, track=TRACK):
    img = Image.new("RGBA", (w * SS, h * SS), rgb(track) + (0,))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, w * SS - 1, h * SS - 1), h * SS // 2, fill=rgb(track) + (255,))
    if frac > .004:
        fw = min(w * SS, max(h * SS, round(frac * w * SS)))
        d.rounded_rectangle((0, 0, fw - 1, h * SS - 1), h * SS // 2, fill=rgb(fill) + (255,))
    return img.resize((w, h), Image.LANCZOS)


@lru_cache(maxsize=128)
def dot_img(size, fill, ring=None, halo=False):
    pad = 4 if halo else 0
    N = (size + 2 * pad) * SS
    img = Image.new("RGBA", (N, N), rgb(fill or ring) + (0,))
    if halo:
        layer = Image.new("RGBA", (N, N), rgb(fill) + (0,))
        ImageDraw.Draw(layer).ellipse((pad * SS, pad * SS, N - pad * SS - 1, N - pad * SS - 1), fill=rgb(fill) + (200,))
        img = Image.alpha_composite(img, layer.filter(ImageFilter.GaussianBlur(pad * SS * .6)))
    d = ImageDraw.Draw(img)
    box = (pad * SS, pad * SS, N - pad * SS - 1, N - pad * SS - 1)
    d.ellipse(box, fill=(rgb(fill) + (255,)) if fill else None,
              outline=(rgb(ring) + (255,)) if ring else None, width=int(1.6 * SS))
    return img.resize((size + 2 * pad, size + 2 * pad), Image.LANCZOS)


@lru_cache(maxsize=8)
def _pil_font(px):
    for name in ("segoeuib.ttf", "arialbd.ttf"):
        try:
            return ImageFont.truetype(name, px)
        except OSError:
            continue
    return ImageFont.load_default()


@lru_cache(maxsize=256)
def glyph(name, size, color):
    """Biểu tượng vector đơn sắc vẽ theo tọa độ chuẩn hóa 0..1."""
    N = size * SS
    col = rgb(color) + (255,)
    clear = rgb(color) + (0,)
    img = Image.new("RGBA", (N, N), clear)
    d = ImageDraw.Draw(img)

    def P(x, y):
        return x * N, y * N

    def dot(x, y, r, fill=col):
        d.ellipse((x * N - r * N, y * N - r * N, x * N + r * N, y * N + r * N), fill=fill)

    def line(pts, w, fill=col):
        d.line([P(*p) for p in pts], fill=fill, width=int(w * N), joint="curve")
        for p in (pts[0], pts[-1]):
            dot(*p, w / 2, fill)

    def rr(x1, y1, x2, y2, r, fill=col):
        d.rounded_rectangle((x1 * N, y1 * N, x2 * N, y2 * N), r * N, fill=fill)

    if name == "power":
        d.arc((.17 * N, .20 * N, .83 * N, .86 * N), 300, 240, fill=col, width=int(.10 * N))
        line([(.5, .08), (.5, .48)], .10)
    elif name == "target":
        d.ellipse((.2 * N, .2 * N, .8 * N, .8 * N), outline=col, width=int(.08 * N))
        dot(.5, .5, .09)
        for a, b in (((.5, .04), (.5, .2)), ((.5, .8), (.5, .96)), ((.04, .5), (.2, .5)), ((.8, .5), (.96, .5))):
            line([a, b], .07)
    elif name == "reset":
        c, r = .5, .32
        d.arc((c * N - r * N, c * N - r * N, c * N + r * N, c * N + r * N), 25, 305, fill=col, width=int(.10 * N))
        th = math.radians(305)
        ex, ey = c + r * math.cos(th), c + r * math.sin(th)
        tx, ty = -math.sin(th), math.cos(th)
        nx, ny = math.cos(th), math.sin(th)
        d.polygon([P(ex + tx * .17, ey + ty * .17), P(ex + nx * .15 - tx * .02, ey + ny * .15 - ty * .02),
                   P(ex - nx * .15 - tx * .02, ey - ny * .15 - ty * .02)], fill=col)
    elif name == "check":
        line([(.20, .53), (.42, .74), (.80, .29)], .13)
    elif name == "alert":
        line([(.5, .18), (.5, .60)], .13)
        dot(.5, .81, .075)
    elif name == "question":
        d.text(P(.5, .54), "?", font=_pil_font(int(.95 * N)), fill=col, anchor="mm")
    elif name == "pause":
        rr(.24, .16, .43, .84, .06)
        rr(.57, .16, .76, .84, .06)
    elif name == "scan":
        for sx in (0, 1):
            for sy in (0, 1):
                x0, y0 = (.12, .12) if not sx else (.88, .12)
                y0 = .12 if not sy else .88
                x0 = .12 if not sx else .88
                dx, dy = (.24 if not sx else -.24), (.24 if not sy else -.24)
                line([(x0 + dx, y0), (x0, y0), (x0, y0 + dy)], .09)
        dot(.5, .5, .09)
    elif name == "hand":
        rr(.27, .46, .73, .90, .16)
        for x, top in ((.27, .20), (.41, .10), (.55, .12), (.69 - .04, .22)):
            rr(x, top, x + .12, .62, .06)
        rr(.10, .44, .30, .60, .08)
    elif name == "gallery":
        d.rounded_rectangle((.10 * N, .18 * N, .90 * N, .82 * N), .12 * N, outline=col, width=int(.08 * N))
        dot(.33, .40, .07)
        d.polygon([P(.18, .74), P(.42, .50), P(.56, .64), P(.68, .54), P(.82, .74)], fill=col)
    elif name == "folder":
        rr(.10, .20, .46, .42, .08)
        rr(.10, .30, .90, .82, .11)
        rr(.19, .39, .81, .73, .05, fill=clear)
    elif name == "trash":
        rr(.36, .12, .64, .22, .04)
        rr(.16, .22, .84, .32, .05)
        rr(.23, .34, .77, .88, .08)
        rr(.37, .44, .44, .76, .03, fill=clear)
        rr(.56, .44, .63, .76, .03, fill=clear)
    elif name in ("camera", "camera_off"):
        rr(.30, .17, .56, .34, .05)
        rr(.07, .27, .93, .83, .11)
        d.ellipse((.5 * N - .19 * N, .55 * N - .19 * N, .5 * N + .19 * N, .55 * N + .19 * N), fill=clear)
        d.ellipse((.5 * N - .095 * N, .55 * N - .095 * N, .5 * N + .095 * N, .55 * N + .095 * N), fill=col)
        if name == "camera_off":
            line([(.14, .90), (.88, .10)], .17, fill=clear)
            line([(.14, .90), (.88, .10)], .075)
    else:
        dot(.5, .5, .3)
    return img.resize((size, size), Image.LANCZOS)


@lru_cache(maxsize=64)
def badge_icon(name, color, size):
    """Biểu tượng trạng thái đặt trong huy hiệu tròn nền pha màu."""
    N = size * SS
    img = Image.new("RGBA", (N, N), rgb(color) + (0,))
    d = ImageDraw.Draw(img)
    d.ellipse((0, 0, N - 1, N - 1), fill=rgb(mix(CARD, color, .16)) + (255,),
              outline=rgb(mix(CARD, color, .55)) + (255,), width=int(1.5 * SS))
    img = img.resize((size, size), Image.LANCZOS)
    inner = round(size * .52)
    g = glyph(name, inner, color)
    img.alpha_composite(g, ((size - inner) // 2, (size - inner) // 2))
    return img


@lru_cache(maxsize=64)
def bin_icon(label, color, size):
    """Thùng rác vẽ nét rõ ràng, có biểu tượng vật liệu khắc rỗng trên thân."""
    N = size * SS
    col = rgb(color) + (255,)
    clear = rgb(color) + (0,)
    img = Image.new("RGBA", (N, N), clear)
    d = ImageDraw.Draw(img)
    pts = [(.18 * N, .31 * N), (.82 * N, .31 * N), (.74 * N, .94 * N), (.26 * N, .94 * N)]
    d.polygon(pts, fill=col)
    d.line(pts + [pts[0]], fill=col, width=int(.06 * N), joint="curve")
    d.rounded_rectangle((.10 * N, .14 * N, .90 * N, .27 * N), .05 * N, fill=col)
    d.rounded_rectangle((.37 * N, .04 * N, .63 * N, .16 * N), .05 * N, fill=col)
    d.rounded_rectangle((.44 * N, .08 * N, .56 * N, .17 * N), .02 * N, fill=clear)

    def cut(x1, y1, x2, y2, r):
        d.rounded_rectangle((x1 * N, y1 * N, x2 * N, y2 * N), r * N, fill=clear)

    if label == "plastic":  # chai nhựa
        cut(.465, .39, .535, .45, .015)
        cut(.45, .45, .55, .55, .04)
        cut(.385, .53, .615, .87, .06)
    elif label == "metal":  # lon thiếc
        cut(.40, .42, .60, .87, .04)
        d.rectangle((.40 * N, .50 * N, .60 * N, .53 * N), fill=col)
        d.rectangle((.40 * N, .76 * N, .60 * N, .79 * N), fill=col)
    elif label == "cardboard":  # thùng carton
        cut(.35, .45, .65, .85, .03)
        d.rectangle((.485 * N, .45 * N, .515 * N, .62 * N), fill=col)
    elif label == "glass":  # chai/lọ thủy tinh
        cut(.45, .38, .55, .54, .02)
        d.ellipse((.35 * N, .51 * N, .65 * N, .87 * N), fill=clear)
        d.arc((.40 * N, .56 * N, .60 * N, .82 * N), 200, 260, fill=col, width=max(2, int(.03 * N)))
    elif label == "paper":  # tờ giấy gập góc
        d.polygon([(.37 * N, .42 * N), (.55 * N, .42 * N), (.64 * N, .51 * N),
                   (.64 * N, .86 * N), (.37 * N, .86 * N)], fill=clear)
        d.polygon([(.55 * N, .42 * N), (.64 * N, .51 * N), (.55 * N, .51 * N)], fill=col)
        d.rectangle((.42 * N, .59 * N, .59 * N, .625 * N), fill=col)
        d.rectangle((.42 * N, .69 * N, .59 * N, .725 * N), fill=col)
        d.rectangle((.42 * N, .78 * N, .53 * N, .815 * N), fill=col)
    elif label == "trash":  # dấu X rác hỗn hợp
        w_x = max(2, int(.07 * N))
        d.line([(.37 * N, .48 * N), (.63 * N, .78 * N)], fill=clear, width=w_x)
        d.line([(.63 * N, .48 * N), (.37 * N, .78 * N)], fill=clear, width=w_x)
    else:
        d.ellipse((.38 * N, .48 * N, .62 * N, .72 * N), fill=clear)
    return img.resize((size, size), Image.LANCZOS)


@lru_cache(maxsize=8)
def logo_img(size):
    N = size * SS
    t = np.add.outer(np.arange(N), np.arange(N)) / (2 * N - 2)
    c1, c2 = np.array(rgb("#06B6D4")), np.array(rgb("#10B981"))
    arr = np.zeros((N, N, 4), np.uint8)
    arr[..., :3] = (c1 + (c2 - c1) * t[..., None]).astype(np.uint8)
    arr[..., 3] = 255
    img = Image.fromarray(arr, "RGBA")
    mask = Image.new("L", (N, N), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, N - 1, N - 1), .28 * N, fill=255)
    img.putalpha(mask)
    d = ImageDraw.Draw(img)
    ax, ay, bx, by = .27, .74, .75, .25
    L = math.hypot(bx - ax, by - ay)
    ux, uy = (bx - ax) / L, (by - ay) / L
    nx, ny = -uy, ux
    top, bottom = [], []
    for i in range(41):
        s = i / 40
        wd = .20 * math.sin(math.pi * s) ** .85
        cx, cy = ax + ux * L * s, ay + uy * L * s
        top.append((cx + nx * wd * 1.05, cy + ny * wd * 1.05))
        bottom.append((cx - nx * wd * .9, cy - ny * wd * .9))
    d.polygon([(x * N, y * N) for x, y in top + bottom[::-1]], fill=(255, 255, 255, 255))
    vein = rgb(mix("#06B6D4", "#10B981", .6)) + (255,)
    d.line([(.33 * N, .68 * N), (.64 * N, .37 * N)], fill=vein, width=int(.035 * N))
    d.line([(.20 * N, .82 * N), (.31 * N, .70 * N)], fill=(255, 255, 255, 255), width=int(.06 * N))
    return img.resize((size, size), Image.LANCZOS)


@lru_cache(maxsize=32)
def round_mask(w, h, r):
    m = Image.new("L", (w * 3, h * 3), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * 3 - 1, h * 3 - 1), r * 3, fill=255)
    return m.resize((w, h), Image.LANCZOS)


# ───────────────────────────── Kho ảnh minh chứng ─────────────────────────────
IMG_EXT = {".jpg", ".jpeg", ".png"}
KHO_HEADER = ["timestamp", "label", "label_vi", "confidence", "image_path"]
NAME_RE = re.compile(r"^(\d{8})_(\d{6})_(\d{3})_([a-z0-9_]+)_(\d+|na)pct(?:_\d+)?$")


def kho_dir():
    """Luôn đọc PROJECT lúc gọi để unit test có thể patch app.PROJECT sang thư mục tạm."""
    return PROJECT / "outputs" / "kho_rac"


class Evidence:
    """Một ảnh minh chứng trong kho: rel = 'plastic/20261124_..._plastic_94pct.jpg'."""
    __slots__ = ("rel", "label", "conf", "ts")

    def __init__(self, rel, conf=None, ts=None):
        self.rel = rel.replace("\\", "/")
        self.label = self.rel.split("/")[0]
        self.conf, self.ts = conf, ts

    @property
    def path(self):
        return kho_dir() / self.rel


def parse_name(rel):
    """Tách (thời điểm, độ tự tin) từ tên file YYYYMMDD_HHMMSS_fff_<nhãn>_<pct>pct.jpg."""
    m = NAME_RE.match(Path(rel).stem)
    ts = conf = None
    if m:
        try:
            ts = datetime.strptime(m[1] + m[2] + m[3], "%Y%m%d%H%M%S%f")
        except ValueError:
            pass
        if m[5] != "na":
            conf = int(m[5]) / 100
    return ts, conf


def list_images():
    root = kho_dir()
    for label in LABELS:
        try:
            for p in (root / label).iterdir():
                if p.suffix.lower() in IMG_EXT and p.is_file():
                    yield f"{label}/{p.name}"
        except OSError:
            continue


def count_archive():
    return sum(1 for _ in list_images())


def load_archive():
    """Đọc history.csv + tự nhận ảnh nằm sẵn trong thư mục con; mới nhất lên đầu."""
    meta = {}
    try:
        with (kho_dir() / "history.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rel = (row.get("image_path") or "").replace("\\", "/")
                if not rel:
                    continue
                try:
                    ts = datetime.fromisoformat(row["timestamp"])
                except (KeyError, TypeError, ValueError):
                    ts = None
                try:
                    conf = float(row["confidence"])
                except (KeyError, TypeError, ValueError):
                    conf = None
                meta[rel] = (ts, conf)
    except (OSError, csv.Error):
        pass
    items = []
    for rel in list_images():
        ts, conf = meta.get(rel, (None, None))
        name_ts, name_conf = parse_name(rel)
        ts = ts or name_ts
        conf = conf if conf is not None else name_conf
        if ts is None:
            try:
                ts = datetime.fromtimestamp((kho_dir() / rel).stat().st_mtime)
            except OSError:
                ts = datetime.min
        items.append(Evidence(rel, conf, ts))
    items.sort(key=lambda e: e.ts, reverse=True)
    return items


def evidence_from_rel(rel):
    if not rel or not (kho_dir() / rel).is_file():
        return None
    ts, conf = parse_name(rel)
    return Evidence(rel, conf, ts or datetime.now())


def save_evidence(frame_rgb, label, prob, now=None):
    """Lưu ảnh minh chứng + ghi history.csv. Trả về đường dẫn tương đối hoặc None nếu lỗi."""
    now = now or datetime.now()
    pct = "na" if prob is None else str(round(prob * 100))
    stem = f"{now:%Y%m%d_%H%M%S}_{now.microsecond // 1000:03d}_{label}_{pct}pct"
    root = kho_dir()
    rel = f"{label}/{stem}.jpg"
    try:
        (root / label).mkdir(parents=True, exist_ok=True)
        n = 2
        while (root / rel).exists():
            rel = f"{label}/{stem}_{n}.jpg"
            n += 1
        Image.fromarray(np.ascontiguousarray(frame_rgb)).convert("RGB").save(root / rel, "JPEG", quality=92)
    except (OSError, ValueError, TypeError):
        return None
    try:
        with (root / "history.csv").open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(KHO_HEADER)
            writer.writerow([now.isoformat(sep=" ", timespec="milliseconds"), label, LABELS.get(label, label),
                             "" if prob is None else f"{prob:.4f}", rel])
    except OSError:
        pass  # ảnh vẫn được tự nhận lại khi quét thư mục
    return rel


def delete_evidence(ev):
    """Xóa file ảnh và dòng tương ứng trong history.csv."""
    try:
        ev.path.unlink(missing_ok=True)
    except OSError:
        return False
    table = kho_dir() / "history.csv"
    try:
        with table.open(newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        keep = [r for i, r in enumerate(rows) if i == 0 or not (len(r) > 4 and r[4].replace("\\", "/") == ev.rel)]
        if len(keep) != len(rows):
            with table.open("w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerows(keep)
    except (OSError, csv.Error):
        pass
    return True


def open_folder(path):
    try:
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(str(path))
    except (OSError, AttributeError):
        pass


@lru_cache(maxsize=256)
def evidence_img(path, mtime, w, h, r, color):
    """Ảnh minh chứng cắt giữa đúng tỉ lệ khung, bo góc, viền mảnh theo màu nhãn (nếu có)."""
    with Image.open(path) as im:
        out = ImageOps.fit(im.convert("RGB"), (w, h), Image.LANCZOS).convert("RGBA")
    out.putalpha(round_mask(w, h, r))
    if color:
        ring = Image.new("RGBA", (w * SS, h * SS), rgb(color) + (0,))
        ImageDraw.Draw(ring).rounded_rectangle((0, 0, w * SS - 1, h * SS - 1), r * SS,
                                               outline=rgb(color) + (255,), width=int(1.5 * SS))
        out = Image.alpha_composite(out, ring.resize((w, h), Image.LANCZOS))
    return out


def load_thumb(path, w, h, r, color=None):
    try:
        return evidence_img(str(path), os.stat(path).st_mtime_ns, w, h, r, color)
    except (OSError, ValueError):
        return None


# ───────────────────────────── Widget tự vẽ ─────────────────────────────
class Readout:
    """Đối tượng giống Label tối giản: có text, cget/config, và gọi callback khi đổi."""

    def __init__(self, text="", callback=None):
        self._text, self._callback = text, callback

    def cget(self, key):
        if key == "text":
            return self._text
        raise KeyError(key)

    def config(self, **kw):
        if "text" in kw:
            self._text = str(kw["text"])
        if self._callback:
            self._callback()

    configure = config

    def __getitem__(self, key):
        return self.cget(key)


class Surface(tk.Canvas):
    """Canvas nền tối, tự vẽ lại khi đổi kích thước; có cache ảnh theo widget."""

    def __init__(self, master, bg=BG, **kw):
        super().__init__(master, bg=bg, highlightthickness=0, bd=0, **kw)
        self._photos, self._fonts, self._size, self._sig = {}, {}, (0, 0), None
        self.bind("<Configure>", self._on_configure)

    def _on_configure(self, event):
        self._size = (event.width, event.height)
        self._sig = None
        self.redraw()

    def redraw(self):
        pass

    def begin(self, tag="all"):
        self.delete(tag)
        if len(self._photos) > 60:
            self._photos.clear()

    def font(self, size, bold=False, family=FONT):
        key = (size, bold, family)
        if key not in self._fonts:
            self._fonts[key] = tkfont.Font(root=self, family=family, size=size, weight="bold" if bold else "normal")
        return self._fonts[key]

    def measure(self, text, size, bold=False, family=FONT):
        return self.font(size, bold, family).measure(text)

    def put(self, x, y, key, build, anchor="nw", tags=()):
        photo = self._photos.get(key)
        if photo is None:
            photo = self._photos[key] = ImageTk.PhotoImage(build())
        return self.create_image(x, y, image=photo, anchor=anchor, tags=tags)

    def panel(self, x, y, w, h, r, fill, outline, ow=1, glow=None, gpx=0, tags=()):
        key = ("panel", w, h, r, fill, outline, ow, glow, gpx)
        return self.put(x - gpx, y - gpx, key, lambda: panel_img(w, h, r, fill, outline, ow, glow, gpx), tags=tags)

    def bar(self, x, y, w, h, frac, fill, track=TRACK):
        frac = round(max(0., min(1., frac)), 3)
        return self.put(x, y, ("bar", w, h, frac, fill, track), lambda: bar_img(w, h, frac, fill, track))

    def icon(self, x, y, img_key, build, anchor="nw"):
        return self.put(x, y, ("icon",) + img_key, build, anchor=anchor)

    def label(self, x, y, text, size=10, color=INK, bold=False, anchor="w", family=FONT, **kw):
        return self.create_text(x, y, text=text, fill=color, anchor=anchor, font=self.font(size, bold, family), **kw)

    def refresh(self, signature):
        if signature != self._sig:
            self.redraw()
            self._sig = signature


class PillButton(Surface):
    STYLE = {
        "primary": dict(fill=EMERALD, hover="#34D399", fg="#03140D", outline=EMERALD),
        "danger": dict(fill=ROSE, hover="#FB7185", fg="#FFFFFF", outline=ROSE),
        "secondary": dict(fill="#17263B", hover="#1E3350", fg=INK, outline="#2D4466"),
        "ghost": dict(fill=BG, hover="#121D2D", fg=MUTED, outline="#25364D"),
    }
    HEIGHT = 40

    def __init__(self, master, text, command=None, style="secondary", icon=None, state="normal", short=None):
        self._own = dict(text=text, state=state, style=style, icon=icon, short=short, compact=False)
        self.command, self.hover = command, False
        super().__init__(master, width=60, height=self.HEIGHT)
        self.bind("<Enter>", lambda e: self._set_hover(True))
        self.bind("<Leave>", lambda e: self._set_hover(False))
        self.bind("<ButtonRelease-1>", self._click)
        self._fit()

    def shown(self):
        o = self._own
        return o["short"] if (o["compact"] and o["short"]) else o["text"]

    def _fit(self):
        o = self._own
        w = self.measure(self.shown(), 10, True) + 38 + (26 if o["icon"] else 0)
        tk.Canvas.configure(self, width=w, height=self.HEIGHT,
                            cursor="hand2" if o["state"] == "normal" else "arrow")
        self._size = (w, self.HEIGHT)
        self.redraw()

    def configure(self, cnf=None, **kw):
        kw = {**(cnf or {}), **kw}
        for key in list(self._own):
            if key in kw:
                self._own[key] = kw.pop(key)
        if "command" in kw:
            self.command = kw.pop("command")
        if kw:
            tk.Canvas.configure(self, **kw)
        self._fit()

    config = configure

    def cget(self, key):
        return self._own[key] if key in self._own else tk.Canvas.cget(self, key)

    def _set_hover(self, value):
        self.hover = value and self._own["state"] == "normal"
        self.redraw()

    def _click(self, event):
        w, h = self._size
        if 0 <= event.x <= w and 0 <= event.y <= h:
            self.invoke()

    def invoke(self):
        if self._own["state"] == "normal" and self.command:
            return self.command()

    def redraw(self):
        self.begin()
        w, h = self._size
        if w < 8:
            return
        o = self._own
        st = self.STYLE[o["style"]]
        enabled = o["state"] == "normal"
        fill = (st["hover"] if self.hover else st["fill"]) if enabled else CARD2
        fg = st["fg"] if enabled else DIM
        outline = st["outline"] if enabled else BORDER
        self.panel(0, 0, w, h, h // 2, fill, outline)
        text = self.shown()
        tw = self.measure(text, 10, True)
        icon_w = 26 if o["icon"] else 0
        x = (w - tw - icon_w) // 2
        if o["icon"]:
            self.icon(x, (h - 18) // 2, ("glyph", o["icon"], 18, fg), lambda: glyph(o["icon"], 18, fg))
        self.label(x + icon_w, h // 2 - 1, text, 10, fg, True)


class CameraIndex(Surface):
    """Bộ chọn chỉ số camera dạng viên thuốc: Camera [−] 0 [+]."""

    def __init__(self, master, to=16):
        self._own = dict(state="normal", compact=False)
        self.value, self.max = 0, to
        super().__init__(master, width=168, height=40)
        self.bind("<ButtonRelease-1>", self._click)
        self._fit()

    def _fit(self):
        w = (0 if self._own["compact"] else self.measure("Camera", 9, True) + 12) + 112
        tk.Canvas.configure(self, width=w, height=40)
        self._size = (w, 40)
        self.redraw()

    def get(self):
        return str(self.value)

    def configure(self, cnf=None, **kw):
        kw = {**(cnf or {}), **kw}
        for key in list(self._own):
            if key in kw:
                self._own[key] = kw.pop(key)
        if kw:
            tk.Canvas.configure(self, **kw)
        self._fit()

    config = configure

    def cget(self, key):
        return self._own[key] if key in self._own else tk.Canvas.cget(self, key)

    def _click(self, event):
        if self._own["state"] != "normal":
            return
        w = self._size[0]
        if abs(event.x - (w - 82)) <= 14:
            self.value = max(0, self.value - 1)
        elif abs(event.x - (w - 24)) <= 14:
            self.value = min(self.max, self.value + 1)
        else:
            return
        self.redraw()

    def redraw(self):
        self.begin()
        w, h = self._size
        if w < 8:
            return
        on = self._own["state"] == "normal"
        self.panel(0, 0, w, h, h // 2, "#17263B" if on else CARD2, "#2D4466" if on else BORDER)
        if not self._own["compact"]:
            self.label(16, h // 2 - 1, "Camera", 9, MUTED, True)
        for cx, sign in ((w - 82, "−"), (w - 24, "+")):
            self.panel(cx - 13, h // 2 - 13, 26, 26, 13, TRACK, None)
            self.label(cx, h // 2 - 1, sign, 13, INK if on else DIM, True, anchor="center")
        self.label(w - 53, h // 2 - 1, str(self.value), 12, INK if on else DIM, True, anchor="center")


class Header(Surface):
    def __init__(self, master):
        super().__init__(master, height=70)
        self.state, self.timing, self.arch = "off", "", ""

    def redraw(self):
        self.begin()
        w, h = self._size
        if w < 100:
            return
        self.icon(22, 15, ("logo", 40), lambda: logo_img(40))
        self.label(76, h // 2, "Trạm Phân Loại Rác", 16, INK, True)
        # Pill trạng thái + pill tốc độ (căn phải)
        speed = self.timing or "-- ms, -- FPS"
        speed = speed.replace("   ·   ", ", ").replace(" · ", ", ").replace("khung/giây", "FPS")
        right = w - 22
        sw = self.measure(speed, 9, True) + 30
        self.panel(right - sw, 20, sw, 30, 15, CARD, BORDER)
        self.label(right - sw // 2, 34, speed, 9, INK if self.timing else DIM, True, anchor="center")
        live = {"off": ("Ngoại tuyến", DIM), "loading": ("Đang khởi động", AMBER), "error": ("Cần chú ý", ROSE)}
        text, color = live.get(self.state, ("Trực tiếp", EMERALD))
        lw = self.measure(text, 9, True) + 44
        x = right - sw - 10 - lw
        if x > 480:
            self.panel(x, 20, lw, 30, 15, CARD, mix(BORDER, color, .45))
            self.icon(x + 11, 35, ("dot", color), lambda: dot_img(8, color, None, True), anchor="w")
            self.label(x + 31, 34, text, 9, color, True)
        self.create_line(0, h - 1, w, h - 1, fill=BORDER)

    def update_view(self, state, timing, arch):
        self.state, self.timing, self.arch = state, timing, arch
        self.refresh((self._size, state, timing, arch))


class Banner(Surface):
    def __init__(self, master, status):
        super().__init__(master, height=108)
        self.status = status
        self.state, self.sub, self.target, self.shown = "off", "", 0.0, 0.0

    def animate(self):
        diff = self.target - self.shown
        if abs(diff) > .004:
            self.shown += diff * .4
        elif diff:
            self.shown = self.target
        self.update_view()

    def update_view(self):
        self.refresh((self._size, self.state, self.status.cget("text"), self.sub, round(self.shown * 200)))

    def redraw(self):
        self.begin()
        w, h = self._size
        if w < 200:
            return
        caption, color, icon = STATES.get(self.state, STATES["idle"])
        self.panel(0, 0, w, h, 18, CARD, mix(BORDER, color, .55 if self.state != "off" else 0))
        self.icon(20, (h - 54) // 2, ("badge", icon, color, 54), lambda: badge_icon(icon, color, 54))
        x = 90
        self.label(x, 22, caption, 8, color, True)
        # Các bước quét (căn phải)
        if w >= 540:
            step = STEP_OF.get(self.state, -1)
            if self.state in ("off", "loading", "error"):
                step = -1
            sx = w - 20
            for i in reversed(range(3)):
                tw = self.measure(STEPS[i], 8, i == step)
                sx -= tw
                is_done, is_now = 0 <= step > i, i == step
                tint = EMERALD if (is_done or (is_now and self.state == "locked")) else (color if is_now else DIM)
                self.label(sx, 22, STEPS[i], 8, tint if (is_done or is_now) else DIM, is_now)
                sx -= 8 + 18
                self.icon(sx, 13, ("dot", 18, tint, is_now or is_done),
                          lambda t=tint, f=(is_now or is_done): dot_img(18, t if f else None, None if f else DIM))
                self.label(sx + 9, 22, "✓" if is_done else str(i + 1), 8, "#03140D" if (is_now or is_done) else DIM, True,
                           anchor="center")
                if i:
                    self.create_line(sx - 12, 22, sx - 2, 22, fill=BORDER)
                sx -= 16
        text = self.status.cget("text")
        size = 14 if len(text) <= 58 else 11
        self.create_text(x, 38, text=text, fill=INK, anchor="nw", font=self.font(size, True), width=w - x - 22)
        if self.sub:
            self.label(x, 62, self.sub, 10, MUTED, anchor="nw")
        if self.state in ("calibrating", "scanning", "locked") or self.shown > .004:
            self.bar(x, h - 18, w - x - 22, 6, self.shown, color)


class CameraView(Surface):
    def __init__(self, master):
        super().__init__(master, bg=BG)
        self.last, self.photo, self.accent = None, None, BORDER

    def set_accent(self, color):
        if color != self.accent:
            self.accent = color
            self.paint_panel()

    def paint_panel(self):
        self.delete("panel")
        w, h = self._size
        if w > 60:
            self.panel(0, 0, w, h, 20, "#080E17", self.accent, ow=2 if self.accent != BORDER else 1, tags="panel")
            self.tag_lower("panel")

    def redraw(self):
        self.begin()
        self.photo = None
        w, h = self._size
        if w < 60:
            return
        self.paint_panel()
        if self.last is None:
            cx, cy = w // 2, h // 2
            self.icon(cx, cy - 46, ("badge", "camera_off", DIM, 72), lambda: badge_icon("camera_off", DIM, 72), anchor="center")
            self.label(cx, cy + 12, "Bật camera để thấy vùng quét", 15, INK, True, anchor="center", tags="hint")
            self.label(cx, cy + 40, "Đặt webcam cố định · dọn trống mặt bàn · ghi nhận nền", 10, MUTED, anchor="center", tags="hint")
        else:
            self.show(self.last)

    def clear(self):
        self.last = None
        self.set_accent(BORDER)
        self.redraw()

    def show(self, frame):
        self.last = frame
        W, H = self._size
        if W < 60:
            return
        fh, fw = frame.shape[:2]
        scale = min((W - 10) / fw, (H - 10) / fh)
        w, h = max(1, int(fw * scale)), max(1, int(fh * scale))
        small = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        img = Image.fromarray(small).convert("RGBA")
        img.putalpha(round_mask(w, h, 15))
        if self.photo is not None and self.photo.width() == w and self.photo.height() == h:
            self.photo.paste(img)
            self.coords("frame", W // 2, H // 2)
            return
        self.delete("frame")
        self.photo = ImageTk.PhotoImage(img)
        self.create_image(W // 2, H // 2, image=self.photo, tags="frame")


class BinCard(Surface):
    INSET = 5

    def __init__(self, master, label, name):
        super().__init__(master, height=102)
        self.label_key, self.name = label, name
        self.color = COLORS.get(label, CYAN)
        self.target = self.shown = 0.0
        self.active = False

    def set(self, prob, active):
        self.target = prob
        if active != self.active:
            self.active = active
            self.update_view()

    def animate(self):
        diff = self.target - self.shown
        if abs(diff) > .004:
            self.shown += diff * .4
        elif diff:
            self.shown = self.target
        self.update_view()

    def update_view(self):
        self.refresh((self._size, round(self.shown * 100), self.active))

    def redraw(self):
        self.begin()
        W, H = self._size
        if W < 120 or H < 56:
            return
        I, color, on = self.INSET, self.color, self.active
        x0, y0, w, h = I, I, W - 2 * I, H - 2 * I
        if on:
            self.panel(x0, y0, w, h, 16, mix(CARD, color, .15), color, ow=2, glow=color, gpx=I)
        else:
            self.panel(x0, y0, w, h, 16, CARD, BORDER)
        compact = w < 270
        isize = min(h - 20, 44 if compact else 56)
        pad = 12 if compact else 16
        tint = color if on else mix(CARD, color, .65)
        self.icon(x0 + pad, y0 + (h - isize) // 2, ("bin", self.label_key, tint, isize),
                  lambda: bin_icon(self.label_key, tint, isize))
        tx, right = x0 + pad + isize + (10 if compact else 14), x0 + w - pad
        fsize = 11 if compact else 14
        top = y0 + (h - 42) / 2
        ny = top + 10
        self.label(tx, ny, self.name, fsize, INK, True)
        if on:
            name_end = tx + self.measure(self.name, fsize, True) + 6
            for text in ("BỎ VÀO THÙNG NÀY", "THÙNG NÀY", "CHỌN"):
                bw = self.measure(text, 8, True) + 16
                if right - bw >= name_end:
                    self.panel(right - bw, round(ny) - 9, bw, 18, 9, color, None)
                    self.label(right - bw / 2, ny, text, 8, "#03121A", True, anchor="center")
                    break
        pct_space = 38 if compact else 44
        bar_w = max(24, right - pct_space - tx)
        by = round(top + 32)
        self.bar(tx, by, bar_w, 7, self.shown, color)
        self.label(right, by + 3, f"{self.shown:.0%}", 10 if compact else 11, color if on else MUTED, True, anchor="e")


class StatsCard(Surface):
    INSET, ROW = 5, 34

    def __init__(self, master, on_archive=None, on_entry=None):
        super().__init__(master, height=268)
        self.counts, self.log = {k: 0 for k in LABELS}, []
        self.archive_count = 0
        self.on_archive, self.on_entry = on_archive, on_entry
        self.hits, self.hover = [], None
        self.bind("<Motion>", lambda e: self._set_hover(self._hit(e.x, e.y)))
        self.bind("<Leave>", lambda e: self._set_hover(None))
        self.bind("<ButtonRelease-1>", self._click)

    def _signature(self):
        return self._size, tuple(self.counts.values()), tuple(self.log[:3]), self.archive_count

    def set(self, counts, log):
        self.counts, self.log = dict(counts), list(log)
        self.refresh(self._signature())

    def set_archive_count(self, n):
        self.archive_count = n
        self.refresh(self._signature())

    def invalidate(self):
        self._sig = None
        self.redraw()

    def _hit(self, x, y):
        for x1, y1, x2, y2, kind, payload in self.hits:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return kind, payload
        return None

    def _set_hover(self, hit):
        if hit != self.hover:
            self.hover = hit
            tk.Canvas.configure(self, cursor="hand2" if hit else "")
            self.redraw()

    def _click(self, event):
        hit = self._hit(event.x, event.y)
        if hit and hit[0] == "archive" and self.on_archive:
            self.on_archive()
        elif hit and hit[0] == "row" and self.on_entry:
            self.on_entry(hit[1])

    def redraw(self):
        self.begin()
        self.hits = []
        W, H = self._size
        if W < 200 or H < 120:
            return
        I = self.INSET
        x0, y0, w, h = I, I, W - 2 * I, H - 2 * I
        self.panel(x0, y0, w, h, 18, CARD, BORDER)
        pad = 14
        self.label(x0 + pad, y0 + 20, "THỐNG KÊ", 9, MUTED, True)
        total_val = sum(self.counts.values())
        tot_txt = f"Tổng: {total_val}"
        tw_tot = self.measure(tot_txt, 8, True) + 18
        tx_tot = x0 + pad + self.measure("THỐNG KÊ", 9, True) + 10
        self.panel(tx_tot, y0 + 10, tw_tot, 20, 10, CARD2, "#2E4262")
        self.label(tx_tot + tw_tot / 2, y0 + 20, tot_txt, 8, INK, True, anchor="center")
        # Nút pill "Kho ảnh (N)" góc phải hàng tiêu đề
        text = f"Kho ảnh ({self.archive_count})"
        bw, bh = self.measure(text, 9, True) + 44, 24
        bx, by = x0 + w - pad - bw, y0 + 8
        hot = self.hover == ("archive", None)
        self.panel(bx, by, bw, bh, bh // 2, "#1E3350" if hot else "#17263B", mix("#2D4466", CYAN, .7) if hot else "#2D4466")
        fg = CYAN if hot else (mix(MUTED, CYAN, .55) if self.archive_count else MUTED)
        self.icon(bx + 12, by + 5, ("glyph", "gallery", 14, fg), lambda: glyph("gallery", 14, fg))
        self.label(bx + 32, by + bh // 2 - 1, text, 9, INK if hot else "#CBD5E1", True)
        self.hits.append((bx, by, bx + bw, by + bh, "archive", None))
        # Lưới 3x2 gọn đẹp cho đủ 6 loại rác
        cols, gap, tile_h = 3, 7, 30
        tw = (w - 2 * pad - (cols - 1) * gap) // cols
        ty = y0 + 38
        items_cnt = [(k, LABELS[k], COLORS.get(k, CYAN), self.counts.get(k, 0)) for k in LABELS]
        for i, (key, name, color, value) in enumerate(items_cnt):
            r_idx, c_idx = divmod(i, cols)
            tx = x0 + pad + c_idx * (tw + gap)
            py = ty + r_idx * (tile_h + gap)
            self.panel(tx, py, tw, tile_h, 10, CARD2, mix(BORDER, color, .35))
            self.icon(tx + 9, py + tile_h // 2, ("dot", color, 7), lambda c=color: dot_img(7, c), anchor="w")
            self.label(tx + 21, py + tile_h // 2, name, 8, MUTED, True)
            self.label(tx + tw - 9, py + tile_h // 2, str(value), 10, color, True, anchor="e")
        n_rows_cnt = -(-len(items_cnt) // cols)
        ly = ty + n_rows_cnt * tile_h + (n_rows_cnt - 1) * gap + 15
        self.label(x0 + pad, ly, "NHẬT KÝ GẦN ĐÂY", 9, MUTED, True)
        rs, RH = ly + 13, self.ROW
        rows = max(1, min(3, (y0 + h - 8 - rs) // RH))
        if not self.log:
            self.label(x0 + w / 2, rs + (rows * RH) // 2, "Chưa có lượt phân loại. Hãy đặt vật vào khung",
                       9, DIM, anchor="center")
        visible = self.log[:rows]
        thumbs = []
        for entry in visible:
            key = entry[1]
            rel = entry[3] if len(entry) > 3 else None
            color = COLORS.get(key, CYAN)
            thumbs.append(load_thumb(kho_dir() / rel, 38, 26, 6, color) if rel else None)
        for i, entry in enumerate(visible):
            stamp, key, prob = entry[:3]
            rel = entry[3] if len(entry) > 3 else None
            thumb = thumbs[i]
            top, color = rs + i * RH, COLORS.get(key, CYAN)
            cy = top + RH // 2
            hot = (thumb is not None) and self.hover == ("row", rel)
            if hot:
                self.panel(x0 + 8, top + 1, w - 16, RH - 2, 10, mix(CARD, INK, .06), None)
            if thumb is not None:
                self.hits.append((x0 + 8, top, x0 + w - 8, top + RH, "row", rel))
                self.put(x0 + pad, top + (RH - 26) // 2, ("thumb", rel, color), lambda t=thumb: t)
            else:
                self.panel(x0 + pad, top + (RH - 26) // 2, 38, 26, 6, CARD2, BORDER)
                self.icon(x0 + pad + 12, cy - 7, ("glyph", "gallery", 14, DIM), lambda: glyph("gallery", 14, DIM))
            tx = x0 + pad + 38 + 12
            self.label(tx, cy, stamp, 9, MUTED if hot else DIM, family="Consolas")
            self.icon(tx + 68, cy, ("dot", color, 8), lambda c=color: dot_img(8, c), anchor="w")
            self.label(tx + 84, cy, LABELS.get(key, key), 10, INK, True)
            if prob is not None:
                self.label(x0 + w - pad, cy, f"{prob:.0%}", 10, color, True, anchor="e")
            nxt_rel = visible[i + 1][3] if i + 1 < len(visible) and len(visible[i + 1]) > 3 else None
            nxt_hot = i + 1 < len(visible) and thumbs[i + 1] is not None and self.hover == ("row", nxt_rel)
            if i < len(visible) - 1 and not hot and not nxt_hot:
                self.create_line(x0 + pad, top + RH, x0 + w - pad, top + RH, fill="#1A283B")
        if self.hover and (self.hover[0], self.hover[1]) not in [(k[4], k[5]) for k in self.hits]:
            self.hover = None
            tk.Canvas.configure(self, cursor="")


class ArchiveTitle(Surface):
    def __init__(self, master):
        super().__init__(master, height=76)
        self.count = 0

    def set_count(self, n):
        self.count = n
        self.redraw()

    def redraw(self):
        self.begin()
        w, h = self._size
        if w < 100:
            return
        self.icon(24, (h - 44) // 2, ("badge", "gallery", EMERALD, 44), lambda: badge_icon("gallery", EMERALD, 44))
        self.label(82, h // 2 - 11, "Kho Minh Chứng Phân Loại Rác", 15, INK, True)
        self.label(82, h // 2 + 14, f"{self.count} ảnh đang lưu  ·  outputs/kho_rac", 9, MUTED)


class ArchiveTabs(Surface):
    """Thanh tab lọc: Tất cả / Nhựa / Kim loại / Bìa carton, có điểm nhấn màu theo vật liệu."""

    def __init__(self, master, on_select):
        super().__init__(master, height=58)
        self.on_select, self.active, self.counts = on_select, "all", {}
        self.hits, self.hover = [], None
        self.bind("<Motion>", lambda e: self._set_hover(self._hit(e.x, e.y)))
        self.bind("<Leave>", lambda e: self._set_hover(None))
        self.bind("<ButtonRelease-1>", self._click)

    def set_counts(self, counts):
        self.counts = counts
        self.redraw()

    def _hit(self, x, y):
        for x1, y1, x2, y2, key in self.hits:
            if x1 <= x <= x2 and y1 <= y <= y2:
                return key
        return None

    def _set_hover(self, key):
        if key != self.hover:
            self.hover = key
            tk.Canvas.configure(self, cursor="hand2" if key else "")
            self.redraw()

    def _click(self, event):
        key = self._hit(event.x, event.y)
        if key and key != self.active:
            self.active = key
            self.redraw()
            self.on_select(key)

    def redraw(self):
        self.begin()
        self.hits = []
        w, h = self._size
        if w < 100:
            return
        x, y, bh, gap = 24, 10, 34, 8
        tabs = [("all", "Tất cả", EMERALD)] + [(k, LABELS[k], COLORS.get(k, CYAN)) for k in LABELS]
        rows_used = 1
        for key, name, color in tabs:
            text = f"{name} ({self.counts.get(key, 0)})"
            on, hot = key == self.active, key == self.hover
            bw = self.measure(text, 9, True) + (44 if key != "all" else 30)
            if x > 24 and x + bw > w - 24:
                x = 24
                y += bh + gap
                rows_used += 1
            fill = mix(BG, color, .20) if on else (mix(BG, color, .08) if hot else CARD)
            outline = color if on else (mix(BORDER, color, .5) if hot else BORDER)
            self.panel(x, y, bw, bh, bh // 2, fill, outline, ow=2 if on else 1)
            tx = x + 15
            cy = y + bh // 2
            if key != "all":
                self.icon(tx, cy, ("dot", color, 8), lambda c=color: dot_img(8, c), anchor="w")
                tx += 15
            self.label(tx, cy - 1, text, 9, color if on else (INK if hot else MUTED), True)
            self.hits.append((x, y, x + bw, y + bh, key))
            x += bw + gap
        needed_h = 20 + rows_used * bh + (rows_used - 1) * gap
        if int(self.cget("height")) != needed_h:
            tk.Canvas.configure(self, height=needed_h)
        self.create_line(0, max(h, needed_h) - 1, w, max(h, needed_h) - 1, fill=BORDER)


class ArchiveGrid(Surface):
    """Lưới thẻ ảnh minh chứng, cuộn mượt bằng con lăn chuột (chỉ vẽ thẻ đang nhìn thấy)."""
    CW, CH, GAP, TOP, SIDE = 252, 244, 18, 20, 24

    def __init__(self, master, on_open):
        super().__init__(master, bg=BG)
        self.items, self.on_open, self.empty = [], on_open, ("Kho chưa có ảnh minh chứng", "")
        self.off = self.tgt = 0.0
        self.hover, self.pt, self._anim = None, None, None
        self.bind("<Motion>", self._motion)
        self.bind("<Leave>", self._leave)
        self.bind("<ButtonRelease-1>", self._click)

    def destroy(self):
        if self._anim is not None:
            self.after_cancel(self._anim)
            self._anim = None
        super().destroy()

    def set_items(self, items, empty=None, reset=False):
        self.items = items
        if empty:
            self.empty = empty
        if reset:
            self.off = self.tgt = 0.0
        self._clamp()
        self.redraw()

    def _geom(self):
        W, H = self._size
        cols = max(1, (W - 2 * self.SIDE + self.GAP) // (self.CW + self.GAP))
        left = self.SIDE
        rows = -(-len(self.items) // cols)
        content = self.TOP * 2 + rows * (self.CH + self.GAP) - self.GAP if rows else 0
        return cols, left, max(0, content - H)

    def _clamp(self):
        top = self._geom()[2]
        self.tgt, self.off = min(max(self.tgt, 0), top), min(max(self.off, 0), top)

    def _card_at(self, x, y):
        cols, left, _ = self._geom()
        dx, dy = x - left, y + self.off - self.TOP
        if dx < 0 or dy < 0:
            return None
        col, row = int(dx // (self.CW + self.GAP)), int(dy // (self.CH + self.GAP))
        if col >= cols or dx - col * (self.CW + self.GAP) > self.CW or dy - row * (self.CH + self.GAP) > self.CH:
            return None
        i = row * cols + col
        return i if i < len(self.items) else None

    # ── tương tác ──
    def wheel(self, event):
        self.tgt = min(max(self.tgt - event.delta / 120 * 96, 0), self._geom()[2])
        if self._anim is None:
            self._tick()

    def _tick(self):
        self._anim = None
        diff = self.tgt - self.off
        if abs(diff) < .6:
            self.off = self.tgt
        else:
            self.off += diff * .3
            self._anim = self.after(16, self._tick)
        self.redraw()

    def _motion(self, event):
        self.pt = (event.x, event.y)
        hot = self._card_at(*self.pt)
        if hot != self.hover:
            tk.Canvas.configure(self, cursor="hand2" if hot is not None else "")
            self.redraw()

    def _leave(self, _event):
        self.pt = None
        if self.hover is not None:
            tk.Canvas.configure(self, cursor="")
            self.redraw()

    def _click(self, event):
        i = self._card_at(event.x, event.y)
        if i is not None:
            self.on_open(self.items[i])

    # ── vẽ ──
    def redraw(self):
        self.begin()
        W, H = self._size
        if W < 100 or H < 60:
            return
        if not self.items:
            cx, cy = W // 2, H // 2
            self.icon(cx, cy - 50, ("badge", "gallery", DIM, 76), lambda: badge_icon("gallery", DIM, 76), anchor="center")
            self.label(cx, cy + 10, self.empty[0], 14, INK, True, anchor="center")
            self.label(cx, cy + 36, self.empty[1] or "Ảnh sẽ tự lưu tại đây mỗi khi trạm chốt nhãn một vật", 10, MUTED,
                       anchor="center")
            return
        cols, left, top_max = self._geom()
        off = round(self.off)
        self.hover = self._card_at(*self.pt) if self.pt else None
        CW, CH = self.CW, self.CH
        for i, ev in enumerate(self.items):
            row, col = divmod(i, cols)
            cx, cy = left + col * (CW + self.GAP), self.TOP + row * (CH + self.GAP) - off
            if cy + CH < 0 or cy > H:
                continue
            color = COLORS.get(ev.label, CYAN)
            hot = i == self.hover
            self.panel(cx, cy, CW, CH, 16, mix(CARD, INK, .04) if hot else CARD, mix(BORDER, color, .75) if hot else BORDER,
                       ow=2 if hot else 1)
            thumb = load_thumb(ev.path, 228, 166, 11, None)
            if thumb is not None:
                self.put(cx + 12, cy + 12, ("card", ev.rel), lambda t=thumb: t)
            else:
                self.panel(cx + 12, cy + 12, 228, 166, 11, CARD2, BORDER)
                self.icon(cx + 126, cy + 95, ("badge", "alert", DIM, 40), lambda: badge_icon("alert", DIM, 40), anchor="center")
            name = LABELS.get(ev.label, ev.label)
            text = name + (f" · {ev.conf:.0%}" if ev.conf is not None else "")
            bw, by = self.measure(text, 9, True) + 34, cy + 188
            self.panel(cx + 12, by, bw, 22, 11, mix(CARD, color, .16), mix(CARD, color, .55))
            self.icon(cx + 23, by + 11, ("dot", color, 8), lambda c=color: dot_img(8, c), anchor="w")
            self.label(cx + 38, by + 10, text, 9, color, True)
            stamp = ev.ts.strftime("%d/%m/%Y %H:%M:%S") if ev.ts and ev.ts != datetime.min else "Không rõ thời gian"
            self.label(cx + 14, cy + 226, stamp, 9, MUTED if not hot else INK, family="Consolas")
        if top_max > 0:  # thanh cuộn mảnh
            th = max(36, round(H * H / (H + top_max)))
            ty = round(off / top_max * (H - th - 8)) + 4
            self.panel(W - 9, ty, 4, th, 2, "#2D4466", None)


class ViewerInfo(Surface):
    def __init__(self, master, ev, width):
        super().__init__(master, width=width, height=102)
        self.ev = ev

    def redraw(self):
        self.begin()
        w, h = self._size
        if w < 300:
            return
        ev = self.ev
        color, name = COLORS.get(ev.label, CYAN), LABELS.get(ev.label, ev.label)
        self.panel(0, 0, w, h, 16, CARD, BORDER)
        self.icon(16, 9, ("bin", ev.label, color, 48), lambda: bin_icon(ev.label, color, 48))
        self.label(76, 22, name, 14, INK, True)
        self.label(76, 44, f"Thùng {name.lower()}", 9, MUTED)
        conf = f"{ev.conf:.0%}" if ev.conf is not None else "—"
        self.label(w - 18, 22, conf, 20, color, True, anchor="e")
        self.label(w - 18, 46, "độ tự tin", 8, MUTED, anchor="e")
        cw = self.measure(conf, 20, True)
        stamp = ev.ts if ev.ts and ev.ts != datetime.min else None
        self.label(w - 18 - cw - 26, 22, stamp.strftime("%d/%m/%Y") if stamp else "—", 11, INK, True, anchor="e")
        self.label(w - 18 - cw - 26, 46, stamp.strftime("%H:%M:%S") if stamp else "", 9, MUTED, anchor="e",
                   family="Consolas")
        self.create_line(16, 68, w - 16, 68, fill=BORDER)
        self.label(18, 85, f"outputs/kho_rac/{ev.rel}", 8, DIM, family="Consolas")


class EvidenceViewer(tk.Toplevel):
    """Hộp thoại xem ảnh minh chứng phóng to, kèm thông tin, Mở thư mục và Xóa ảnh này."""

    def __init__(self, master, ev, on_deleted=None):
        super().__init__(master, bg=BG)
        self.ev, self.on_deleted = ev, on_deleted
        self._armed, self._timer, self.photo = False, None, None
        self.title("Xem Ảnh Minh Chứng")
        self.resizable(False, False)
        color = COLORS.get(ev.label, CYAN)
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        try:
            with Image.open(ev.path) as im:
                im = im.convert("RGB")
        except (OSError, ValueError):
            im = None
        if im is not None:
            k = min(min(820, sw - 160) / im.width, min(540, sh - 330) / im.height, 1.6)
            w, h = max(1, round(im.width * k)), max(1, round(im.height * k))
            pic = im.resize((w, h), Image.LANCZOS).convert("RGBA")
            pic.putalpha(round_mask(w, h, 14))
            ring = Image.new("RGBA", (w * SS, h * SS), rgb(color) + (0,))
            ImageDraw.Draw(ring).rounded_rectangle((0, 0, w * SS - 1, h * SS - 1), 14 * SS, outline=rgb(color) + (255,),
                                                   width=2 * SS)
            pic = Image.alpha_composite(pic, ring.resize((w, h), Image.LANCZOS))
            stage = Image.new("RGBA", (w, h), rgb(BG) + (255,))
            stage.alpha_composite(pic)
            self.photo = ImageTk.PhotoImage(stage)
            tk.Label(self, image=self.photo, bg=BG, bd=0).pack(padx=20, pady=(20, 12))
        else:
            w, h = 640, 360
            tk.Label(self, text="Không đọc được ảnh minh chứng này", bg=BG, fg=MUTED, font=(FONT, 12, "bold"),
                     width=48, height=10).pack(padx=20, pady=(20, 12))
        total = max(w, 560)
        ViewerInfo(self, ev, total).pack(padx=20)
        bar = tk.Frame(self, bg=BG)
        bar.pack(fill="x", padx=20, pady=(12, 20))
        PillButton(bar, "Mở thư mục", lambda: open_folder(ev.path.parent), "secondary", "folder").pack(side="left", padx=(0, 10))
        self.delete_btn = PillButton(bar, "Xóa ảnh này", self._delete, "secondary", "trash")
        self.delete_btn.pack(side="left")
        PillButton(bar, "Đóng", self.destroy, "ghost").pack(side="right")
        self.bind("<Escape>", lambda e: self.destroy())
        self.transient(master)
        self.update_idletasks()
        mx, my = master.winfo_rootx() + master.winfo_width() // 2, master.winfo_rooty() + master.winfo_height() // 2
        x = max(0, min(sw - self.winfo_reqwidth(), mx - self.winfo_reqwidth() // 2))
        y = max(0, min(sh - self.winfo_reqheight() - 40, my - self.winfo_reqheight() // 2))
        self.geometry(f"+{x}+{y}")
        self.focus_set()

    def destroy(self):
        if self._timer is not None:
            self.after_cancel(self._timer)
            self._timer = None
        super().destroy()

    def _disarm(self):
        self._timer, self._armed = None, False
        self.delete_btn.config(text="Xóa ảnh này", style="secondary")

    def _delete(self):
        if not self._armed:  # bấm lần 1 để xác nhận, lần 2 mới xóa
            self._armed = True
            self.delete_btn.config(text="Bấm lần nữa để xóa", style="danger")
            self._timer = self.after(3000, self._disarm)
            return
        if delete_evidence(self.ev) and self.on_deleted:
            self.on_deleted()
        self.destroy()


class ArchiveWindow(tk.Toplevel):
    """Cửa sổ Kho Minh Chứng: tab lọc + lưới thẻ ảnh, đồng bộ bảng màu Eco-Tech Deep Slate."""

    def __init__(self, master, on_change=None):
        super().__init__(master, bg=BG)
        self.on_change, self.items, self.tab = on_change, [], "all"
        self.title("Kho Minh Chứng")
        self.geometry("840x660")
        self.minsize(740, 520)
        head = tk.Frame(self, bg=BG)
        head.pack(fill="x")
        self.title_bar = ArchiveTitle(head)
        self.title_bar.pack(side="left", fill="x", expand=True)
        PillButton(head, "Làm mới", self.reload, "secondary", "reset").pack(side="right", padx=(0, 24), pady=18)
        PillButton(head, "Mở thư mục", lambda: open_folder(kho_dir()), "secondary", "folder").pack(side="right", padx=(0, 10), pady=18)
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")
        self.tabs = ArchiveTabs(self, self._select)
        self.tabs.pack(fill="x")
        self.grid_view = ArchiveGrid(self, self._open)
        self.grid_view.pack(fill="both", expand=True)
        self.bind("<MouseWheel>", self.grid_view.wheel)
        self.bind("<Escape>", lambda e: self.destroy())
        self.transient(master)
        self.update_idletasks()
        self.geometry(f"+{master.winfo_rootx() + 40}+{master.winfo_rooty() + 30}")
        self.reload()

    def reload(self):
        self.items = load_archive()
        counts = {"all": len(self.items), **{k: sum(e.label == k for e in self.items) for k in LABELS}}
        self.title_bar.set_count(len(self.items))
        self.tabs.set_counts(counts)
        self._apply()

    def _apply(self, reset=False):
        shown = self.items if self.tab == "all" else [e for e in self.items if e.label == self.tab]
        empty = ("Kho chưa có ảnh minh chứng", "") if (not self.items or self.tab == "all") else \
            (f"Chưa có ảnh {LABELS[self.tab].lower()}", "Hãy chọn tab khác hoặc đặt thêm vật vào khung")
        self.grid_view.set_items(shown, empty, reset)

    def _select(self, key):
        self.tab = key
        self._apply(reset=True)

    def _open(self, ev):
        EvidenceViewer(self, ev, on_deleted=self._deleted)

    def _deleted(self):
        if self.on_change:
            self.on_change()
        else:
            self.reload()


# ───────────────────────────── Webcam + HUD ─────────────────────────────
def open_camera(index):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cam = cv2.VideoCapture(index, backend)
        if cam.isOpened():
            cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            ok, frame = cam.read()
            if ok and frame is not None:
                return cam, frame
        cam.release()
    raise OSError(f"Không mở được camera {index}. Hãy đóng ứng dụng dùng camera, kiểm tra quyền Camera của Windows hoặc thử số khác.")


HUD_EMERALD, HUD_AMBER, HUD_ROSE = (129, 185, 16), (11, 158, 245), (94, 63, 244)  # BGR
HUD_INK, HUD_PANEL = (249, 245, 241), (30, 19, 11)
HUD_CHIP = {"uncalibrated": "NO BACKGROUND", "calibrating": "CALIBRATING", "idle": "READY", "blocked": "BLOCKED",
            "moving": "HOLD STILL", "hand": "HAND DETECTED", "scanning": "SCANNING", "unknown": "UNCLEAR"}


def draw_hud(frame, reading):
    """Vẽ HUD trải rộng toàn màn hình video (toàn khung hình đã được quét trước đó)."""
    state = getattr(reading, "state", "idle")
    color = HUD_EMERALD if (reading.label or state == "locked") else HUD_ROSE if state in ("hand", "blocked") else HUD_AMBER
    h, w = frame.shape[:2]
    pad = max(6, round(min(w, h) * .018))
    x1, y1, x2, y2 = pad, pad, w - 1 - pad, h - 1 - pad
    rw, rh = max(1, x2 - x1), max(1, y2 - y1)

    # Lưới 1/3 mảnh trên toàn khung video (không làm tối viền ngoài)
    grid = frame.copy()
    for k in (1, 2):
        px = x1 + rw * k // 3
        py = y1 + rh * k // 3
        cv2.line(grid, (px, y1), (px, y2), (255, 255, 255), 1, cv2.LINE_AA)
        cv2.line(grid, (x1, py), (x2, py), (255, 255, 255), 1, cv2.LINE_AA)
    out = cv2.addWeighted(grid, .13, frame, .87, 0)
    cv2.rectangle(out, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)
    # 4 góc ngắm sát 4 góc màn hình video
    arm, t = max(24, round(min(rw, rh) * .10)), 3
    for cx, dx in ((x1, 1), (x2, -1)):
        for cy, dy in ((y1, 1), (y2, -1)):
            cv2.line(out, (cx, cy), (cx + dx * arm, cy), color, t, cv2.LINE_AA)
            cv2.line(out, (cx, cy), (cx, cy + dy * arm), color, t, cv2.LINE_AA)
    # Khung mảnh quanh vật thể phát hiện được
    if reading.bbox and state in ("scanning", "locked", "unknown", "hand"):
        bx1, by1, bx2, by2 = (int(v) for v in reading.bbox)
        bx1, by1 = max(x1 + 2, bx1), max(y1 + 2, by1)
        bx2, by2 = min(x2 - 2, bx2), min(y2 - 2, by2)
        if bx2 > bx1 + 4 and by2 > by1 + 4:
            seg = max(10, min(bx2 - bx1, by2 - by1) // 4)
            for px, dx in ((bx1, 1), (bx2, -1)):
                for py, dy in ((by1, 1), (by2, -1)):
                    cv2.line(out, (px, py), (px + dx * seg, py), color, 2, cv2.LINE_AA)
                    cv2.line(out, (px, py), (px, py + dy * seg), color, 2, cv2.LINE_AA)
    # Nhãn trạng thái ở góc trên trái
    if state == "locked" and reading.label:
        score = (reading.probs or {}).get(reading.label)
        text = reading.label.upper() + (f" {score:.0%}" if score is not None else "")
    elif state == "calibrating":
        text = f"CALIBRATING {round(reading.progress * 12)}/12"
    else:
        text = HUD_CHIP.get(state, "READY")
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .52, 1)
    cx, cy = x1 + 14, y1 + 14
    cv2.rectangle(out, (cx, cy), (cx + tw + 34, cy + th + 14), HUD_PANEL, -1)
    cv2.rectangle(out, (cx, cy), (cx + tw + 34, cy + th + 14), color, 1, cv2.LINE_AA)
    cv2.circle(out, (cx + 12, cy + (th + 14) // 2), 4, color, -1, cv2.LINE_AA)
    cv2.putText(out, text, (cx + 24, cy + th + 7), cv2.FONT_HERSHEY_SIMPLEX, .52, HUD_INK, 1, cv2.LINE_AA)
    # Thanh tiến trình nhỏ ở đáy màn hình video
    if state in ("calibrating", "scanning") and reading.progress > 0:
        bw, bh = round(rw * .42), 6
        bx, by = x1 + (rw - bw) // 2, y2 - 22
        cv2.rectangle(out, (bx - 2, by - 2), (bx + bw + 2, by + bh + 2), HUD_PANEL, -1)
        cv2.rectangle(out, (bx, by), (bx + round(bw * min(1., reading.progress)), by + bh), color, -1)
    return out


def camera_loop(index, checkpoint, messages, stop, calibrate):
    cam = None
    final = "Đã tắt camera"

    def send(packet):
        try:
            messages.get_nowait()
        except queue.Empty:
            pass
        messages.put_nowait(packet)
    durations = []
    try:
        cv2.setNumThreads(min(8, os.cpu_count() or 4))
        predictor = Predictor(checkpoint)
        arch = getattr(predictor, "arch", "")
        scanner = Scanner(predictor, PalmDetector())
        if stop.is_set():
            return
        cam, frame = open_camera(index)
        recent = deque(maxlen=30)
        while not stop.is_set():
            start = time.perf_counter()
            # Camera cố định. Quét toàn bộ khung hình TRƯỚC khi vẽ HUD lên bản hiển thị.
            frame = cv2.flip(frame, 1)
            if calibrate.is_set():
                scanner.calibrate()
                calibrate.clear()
            reading = scanner.step(frame)
            preview = cv2.cvtColor(draw_hud(frame, reading), cv2.COLOR_BGR2RGB)
            end = time.perf_counter()
            ms = (end-start)*1000
            durations.append(ms)
            recent.append(end)
            fps = (len(recent)-1)/(recent[-1]-recent[0]) if len(recent) > 1 else 0.
            send(("frame", preview, reading, f"Xử lý {ms:.1f} ms   ·   {fps:.1f} khung/giây", arch))
            ok, frame = cam.read()
            if not ok or frame is None:
                raise OSError("Mất tín hiệu webcam. Kiểm tra kết nối rồi bật lại camera.")
    except Exception as error:
        final = f"{error}"
    finally:
        if cam is not None:
            cam.release()
        if durations:
            folder = PROJECT / "outputs" / "webcam"
            try:
                folder.mkdir(parents=True, exist_ok=True)
                with (folder / "sessions.csv").open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if f.tell() == 0:
                        writer.writerow(["local_time", "camera", "frames", "p50_processing_ms", "p95_processing_ms"])
                    writer.writerow([datetime.now().isoformat(timespec="seconds"), index, len(durations),
                                     *np.percentile(durations, [50, 95]).round(2)])
            except OSError as error:
                final += f" · Không ghi được số liệu: {error}"
        send(("end", final))


# ───────────────────────────── Ứng dụng ─────────────────────────────
RELEASE_STATES = {"idle", "uncalibrated", "calibrating", "blocked", "hand"}


class Station:
    def __init__(self, root, checkpoint, camera=0):
        self.root, self.checkpoint, self.camera_index = root, checkpoint, camera
        root.title("Trạm hướng dẫn phân loại rác")
        root.configure(bg=BG)
        root.geometry("1240x740")
        root.minsize(980, 640)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)
        self.worker = None
        self.stop, self.calibrate = threading.Event(), threading.Event()
        self.messages = queue.Queue(maxsize=1)
        self.close_requested = False
        self.state, self.arch = "off", ""
        self.counts, self.log, self.armed = {k: 0 for k in LABELS}, [], None

        self.status = Readout("", self._status_changed)
        self.timing = Readout("", self._timing_changed)

        self.header = Header(root)
        self.header.grid(row=0, sticky="ew")
        body = tk.Frame(root, bg=BG)
        body.grid(row=1, sticky="nsew", pady=(6, 14))
        body.rowconfigure(0, weight=1)
        body.columnconfigure(0, weight=58, uniform="cols")
        body.columnconfigure(1, weight=42, uniform="cols")

        # Cột trái: camera + banner hướng dẫn + thanh công cụ
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(20, 10))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.view = CameraView(left)
        self.view.grid(row=0, sticky="nsew", pady=(7, 12))
        self.banner = Banner(left, self.status)
        self.banner.grid(row=1, sticky="ew", pady=(0, 12))
        bar = tk.Frame(left, bg=BG, height=40)
        bar.grid(row=2, sticky="ew", pady=(0, 7))
        bar.bind("<Configure>", self._toolbar_resize)
        self.toggle = PillButton(bar, "Bật camera", self.toggle_camera, "primary", "power")
        self.toggle.pack(side="left", padx=(0, 10))
        self.learn = PillButton(bar, "Ghi nhận nền trống", lambda: self.calibrate.set(), "secondary", "target",
                                state="disabled", short="Ghi nền")
        self.learn.pack(side="left")
        self.reset = PillButton(bar, "Đặt lại bộ đếm", self.reset_stats, "ghost", "reset", short="Đặt lại")
        self.reset.pack(side="right")

        # Cột phải: lưới 2x3 thẻ thùng (6 loại rác) + thống kê
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew", padx=(3, 15))
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        bin_grid = tk.Frame(right, bg=BG)
        bin_grid.grid(row=0, column=0, sticky="nsew")
        bin_cols = 2 if len(LABELS) > 3 else 1
        for c in range(bin_cols):
            bin_grid.columnconfigure(c, weight=1, uniform="bcol")
        self.bins = {}
        for i, (label, name) in enumerate(LABELS.items()):
            r_idx, c_idx = divmod(i, bin_cols)
            bin_grid.rowconfigure(r_idx, weight=1, minsize=96)
            card = BinCard(bin_grid, label, name)
            card.grid(row=r_idx, column=c_idx, sticky="nsew")
            self.bins[label] = card
        self.stats = StatsCard(right, on_archive=self.open_archive, on_entry=self.open_entry)
        self.stats.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        self.archive_win = None
        self.stats.set_archive_count(count_archive())

        if not checkpoint.is_file():
            self.show("error", "Thiếu checkpoint. Chạy scripts/train.py hoặc chép best.pt vào outputs/model.")
        else:
            self.show("off", "Đặt webcam cố định và dọn trống mặt bàn")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.timer = root.after(30, self.poll)

    # ── cập nhật giao diện ──
    def _status_changed(self):
        self.banner.update_view()

    def _timing_changed(self):
        self.header.update_view(self.state, self.timing.cget("text"), self.arch)

    def _toolbar_resize(self, event):
        compact = event.width < 560
        if self.learn.cget("compact") != compact:
            for button in (self.learn, self.reset):
                button.config(compact=compact)

    def show(self, state, text, progress=0.0, sub=""):
        self.state = state
        self.banner.state, self.banner.sub, self.banner.target = state, sub, progress
        self.status.config(text=text)
        self._timing_changed()

    def set_bins(self, reading):
        probs = getattr(reading, "probs", None) or {}
        state = getattr(reading, "state", "idle")
        for label, card in self.bins.items():
            shown = state in ("scanning", "unknown", "locked")
            card.set(float(probs.get(label, 0.)) if shown else 0.,
                     state == "locked" and reading.label == label)

    def count(self, reading, frame_rgb=None):
        """Đếm theo cạnh lên: chỉ +1 khi khóa nhãn mới sau khi bàn trống hoặc đổi vật."""
        state = getattr(reading, "state", "idle")
        if state == "locked" and reading.label:
            if reading.label != self.armed:
                self.armed = reading.label
                self.counts[reading.label] = self.counts.get(reading.label, 0) + 1
                now = datetime.now()
                prob = (getattr(reading, "probs", None) or {}).get(reading.label)
                rel = save_evidence(frame_rgb, reading.label, prob, now=now) if frame_rgb is not None else None
                self.log.insert(0, (now.strftime("%H:%M:%S"), reading.label, prob, rel))
                del self.log[20:]
                self.stats.set(self.counts, self.log)
                if rel:
                    self.refresh_archive_count()
                    if self.archive_win is not None and self.archive_win.winfo_exists():
                        self.archive_win.reload()
        elif state in RELEASE_STATES:
            self.armed = None

    def reset_stats(self):
        """Chỉ đặt lại bộ đếm + nhật ký phiên; kho ảnh trên đĩa được giữ nguyên."""
        self.counts, self.log, self.armed = {k: 0 for k in LABELS}, [], None
        self.stats.set(self.counts, self.log)

    # ── kho ảnh minh chứng ──
    def refresh_archive_count(self):
        self.stats.set_archive_count(count_archive())

    def archive_changed(self):
        self.refresh_archive_count()
        self.stats.invalidate()
        if self.archive_win is not None and self.archive_win.winfo_exists():
            self.archive_win.reload()

    def open_archive(self):
        win = self.archive_win
        if win is not None and win.winfo_exists():
            win.reload()
            win.deiconify()
            win.lift()
            win.focus_force()
        else:
            self.archive_win = ArchiveWindow(self.root, on_change=self.archive_changed)

    def open_entry(self, rel):
        ev = evidence_from_rel(rel)
        if ev is not None:
            EvidenceViewer(self.root, ev, on_deleted=self.archive_changed)

    # ── điều khiển camera ──
    def toggle_camera(self):
        if self.worker is not None:
            self.stop.set()
            self.toggle.config(state="disabled")
            self.learn.config(state="disabled")
            for card in self.bins.values():
                card.set(0., False)
            self.show("loading", "Đang tắt camera…")
            return
        self.stop, self.calibrate = threading.Event(), threading.Event()
        self.show("loading", "Đang nạp mô hình và mở camera…")
        self.toggle.config(text="Tắt camera", style="danger")
        self.worker = threading.Thread(target=camera_loop, daemon=True,
            args=(self.camera_index, self.checkpoint, self.messages, self.stop, self.calibrate))
        self.worker.start()

    def poll(self):
        try:
            packet = self.messages.get_nowait()
        except queue.Empty:
            packet = None
        if packet and packet[0] == "end":
            self.worker = None
            self.armed = None
            self.view.clear()
            self.timing.config(text="")
            for card in self.bins.values():
                card.set(0., False)
            self.show("off" if packet[1] == "Đã tắt camera" else "error", packet[1])
            self.toggle.config(text="Bật camera", style="primary", state="normal")
            self.learn.config(state="disabled")
        elif packet and not self.stop.is_set():
            _, rgb_frame, reading, timing = packet[:4]
            if len(packet) > 4 and packet[4]:
                self.arch = packet[4]
            state = getattr(reading, "state", "idle")
            self.view.show(rgb_frame)
            self.view.set_accent(EMERALD if state == "locked" else ROSE if state in ("hand", "blocked") else BORDER)
            self.timing.config(text=timing)
            sub = ""
            if state == "locked" and reading.label:
                sub = f"Hãy bỏ vào Thùng {LABELS[reading.label].lower()}"
            progress = 1.0 if state == "locked" else (reading.progress if state in ("calibrating", "scanning") else 0.0)
            self.show(state, reading.message, progress, sub)
            self.set_bins(reading)
            self.count(reading, rgb_frame)
            self.learn.config(state="normal")
        self.banner.animate()
        for card in self.bins.values():
            card.animate()
        if self.close_requested and self.worker is None:
            self.root.destroy()
        else:
            self.timer = self.root.after(25, self.poll)

    def close(self):
        self.close_requested = True
        if self.worker is not None:
            self.stop.set()
            self.root.after(2500, self.root.destroy)  # Một số driver chặn trong read().
        else:
            self.root.destroy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=BEST)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    root = tk.Tk()
    Station(root, args.checkpoint, camera=args.camera)
    root.mainloop()
