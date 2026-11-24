"""Xử lý từng vùng quét BGR; dùng chung cho webcam và kiểm thử ngoại tuyến."""
import os
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from material import LABELS, PROJECT


class PalmDetector:
    def __init__(self):
        path = PROJECT / "assets" / "palm" / "palm.onnx"
        if not path.is_file():
            raise FileNotFoundError("Thiếu assets/palm/palm.onnx. Hãy lấy lại thư mục assets từ dự án.")
        self.net = cv2.dnn.readNetFromONNX(str(path))

    def __call__(self, roi):
        # Hỗ trợ cả ROI vuông lẫn toàn khung hình chữ nhật (pad vuông để không méo tỉ lệ bàn tay).
        h, w = roi.shape[:2]
        if h != w:
            side = max(h, w)
            square = np.zeros((side, side, 3), dtype=roi.dtype)
            y0, x0 = (side - h) // 2, (side - w) // 2
            square[y0:y0 + h, x0:x0 + w] = roi
            src = square
        else:
            src = roi
        small = cv2.resize(src, (192, 192), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        self.net.setInput(rgb[None])
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        logits = next(output for output in outputs if output.shape[-1] == 1)
        return float(logits.max()) > 0.4055  # sigmoid(logit) > 0.60


class FaceDetector:
    """Phát hiện khuôn mặt nhẹ bằng Haar Cascade đi kèm OpenCV (kèm CLAHE cho phòng tối)."""

    def __init__(self):
        xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.cascade = cv2.CascadeClassifier(xml) if os.path.isfile(xml) else None
        self.clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(4, 4))

    def __call__(self, roi):
        if self.cascade is None or self.cascade.empty():
            return False
        h, w = roi.shape[:2]
        scale = 240.0 / max(1, max(h, w))
        small = cv2.resize(roi, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        gray = self.clahe.apply(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
        faces = self.cascade.detectMultiScale(gray, scaleFactor=1.12, minNeighbors=4, minSize=(36, 36))
        return len(faces) > 0


@dataclass
class Reading:
    message: str
    label: str | None = None
    probs: dict[str, float] | None = None
    progress: float = 0.0
    state: str = "idle"
    bbox: tuple[int, int, int, int] | None = None


class Scanner:
    def __init__(self, predictor, palm, face=None):
        self.predictor, self.palm = predictor, palm
        self.face = face if face is not None else FaceDetector()
        self._mild_clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        self.background = None
        self.bg_lab = None
        self.samples = []
        self.calibrating = False
        self.previous = None
        self._reset_object_state()

    def _reset_object_state(self):
        self.settled_ref = None
        self.cached_palm = None
        self.cached_face = None
        self.locked_reading = None
        self.ema_probs = None
        self.stable_count = 0
        self.last_candidate = None

    def calibrate(self):
        self.samples.clear()
        self.background = self.bg_lab = self.previous = None
        self._reset_object_state()
        self.calibrating = True

    def clear(self, text, state="idle", bbox=None):
        self._reset_object_state()
        return Reading(text, state=state, bbox=bbox)

    def _has_skin_candidate(self, small_bgr, fg_mask):
        ycrcb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1]
        cb = ycrcb[:, :, 2]
        skin = (fg_mask > 0) & (cr >= 133) & (cr <= 173) & (cb >= 77) & (cb <= 127)
        return float(skin.mean()) >= 0.015

    def _suppress_shadows(self, aligned_bgr, mask):
        """Loại bỏ bóng đổ mềm trên tường/bàn (chỉ tối đi, không đổi màu sắc và không có biên sắc nét)."""
        if self.bg_lab is None:
            return mask
        cur_lab = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2LAB).astype(np.int16)
        l_cur, a_cur, b_cur = cur_lab[:, :, 0], cur_lab[:, :, 1], cur_lab[:, :, 2]
        l_bg, a_bg, b_bg = self.bg_lab[:, :, 0], self.bg_lab[:, :, 1], self.bg_lab[:, :, 2]
        gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
        grad = np.abs(gx) + np.abs(gy)
        shadow = (
            (mask > 0)
            & (l_cur < l_bg - 8)
            & (l_cur > (l_bg * 35) // 100)
            & (np.abs(a_cur - a_bg) < 9)
            & (np.abs(b_cur - b_bg) < 9)
            & (grad < 45)
        )
        cleaned = mask.copy()
        cleaned[shadow] = 0
        return cleaned

    def _mild_enhance(self, crop_bgr):
        """Tăng tương phản dịu (8x8) chỉ cho góc nhìn thứ 2 khi phòng tối — giữ ánh kim loại mà không làm mất độ trong thủy tinh."""
        if crop_bgr.size == 0 or float(crop_bgr.mean()) >= 115.0:
            return crop_bgr
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_ch = self._mild_clahe.apply(l_ch)
        return cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    def _infer_two_views(self, tight_bgr, wide_bgr):
        wide_in = self._mild_enhance(wide_bgr)
        if hasattr(self.predictor, "predict_batch_details"):
            return self.predictor.predict_batch_details([tight_bgr, wide_in], bgr=True)
        # Dự phòng tương thích nếu predictor mô phỏng chỉ có .predict(PIL.Image)
        results = []
        for crop_bgr in (tight_bgr, wide_in):
            pil_img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            lbl, score, accepted = self.predictor.predict(pil_img)
            probs = {k: (score if k == lbl else (1.0 - score) / max(1, len(LABELS) - 1)) for k in LABELS}
            results.append((lbl, score, accepted, probs, 1.0))
        return results

    def step(self, roi):
        small = cv2.GaussianBlur(cv2.resize(roi, (160, 160), interpolation=cv2.INTER_AREA), (5, 5), 0)
        if self.calibrating:
            self.samples.append(small)
            if len(self.samples) < 12:
                return Reading(
                    f"Giữ bàn trống · {len(self.samples)}/12",
                    progress=len(self.samples) / 12.0,
                    state="calibrating",
                )
            self.background = np.median(self.samples, axis=0).astype(np.uint8)
            self.bg_lab = cv2.cvtColor(self.background, cv2.COLOR_BGR2LAB).astype(np.int16)
            self.samples.clear()
            self.calibrating = False
        if self.background is None:
            return self.clear("Dọn trống khung rồi bấm “Ghi nhận nền trống”", state="uncalibrated")

        # Bù lệch phơi sáng tự động (auto-exposure) toàn cục giữa nền và khung hiện tại
        delta = np.median(small.astype(np.int16) - self.background.astype(np.int16), axis=(0, 1))
        delta = np.clip(delta, -35, 35)
        aligned = np.clip(small.astype(np.int16) - delta, 0, 255).astype(np.uint8)
        diff = cv2.absdiff(aligned, self.background).max(axis=2)
        mask = (diff > 24).astype(np.uint8)
        mask = self._suppress_shadows(aligned, mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        if self.previous is None:
            moving = False
        else:
            moving = (cv2.absdiff(small, self.previous).max(axis=2) > 28).mean() > 0.065
        self.previous = small

        _, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        regions = [s for s in stats[1:] if s[cv2.CC_STAT_AREA] > 75]
        if not regions or max(s[4] for s in regions) < 0.012 * mask.size:
            return self.clear("Đưa một vật vào khung", state="idle")
        if mask.mean() > 0.62:
            return self.clear("Góc máy hoặc nền đã đổi. Hãy bấm “Ghi nhận nền trống”.", state="blocked")
        if moving:
            return self.clear("Giữ vật đứng yên trong khung", state="moving")

        x, y, width, height, _ = max(regions, key=lambda s: s[4])
        if width / 160.0 > 0.88 and height / 160.0 > 0.85:
            return self.clear("Vật quá sát hoặc nền đã đổi. Hãy lùi vật ra hoặc ghi lại nền.", state="blocked")
        roi_h, roi_w = roi.shape[:2]
        tx1 = max(0, x * roi_w // 160)
        ty1 = max(0, y * roi_h // 160)
        tx2 = min(roi_w, max(tx1 + 1, min(160, x + width) * roi_w // 160))
        ty2 = min(roi_h, max(ty1 + 1, min(160, y + height) * roi_h // 160))
        bbox = (tx1, ty1, tx2, ty2)

        # Dùng lại kết quả nếu vật đứng yên gần như tuyệt đối so với khung mốc
        scene_changed = (
            self.settled_ref is None
            or (cv2.absdiff(small, self.settled_ref).max(axis=2) > 20).mean() > 0.025
        )
        if scene_changed:
            self._reset_object_state()
            self.settled_ref = small
        else:
            if self.cached_face is True:
                return Reading("Có người trong khung. Hãy tránh ra và chỉ đặt vật vào khung.", state="hand", bbox=bbox)
            if self.cached_palm is True:
                return Reading("Có bàn tay trong khung. Hãy đặt vật xuống và rút tay.", state="hand", bbox=bbox)
            if self.locked_reading is not None:
                return self.locked_reading

        # Chặn khuôn mặt và bàn tay (chạy 1 lần khi vật vừa dừng, cache khi đứng yên)
        if self.cached_face is None:
            self.cached_face = bool(self.face(roi))
        if self.cached_face:
            self.locked_reading = None
            self.ema_probs = None
            self.stable_count = 0
            self.last_candidate = None
            return Reading("Có người trong khung. Hãy tránh ra và chỉ đặt vật vào khung.", state="hand", bbox=bbox)

        if self.cached_palm is None:
            self.cached_palm = bool(self.palm(roi)) if self._has_skin_candidate(small, mask) else False
        if self.cached_palm:
            self.locked_reading = None
            self.ema_probs = None
            self.stable_count = 0
            self.last_candidate = None
            return Reading("Có bàn tay trong khung. Hãy đặt vật xuống và rút tay.", state="hand", bbox=bbox)

        # Gộp cả 2 góc cắt (sát vật + rộng 8%) trong cùng 1 batch suy luận
        margin = max(3, int(max(width, height) * 0.08))
        wx1 = max(0, (x - margin) * roi_w // 160)
        wy1 = max(0, (y - margin) * roi_h // 160)
        wx2 = min(roi_w, max(wx1 + 1, min(160, x + width + margin) * roi_w // 160))
        wy2 = min(roi_h, max(wy1 + 1, min(160, y + height + margin) * roi_h // 160))

        tight_crop = roi[ty1:ty2, tx1:tx2]
        wide_crop = roi[wy1:wy2, wx1:wx2]
        (l1, _, a1, p1, _), (l2, _, a2, p2, _) = self._infer_two_views(tight_crop, wide_crop)

        avg_probs = {k: 0.5 * (p1[k] + p2[k]) for k in p1}
        if self.ema_probs is None:
            self.ema_probs = avg_probs
        else:
            self.ema_probs = {k: 0.65 * avg_probs[k] + 0.35 * self.ema_probs[k] for k in avg_probs}

        min_conf = float(getattr(self.predictor, "min_confidence", 0.80))
        top_label = max(self.ema_probs, key=self.ema_probs.get)
        ema_score = float(self.ema_probs[top_label])
        agreed = (l1 == l2 == top_label) and (a1 or a2) and (ema_score >= min_conf)

        if not agreed:
            self.stable_count = 0
            self.last_candidate = None
            return Reading(
                "Chưa rõ vật liệu. Hãy xoay vật hoặc thử góc khác.",
                probs=dict(self.ema_probs),
                progress=0.0,
                state="unknown",
                bbox=bbox,
            )

        if top_label == self.last_candidate:
            self.stable_count += 1
        else:
            self.last_candidate = top_label
            self.stable_count = 1

        needed = 2
        if self.stable_count < needed:
            return Reading(
                "Giữ vật đứng yên. Đang xác nhận vật liệu…",
                probs=dict(self.ema_probs),
                progress=self.stable_count / float(needed),
                state="scanning",
                bbox=bbox,
            )

        reading = Reading(
            f"{LABELS[top_label]} · {ema_score:.0%}",
            label=top_label,
            probs=dict(self.ema_probs),
            progress=1.0,
            state="locked",
            bbox=bbox,
        )
        self.locked_reading = reading
        return reading
