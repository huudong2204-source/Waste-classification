"""Mô hình và phép biến đổi dùng chung cho train, test, ảnh lẻ và webcam."""
import hashlib
import json
import os
import warnings

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.models import (
    MobileNet_V2_Weights,
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    ShuffleNet_V2_X1_0_Weights,
    mobilenet_v2,
    mobilenet_v3_large,
    mobilenet_v3_small,
    shufflenet_v2_x1_0,
)

from check_data import LABELS, PROJECT


RUN = PROJECT / "outputs" / "model"
BEST = RUN / "best.pt"
DATA = PROJECT / "dataset-resized"
DEFAULT_ARCH = "mobilenet_v3_large"
MIN_CONFIDENCE = 0.80
MIN_COSINE_FLOOR = 0.24
SUPPORTED_ARCHS = {
    "mobilenet_v3_large": ("IMAGENET1K_V2", MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "mobilenet_v3_small": ("IMAGENET1K_V1", MobileNet_V3_Small_Weights.IMAGENET1K_V1),
    "mobilenet_v2": ("IMAGENET1K_V2", MobileNet_V2_Weights.IMAGENET1K_V2),
    "shufflenet_v2_x1_0": ("IMAGENET1K_V1", ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1),
}
PREPROCESS = {"size": 224, "mode": "RGB", "interpolation": "bilinear",
              "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}


def device_for(name="auto"):
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("PyTorch chưa nhận CUDA. Hãy chọn --device cpu.")
    return torch.device(name)


class WebcamNoiseAndBlur:
    """Mô phỏng nhiễu hạt ISO và độ nét nhẹ của webcam trong phòng thiếu sáng."""

    def __init__(self, p=0.45):
        self.p = p

    def __call__(self, tensor):
        if torch.rand(1).item() < self.p:
            sigma = 0.01 + 0.035 * torch.rand(1).item()
            tensor = (tensor + torch.randn_like(tensor) * sigma).clamp(0.0, 1.0)
        return tensor


def transform(config, augment=False):
    size = config["size"]
    interp = T.InterpolationMode(config["interpolation"])
    if augment:
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomApply([T.RandomRotation(22, interpolation=interp, fill=110)], p=0.6),
            T.RandomResizedCrop((size, size), scale=(0.65, 1.0), ratio=(0.78, 1.28), interpolation=interp),
            T.ColorJitter(brightness=(0.35, 1.25), contrast=(0.65, 1.35), saturation=(0.65, 1.35), hue=0.04),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.25),
            T.ToTensor(),
            WebcamNoiseAndBlur(p=0.45),
            T.Normalize(config["mean"], config["std"]),
        ])
    return T.Compose([
        T.Resize((size, size), interpolation=interp),
        T.ToTensor(),
        T.Normalize(config["mean"], config["std"]),
    ])


def open_rgb(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def network(count, pretrained=False, arch=DEFAULT_ARCH):
    if arch not in SUPPORTED_ARCHS:
        raise ValueError(f"Kiến trúc chưa hỗ trợ: {arch}. Chọn một trong: {', '.join(SUPPORTED_ARCHS)}")
    _, weight_enum = SUPPORTED_ARCHS[arch]
    weights = weight_enum if pretrained else None
    if arch == "mobilenet_v3_large":
        model = mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, count)
        return model
    if arch == "mobilenet_v3_small":
        model = mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, count)
        return model
    if arch == "mobilenet_v2":
        model = mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.last_channel, count)
        return model
    model = shufflenet_v2_x1_0(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, count)
    return model


def features(model, batch, arch="mobilenet_v2"):
    if arch in ("mobilenet_v3_large", "mobilenet_v3_small"):
        return model.avgpool(model.features(batch)).flatten(1)
    if arch == "shufflenet_v2_x1_0":
        x = model.maxpool(model.conv1(batch))
        x = model.conv5(model.stage4(model.stage3(model.stage2(x))))
        return x.mean([2, 3])
    return F.adaptive_avg_pool2d(model.features(batch), 1).flatten(1)


def classify_from_features(model, emb, arch="mobilenet_v2"):
    if arch == "shufflenet_v2_x1_0":
        return model.fc(emb)
    return model.classifier(emb)


def head_parameters(model, arch="mobilenet_v2"):
    if arch == "shufflenet_v2_x1_0":
        return model.fc.parameters()
    return model.classifier.parameters()


def unfreeze_top_blocks(model, arch="mobilenet_v2"):
    for param in model.parameters():
        param.requires_grad_(False)
    backbone_params = []
    if arch in ("mobilenet_v3_large", "mobilenet_v3_small", "mobilenet_v2"):
        for block in model.features[-4:]:
            for param in block.parameters():
                param.requires_grad_(True)
                backbone_params.append(param)
        head = list(model.classifier.parameters())
    else:
        for module in (model.stage4, model.conv5):
            for param in module.parameters():
                param.requires_grad_(True)
                backbone_params.append(param)
        head = list(model.fc.parameters())
    for param in head:
        param.requires_grad_(True)
    return backbone_params, head


class DualOutputWrapper(nn.Module):
    """Trả đồng thời (logits, l2_normalized_embedding) trong một lượt forward để JIT tối ưu."""

    def __init__(self, model, arch):
        super().__init__()
        self.model = model
        self.arch = arch

    def forward(self, x):
        emb = features(self.model, x, self.arch)
        logits = classify_from_features(self.model, emb, self.arch)
        return logits, F.normalize(emb, dim=1)


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checkpoint(path, device):
    if not path.is_file():
        raise FileNotFoundError(f"Thiếu {path}. Hãy chạy scripts/train.py hoặc chép best.pt vào outputs/model.")
    info = torch.load(path, map_location="cpu", weights_only=True)
    arch = info.get("architecture", "mobilenet_v2")
    if arch not in SUPPORTED_ARCHS or set(info["labels"]) != set(LABELS):
        raise ValueError(f"Checkpoint không khớp danh sách {len(LABELS)} loại vật liệu hiện tại.")
    model = network(len(info["labels"]), arch=arch)
    model.load_state_dict(info["state"])
    model.to(device).eval()
    return model, info


class Predictor:
    def __init__(self, path=BEST, device="auto"):
        self.device = device_for(device)
        self.model, self.info = load_checkpoint(path, self.device)
        self.arch = self.info.get("architecture", "mobilenet_v2")
        self.labels = list(self.info["labels"])
        self.size = int(self.info["preprocess"]["size"])
        self.mean = np.array(self.info["preprocess"]["mean"], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(self.info["preprocess"]["std"], dtype=np.float32).reshape(1, 1, 3)
        self.min_confidence = max(MIN_CONFIDENCE, float(self.info.get("min_confidence", MIN_CONFIDENCE)))
        self.min_similarity = [float(v) for v in self.info["min_similarity"]]
        refs = self.info["reference_centers"].to(self.device)
        self.refs = F.normalize(refs, dim=1)
        self.use_channels_last = self.device.type == "cpu"
        wrapper = DualOutputWrapper(self.model, self.arch).to(self.device).eval()
        if self.use_channels_last:
            wrapper = wrapper.to(memory_format=torch.channels_last)
            try:
                sample = torch.zeros(1, 3, self.size, self.size, device=self.device).to(memory_format=torch.channels_last)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    self.runner = torch.jit.freeze(torch.jit.trace(wrapper, sample))
            except Exception:
                self.runner = wrapper
        else:
            self.runner = wrapper

    def _to_rgb_array(self, image, bgr=False):
        if isinstance(image, np.ndarray):
            arr = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if bgr else image
        else:
            arr = np.asarray(image.convert(self.info["preprocess"]["mode"]))
        resized = cv2.resize(arr, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        norm = (resized.astype(np.float32) * (1.0 / 255.0) - self.mean) / self.std
        return np.transpose(norm, (2, 0, 1))

    @torch.inference_mode()
    def predict_batch_details(self, images, bgr=False):
        batch_np = np.stack([self._to_rgb_array(img, bgr=bgr) for img in images], axis=0)
        x = torch.from_numpy(batch_np).to(self.device)
        if self.use_channels_last:
            x = x.to(memory_format=torch.channels_last)
        logits, emb = self.runner(x)
        probs_batch = logits.softmax(1).cpu().numpy()
        sims_batch = (emb @ self.refs.T).cpu().numpy()
        results = []
        for probs, sims in zip(probs_batch, sims_batch):
            index = int(np.argmax(probs))
            score = float(probs[index])
            similarity = float(sims[index])
            accepted = score >= self.min_confidence and similarity >= MIN_COSINE_FLOOR
            prob_map = {lbl: float(p) for lbl, p in zip(self.labels, probs)}
            results.append((self.labels[index], score, accepted, prob_map, similarity))
        return results

    @torch.inference_mode()
    def predict_details(self, image, bgr=False):
        return self.predict_batch_details([image], bgr=bgr)[0]

    @torch.inference_mode()
    def predict(self, image):
        label, score, accepted, _, _ = self.predict_details(image, bgr=False)
        return label, score, accepted
