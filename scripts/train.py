"""Huấn luyện mô hình phân loại vật liệu (hỗ trợ MobileNetV3-Small, MobileNetV2, ShuffleNetV2)."""
import argparse
import copy
import random
from collections import defaultdict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

from check_data import LABELS, PROJECT, scan
from material import (
    DATA,
    DEFAULT_ARCH,
    PREPROCESS,
    RUN,
    SUPPORTED_ARCHS,
    classify_from_features,
    device_for,
    features,
    file_hash,
    head_parameters,
    network,
    open_rgb,
    transform,
    unfreeze_top_blocks,
    write_json,
)


def split_data(root, seed):
    paths, audit = scan(root)
    if audit["errors"]:
        raise ValueError("Có ảnh lỗi. Chạy check_data.py và kiểm tra audit.json trước khi train.")
    group_for = {}
    conflicting = set()
    for members in audit["duplicate_groups"]["rgb_sha256"].values():
        if len({p.split("/")[0] for p in members}) > 1:
            conflicting.update(members)
            continue
        group_for.update(dict.fromkeys(members, members[0]))
    if conflicting:
        print(f"Loại bỏ {len(conflicting)} tệp trùng điểm ảnh khác nhãn để tránh nhiễu và rò rỉ dữ liệu.", flush=True)
    rng = random.Random(seed)
    splits = {name: [] for name in ("train", "validation", "test")}
    for index, label in enumerate(LABELS):
        groups = defaultdict(list)
        for p in paths[label]:
            rel = p.relative_to(root).as_posix()
            if rel in conflicting:
                continue
            groups[group_for.get(rel, rel)].append({"path": rel, "target": index, "sha256": file_hash(p)})
        if len(groups) < 3:
            raise ValueError(f"Lớp {label} cần ít nhất ba nhóm ảnh độc lập.")
        blocks = list(groups.values())
        rng.shuffle(blocks)
        # Chia theo nhóm nguyên vẹn. Với các ảnh độc lập, hai tập nhỏ chiếm 15% mỗi tập.
        n = len(blocks)
        holdout = max(1, round(.15 * n))
        for split, partition in (("test", blocks[:holdout]), ("validation", blocks[holdout:2*holdout]),
                                 ("train", blocks[2*holdout:])):
            for group in partition:
                splits[split].extend(group)
    return splits


class Pictures(Dataset):
    def __init__(self, root, rows, prepare):
        self.root, self.rows, self.prepare = root, rows, prepare

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        return self.prepare(open_rgb(self.root / row["path"])), row["target"]


@torch.no_grad()
def extract(model, root, rows, device, arch=DEFAULT_ARCH, preprocess=PREPROCESS, augment=False):
    dataset = Pictures(root, rows, transform(preprocess, augment))
    vectors, targets = [], []
    for images, labels in DataLoader(dataset, batch_size=32):
        vectors.append(features(model, images.to(device), arch=arch).cpu())
        targets.append(labels)
    return torch.cat(vectors), torch.cat(targets)


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        total_correct += int((logits.argmax(1) == labels).sum().item())
        total += len(labels)
    return total_correct / max(1, total), total_loss / max(1, total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA)
    parser.add_argument("--arch", choices=list(SUPPORTED_ARCHS), default=DEFAULT_ARCH)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=25, help="Số epoch giai đoạn 1 (đóng băng backbone)")
    parser.add_argument("--finetune-epochs", type=int, default=0, help="Số epoch giai đoạn 2 (mở 3 khối cuối, BN đóng băng)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.epochs < 1 or args.finetune_epochs < 0:
        parser.error("--epochs phải >= 1 và --finetune-epochs phải >= 0")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = device_for(args.device)
    RUN.mkdir(exist_ok=True, parents=True)
    torch.hub.set_dir(str(PROJECT / ".cache" / "torch" / "hub"))

    preprocess = dict(PREPROCESS, size=args.img_size)
    splits = split_data(args.data_root, args.seed)
    print(f"Kiến trúc: {args.arch} ({args.img_size}x{args.img_size}) | Số ảnh:",
          {key: len(value) for key, value in splits.items()}, flush=True)

    model = network(len(LABELS), pretrained=True, arch=args.arch).to(device).eval()
    for param in model.parameters():
        param.requires_grad_(False)
    for param in head_parameters(model, args.arch):
        param.requires_grad_(True)

    # Giai đoạn 1: Huấn luyện lớp phân loại trên đặc trưng (1 lượt gốc + 3 lượt tăng cường miền webcam)
    print(f"Giai đoạn 1 — Trích đặc trưng trên {device} (train gốc, 3 lượt tăng cường, validation)…", flush=True)
    x, y = extract(model, args.data_root, splits["train"], device, arch=args.arch, preprocess=preprocess)
    x_aug1, y_aug1 = extract(model, args.data_root, splits["train"], device, arch=args.arch, preprocess=preprocess, augment=True)
    x_aug2, y_aug2 = extract(model, args.data_root, splits["train"], device, arch=args.arch, preprocess=preprocess, augment=True)
    x_aug3, y_aug3 = extract(model, args.data_root, splits["train"], device, arch=args.arch, preprocess=preprocess, augment=True)
    val_feat, val_target = extract(model, args.data_root, splits["validation"], device, arch=args.arch, preprocess=preprocess)
    val_feat, val_target = val_feat.to(device), val_target.to(device)

    batches = DataLoader(TensorDataset(torch.cat([x, x_aug1, x_aug2, x_aug3]), torch.cat([y, y_aug1, y_aug2, y_aug3])),
                         batch_size=48, shuffle=True)
    # Trọng số lớp cân bằng mềm giúp lớp ít mẫu (như trash) học đều hơn
    cls_counts = torch.bincount(y, minlength=len(LABELS)).float().clamp_min(1.0)
    cls_weights = (cls_counts.sum() / (len(LABELS) * cls_counts)).sqrt()
    cls_weights = (cls_weights / cls_weights.mean()).to(device)

    optimizer = torch.optim.AdamW(head_parameters(model, args.arch), lr=1e-3, weight_decay=1e-2)
    head_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-4)
    best_key = (-1.0, float("-inf"))
    best_state = copy.deepcopy(model.state_dict())
    best_epoch = 1
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        for vectors, labels in batches:
            optimizer.zero_grad()
            logits = classify_from_features(model, vectors.to(device), arch=args.arch)
            loss = F.cross_entropy(logits, labels.to(device), weight=cls_weights)
            loss.backward()
            optimizer.step()
        head_sched.step()
        model.eval()
        with torch.no_grad():
            logits = classify_from_features(model, val_feat, arch=args.arch)
            accuracy = (logits.argmax(1) == val_target).float().mean().item()
            val_loss = F.cross_entropy(logits, val_target).item()
        history.append(dict(stage="head", epoch=epoch, val_accuracy=accuracy, val_loss=val_loss))
        if (accuracy, -val_loss) > best_key:
            best_key = (accuracy, -val_loss)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
        print(f"[Head {epoch:02}/{args.epochs}] validation {accuracy:.2%}, loss {val_loss:.4f}", flush=True)

    # Giai đoạn 2 (tùy chọn): Fine-tune nhẹ các khối cuối, giữ nguyên BatchNorm ở eval() để không lệch miền OOD
    if args.finetune_epochs > 0:
        model.load_state_dict(best_state)
        backbone_params, head_params = unfreeze_top_blocks(model, args.arch)
        train_loader = DataLoader(
            Pictures(args.data_root, splits["train"], transform(preprocess, augment=True)),
            batch_size=32, shuffle=True,
        )
        val_loader = DataLoader(
            Pictures(args.data_root, splits["validation"], transform(preprocess, augment=False)),
            batch_size=32, shuffle=False,
        )
        ft_opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": 1e-4, "weight_decay": 1e-2},
            {"params": head_params, "lr": 5e-4, "weight_decay": 1e-2},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ft_opt, T_max=args.finetune_epochs, eta_min=1e-5)
        print(f"Giai đoạn 2 — Fine-tune {args.finetune_epochs} epoch các khối cuối trên {device}…", flush=True)
        for ft_ep in range(1, args.finetune_epochs + 1):
            model.train()
            for m in model.modules():
                if isinstance(m, torch.nn.BatchNorm2d):
                    m.eval()
            for images, labels in train_loader:
                ft_opt.zero_grad()
                logits = model(images.to(device))
                loss = F.cross_entropy(logits, labels.to(device), weight=cls_weights)
                loss.backward()
                ft_opt.step()
            scheduler.step()
            accuracy, val_loss = evaluate_loader(model, val_loader, device)
            global_epoch = args.epochs + ft_ep
            history.append(dict(stage="finetune", epoch=global_epoch, val_accuracy=accuracy, val_loss=val_loss))
            if (accuracy, -val_loss) > best_key:
                best_key = (accuracy, -val_loss)
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = global_epoch
            print(f"[FineTune {ft_ep:02}/{args.finetune_epochs}] validation {accuracy:.2%}, loss {val_loss:.4f}", flush=True)

    # Hiệu chuẩn lại tâm lớp và ngưỡng Cosine OOD trên đúng trạng thái mô hình tốt nhất
    model.load_state_dict(best_state)
    model.eval()
    x_best, y_best = extract(model, args.data_root, splits["train"], device, arch=args.arch, preprocess=preprocess)
    v_best, v_target = extract(model, args.data_root, splits["validation"], device, arch=args.arch, preprocess=preprocess)
    centers = F.normalize(
        torch.stack([F.normalize(x_best[y_best == i], dim=1).mean(0) for i in range(len(LABELS))]),
        dim=1,
    )
    with torch.no_grad():
        v_logits = classify_from_features(model, v_best.to(device), arch=args.arch).cpu()
        v_correct = v_logits.argmax(1) == v_target
        limits = []
        for i, center in enumerate(centers):
            sims = F.cosine_similarity(v_best[(v_target == i) & v_correct], center[None])
            limits.append(float(sims.quantile(0.05)) if len(sims) else 1.0)

    weights_name = SUPPORTED_ARCHS[args.arch][0]
    checkpoint = dict(
        architecture=args.arch,
        weights=weights_name,
        state={k: v.cpu() for k, v in best_state.items()},
        labels=list(LABELS),
        preprocess=preprocess,
        reference_centers=centers.cpu(),
        min_similarity=limits,
        min_confidence=0.80,
        splits=splits,
        seed=args.seed,
        epoch=best_epoch,
        val_accuracy=best_key[0],
    )
    torch.save(checkpoint, RUN / "best.tmp")
    (RUN / "best.tmp").replace(RUN / "best.pt")
    write_json(RUN / "splits.json", splits)
    write_json(RUN / "training.json", dict(
        architecture=args.arch,
        img_size=args.img_size,
        seed=args.seed,
        device=str(device),
        best_epoch=best_epoch,
        best_val_accuracy=best_key[0],
        epochs=history,
    ))
    print(f"Đã lưu checkpoint tối ưu ({args.arch}, epoch {best_epoch}, val {best_key[0]:.2%}) → {RUN / 'best.pt'}")


if __name__ == "__main__":
    main()
