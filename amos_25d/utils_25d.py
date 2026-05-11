# -*- coding: utf-8 -*-

import csv
import os
import re
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


SLICE_RE = re.compile(r"(?P<case>.+)_z(?P<z>\d+)$")


class AMOS25DSliceDataset(Dataset):
    """
    Dataset 2.5D sobre slices PNG.

    Para cada mascara central z, carga tres imagenes:
        z-1, z, z+1

    Si falta un vecino, repite el slice central.
    """

    def __init__(self, image_dir, mask_dir, transform=None, mask_suffix=".png"):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.mask_suffix = mask_suffix
        self.images = sorted(self.image_dir.glob("*.png"))
        if not self.images:
            raise RuntimeError(f"No se encontraron imagenes PNG en {self.image_dir}")

        self.image_by_stem = {path.stem: path for path in self.images}
        self.samples = []
        for image_path in self.images:
            mask_path = self.mask_dir / f"{image_path.stem}{self.mask_suffix}"
            if mask_path.exists():
                self.samples.append(image_path)

        if not self.samples:
            raise RuntimeError(f"No hay pares imagen/mascara en {self.image_dir} y {self.mask_dir}")

    def __len__(self):
        return len(self.samples)

    def neighbor_stem(self, stem, offset):
        match = SLICE_RE.match(stem)
        if match is None:
            return stem
        case = match.group("case")
        z = int(match.group("z"))
        width = len(match.group("z"))
        return f"{case}_z{z + offset:0{width}d}"

    def load_image_2d(self, stem):
        path = self.image_by_stem.get(stem)
        if path is None:
            path = self.image_by_stem[self.current_center_stem]
        image = np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0
        return image

    def __getitem__(self, index):
        center_path = self.samples[index]
        center_stem = center_path.stem
        self.current_center_stem = center_stem

        prev_stem = self.neighbor_stem(center_stem, -1)
        next_stem = self.neighbor_stem(center_stem, 1)

        image_stack = np.stack(
            [
                self.load_image_2d(prev_stem),
                self.load_image_2d(center_stem),
                self.load_image_2d(next_stem),
            ],
            axis=-1,
        )
        mask = np.array(Image.open(self.mask_dir / f"{center_stem}{self.mask_suffix}"), dtype=np.int64)

        if self.transform is not None:
            augmented = self.transform(image=image_stack, mask=mask)
            image_stack = augmented["image"]
            mask = augmented["mask"]

        if torch.is_tensor(image_stack):
            image_stack = image_stack.permute(2, 0, 1).float() if image_stack.shape[-1] == 3 else image_stack.float()
        else:
            image_stack = torch.from_numpy(np.transpose(image_stack, (2, 0, 1))).float()

        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return image_stack, mask


class DiceCELoss(nn.Module):
    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        probs = torch.softmax(logits, dim=1)
        target_one_hot = torch.zeros_like(probs)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dims)
        cardinality = torch.sum(probs + target_one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        if self.num_classes > 1:
            dice = dice[1:]
        dice_loss = 1.0 - dice.mean()
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def get_train_transforms(image_height, image_width):
    return A.Compose(
        [
            A.Resize(height=image_height, width=image_width, interpolation=1, mask_interpolation=0),
            A.Rotate(limit=15, p=0.5, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ]
    )


def get_val_transforms(image_height, image_width):
    return A.Compose(
        [
            A.Resize(height=image_height, width=image_width, interpolation=1, mask_interpolation=0),
        ]
    )


def get_loaders(train_dir, train_maskdir, val_dir, val_maskdir, batch_size, train_transform, val_transform, num_workers=2, pin_memory=True):
    train_ds = AMOS25DSliceDataset(train_dir, train_maskdir, transform=train_transform)
    val_ds = AMOS25DSliceDataset(val_dir, val_maskdir, transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, val_loader


def save_checkpoint(state, filename):
    print(f"=> Guardando checkpoint en {filename}")
    torch.save(state, filename)


def dice_per_class_from_logits(logits, target, num_classes, eps=1e-6):
    preds = torch.argmax(logits, dim=1)
    dices = []
    for cls in range(1, num_classes):
        pred_cls = preds == cls
        target_cls = target == cls
        intersection = torch.logical_and(pred_cls, target_cls).sum().float()
        denominator = pred_cls.sum().float() + target_cls.sum().float()
        if denominator == 0:
            dices.append(None)
        else:
            dices.append((2.0 * intersection + eps) / (denominator + eps))
    return dices


def validation_and_metrics(model, loader, loss_fn, num_classes, device):
    model.eval()
    val_loss = 0.0
    num_samples = 0
    total_correct = 0
    total_pixels = 0
    dice_sums = torch.zeros(num_classes - 1, device=device)
    dice_counts = torch.zeros(num_classes - 1, device=device)

    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Validacion + metricas 2.5D"):
            data = data.to(device=device)
            targets = targets.to(device=device).long()
            with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                logits = model(data)
                loss = loss_fn(logits, targets)

            val_loss += loss.item() * data.size(0)
            num_samples += data.size(0)

            preds = torch.argmax(logits, dim=1)
            total_correct += (preds == targets).sum().item()
            total_pixels += targets.numel()

            dices = dice_per_class_from_logits(logits, targets, num_classes)
            for idx, dice_value in enumerate(dices):
                if dice_value is not None:
                    dice_sums[idx] += dice_value
                    dice_counts[idx] += 1

    pixel_acc = total_correct / total_pixels * 100
    mean_dice_per_class = dice_sums / torch.clamp(dice_counts, min=1)
    mean_dice = mean_dice_per_class[dice_counts > 0].mean().item()
    dice_by_class = {
        f"dice_class_{cls_idx}": (
            mean_dice_per_class[cls_idx - 1].item()
            if dice_counts[cls_idx - 1] > 0
            else None
        )
        for cls_idx in range(1, num_classes)
    }

    print(f"Pixel accuracy: {pixel_acc:.2f}%")
    print(f"Mean Dice 2.5D sin fondo: {mean_dice:.4f}")
    for cls_idx, value in enumerate(mean_dice_per_class, start=1):
        if dice_counts[cls_idx - 1] > 0:
            print(f"  Clase {cls_idx}: Dice {value.item():.4f}")

    model.train()
    return val_loss / max(num_samples, 1), pixel_acc, mean_dice, dice_by_class


def save_predictions_as_imgs(loader, model, folder, device, max_batches=4):
    os.makedirs(folder, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for idx, (x, y) in enumerate(tqdm(loader, desc="Guardando predicciones 2.5D")):
            if idx >= max_batches:
                break
            x = x.to(device=device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).float()
            pred_vis = preds.unsqueeze(1) / max(preds.max().item(), 1.0)
            y_vis = y.unsqueeze(1).float() / max(y.max().item(), 1.0)
            torchvision.utils.save_image(pred_vis, f"{folder}/{idx}_pred.png")
            torchvision.utils.save_image(y_vis, f"{folder}/{idx}_label.png")

    model.train()


def save_history_row(history_path, row, num_classes):
    history_path = Path(history_path)
    file_exists = history_path.exists()
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with history_path.open("a", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "pixel_accuracy",
            "mean_dice",
            "learning_rate",
            "best_val_loss",
            "best_mean_dice",
            "val_loss_improvement",
            "dice_improvement",
            "epochs_without_val_loss_improvement",
            "epochs_without_dice_improvement",
            "saved_top_val_loss",
            "saved_top_mean_dice",
        ]
        fieldnames.extend(f"dice_class_{cls_idx}" for cls_idx in range(1, num_classes))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def update_top_checkpoints(records, checkpoint, metric_name, metric_value, mode, prefix, checkpoint_dir, top_k=2):
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    epoch = checkpoint["epoch"]
    filename = f"{prefix}_epoch{epoch:03d}_{metric_name}_{metric_value:.4f}.pth.tar"
    candidate = {"metric": float(metric_value), "epoch": epoch, "path": checkpoint_dir / filename}
    records.append(candidate)
    reverse = mode == "max"
    records = sorted(records, key=lambda item: item["metric"], reverse=reverse)
    kept = records[:top_k]
    dropped = records[top_k:]

    is_candidate_kept = any(item["path"] == candidate["path"] for item in kept)
    if is_candidate_kept:
        save_checkpoint(checkpoint, str(candidate["path"]))

    for item in dropped:
        if item["path"].exists():
            os.remove(item["path"])

    return kept, is_candidate_kept
