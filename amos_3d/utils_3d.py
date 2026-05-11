# -*- coding: utf-8 -*-

import csv
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class AMOSPatch3DDataset(Dataset):
    def __init__(self, patch_dir, augment=False):
        self.patch_dir = Path(patch_dir)
        self.augment = augment
        self.patch_paths = sorted(self.patch_dir.glob("*.npz"))
        if not self.patch_paths:
            raise RuntimeError(f"No se encontraron parches .npz en {self.patch_dir}")

    def __len__(self):
        return len(self.patch_paths)

    def __getitem__(self, index):
        data = np.load(self.patch_paths[index])
        image = data["image"].astype(np.float32)
        label = data["label"].astype(np.int64)

        if self.augment:
            image, label = random_flip_3d(image, label)
            image = random_intensity_3d(image)

        image = torch.from_numpy(image).unsqueeze(0).float()
        label = torch.from_numpy(label).long()
        return image, label


def random_flip_3d(image, label):
    for axis in range(3):
        if np.random.rand() < 0.5:
            image = np.flip(image, axis=axis).copy()
            label = np.flip(label, axis=axis).copy()
    return image, label


def random_intensity_3d(image):
    if np.random.rand() < 0.5:
        image = image * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.05, 0.05)
        image = np.clip(image, 0.0, 1.0)
    if np.random.rand() < 0.25:
        noise = np.random.normal(0.0, 0.015, size=image.shape).astype(np.float32)
        image = np.clip(image + noise, 0.0, 1.0)
    return image


class DiceCELoss3D(nn.Module):
    def __init__(self, num_classes, ce_weight=0.7, dice_weight=1.3, class_weights=None, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        probs = torch.softmax(logits, dim=1)
        target_one_hot = torch.zeros_like(probs)
        target_one_hot.scatter_(1, target.unsqueeze(1), 1)

        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * target_one_hot, dims)
        cardinality = torch.sum(probs + target_one_hot, dims)
        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        if self.num_classes > 1:
            dice = dice[1:]

        dice_loss = 1.0 - dice.mean()
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def get_loaders(train_dir, val_dir, batch_size, num_workers=0, pin_memory=True):
    train_ds = AMOSPatch3DDataset(train_dir, augment=True)
    val_ds = AMOSPatch3DDataset(val_dir, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def compute_class_weights(patch_dir, num_classes, max_weight=5.0):
    counts = np.zeros(num_classes, dtype=np.float64)
    patch_paths = sorted(Path(patch_dir).glob("*.npz"))

    for patch_path in tqdm(patch_paths, desc="Calculando pesos de clases 3D"):
        label = np.load(patch_path)["label"].astype(np.int64)
        counts += np.bincount(label.reshape(-1), minlength=num_classes)

    counts = np.maximum(counts, 1.0)
    frequencies = counts / counts.sum()
    weights = 1.0 / np.sqrt(frequencies)
    weights = weights / weights.mean()
    weights[0] = min(weights[0], 0.5)
    weights = np.clip(weights, 0.1, max_weight)
    return torch.tensor(weights, dtype=torch.float32)


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


def validation_function(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for data, targets in tqdm(loader, desc="Validacion 3D"):
            data = data.to(device=device)
            targets = targets.to(device=device).long()
            with torch.amp.autocast(device_type="cuda", enabled=device == "cuda"):
                predictions = model(data)
                loss = loss_fn(predictions, targets)
            val_loss += loss.item() * data.size(0)
            num_samples += data.size(0)

    model.train()
    return val_loss / max(num_samples, 1)


def check_accuracy(loader, model, num_classes, device):
    model.eval()
    total_correct = 0
    total_voxels = 0
    dice_sums = torch.zeros(num_classes - 1, device=device)
    dice_counts = torch.zeros(num_classes - 1, device=device)

    with torch.no_grad():
        for x, y in tqdm(loader, desc="Calculando Dice/accuracy 3D"):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == y).sum().item()
            total_voxels += y.numel()

            dices = dice_per_class_from_logits(logits, y, num_classes)
            for idx, dice_value in enumerate(dices):
                if dice_value is not None:
                    dice_sums[idx] += dice_value
                    dice_counts[idx] += 1

    voxel_acc = total_correct / total_voxels * 100
    mean_dice_per_class = dice_sums / torch.clamp(dice_counts, min=1)
    mean_dice = mean_dice_per_class[dice_counts > 0].mean().item()

    print(f"Voxel accuracy: {voxel_acc:.2f}%")
    print(f"Mean Dice 3D sin fondo: {mean_dice:.4f}")
    for cls_idx, dice_value in enumerate(mean_dice_per_class, start=1):
        if dice_counts[cls_idx - 1] > 0:
            print(f"  Clase {cls_idx}: Dice {dice_value.item():.4f}")

    dice_by_class = {
        f"dice_class_{cls_idx}": (
            mean_dice_per_class[cls_idx - 1].item()
            if dice_counts[cls_idx - 1] > 0
            else None
        )
        for cls_idx in range(1, num_classes)
    }

    model.train()
    return mean_dice, voxel_acc, dice_by_class


def save_history_row(history_path, row, num_classes):
    history_path = Path(history_path)
    file_exists = history_path.exists()
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with history_path.open("a", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "voxel_accuracy",
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
    candidate = {
        "metric": float(metric_value),
        "epoch": epoch,
        "path": checkpoint_dir / filename,
    }

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
