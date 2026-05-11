# -*- coding: utf-8 -*-

import os
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset


class AMOSSliceDataset(Dataset):
    """
    Dataset 2D multiclase para slices ya exportados como imagenes.

    Espera una carpeta de imagenes y otra de mascaras con el mismo nombre base.
    Ejemplo:
        images/case001_045.png
        masks/case001_045.png

    La mascara debe contener etiquetas enteras:
        0 = fondo
        1..N = organos
    """

    def __init__(
        self,
        image_dir,
        mask_dir,
        transform=None,
        image_suffixes=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
        mask_suffix=".png",
        window_min=None,
        window_max=None,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.mask_suffix = mask_suffix
        self.window_min = window_min
        self.window_max = window_max

        self.images = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in image_suffixes
        )

        if len(self.images) == 0:
            raise RuntimeError(f"No se encontraron imagenes en {self.image_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = self.mask_dir / f"{image_path.stem}{self.mask_suffix}"

        if not mask_path.exists():
            raise FileNotFoundError(f"No existe la mascara esperada: {mask_path}")

        image = np.array(Image.open(image_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path), dtype=np.int64)

        if self.window_min is not None and self.window_max is not None:
            image = np.clip(image, self.window_min, self.window_max)
            image = (image - self.window_min) / (self.window_max - self.window_min)
        else:
            image = image / 255.0

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        if image.ndim == 2:
            image = image.unsqueeze(0) if torch.is_tensor(image) else image[None, ...]

        if not torch.is_tensor(image):
            image = torch.from_numpy(image).float()
        else:
            image = image.float()

        if not torch.is_tensor(mask):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()

        return image, mask


class DiceCELoss(nn.Module):
    """
    CrossEntropy + Dice multiclase.

    CrossEntropy estabiliza el aprendizaje pixel a pixel.
    Dice compensa el desbalance entre fondo y organos pequenos.
    """

    def __init__(self, num_classes, ce_weight=1.0, dice_weight=1.0, ignore_index=None, smooth=1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index) if ignore_index is not None else nn.CrossEntropyLoss()

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target)
        probs = torch.softmax(logits, dim=1)

        target_one_hot = torch.zeros_like(probs)
        valid_target = target
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index
            valid_target = target.clone()
            valid_target[~valid_mask] = 0
        else:
            valid_mask = torch.ones_like(target, dtype=torch.bool)

        target_one_hot.scatter_(1, valid_target.unsqueeze(1), 1)
        valid_mask = valid_mask.unsqueeze(1)
        probs = probs * valid_mask
        target_one_hot = target_one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_one_hot, dims)
        cardinality = torch.sum(probs + target_one_hot, dims)
        dice_per_class = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)

        # Excluimos fondo de la Dice loss si hay mas de una clase.
        if self.num_classes > 1:
            dice_per_class = dice_per_class[1:]

        dice_loss = 1.0 - dice_per_class.mean()
        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


def save_checkpoint(state, filename="best_checkpoint.pth.tar"):
    print(f"=> Guardando checkpoint en {filename}")
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cuda"):
    print(f"=> Cargando checkpoint desde {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    mask_suffix=".png",
):
    train_ds = AMOSSliceDataset(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
        mask_suffix=mask_suffix,
    )

    val_ds = AMOSSliceDataset(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
        mask_suffix=mask_suffix,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


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


def check_accuracy(loader, model, num_classes, device="cuda"):
    model.eval()
    total_correct = 0
    total_pixels = 0
    dice_sums = torch.zeros(num_classes - 1, device=device)
    dice_counts = torch.zeros(num_classes - 1, device=device)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1)

            total_correct += (preds == y).sum().item()
            total_pixels += y.numel()

            dices = dice_per_class_from_logits(logits, y, num_classes)
            for idx, dice_value in enumerate(dices):
                if dice_value is not None:
                    dice_sums[idx] += dice_value
                    dice_counts[idx] += 1

    pixel_acc = total_correct / total_pixels * 100
    mean_dice_per_class = dice_sums / torch.clamp(dice_counts, min=1)
    mean_dice = mean_dice_per_class[dice_counts > 0].mean().item()

    print(f"Pixel accuracy: {pixel_acc:.2f}%")
    print(f"Mean Dice sin fondo: {mean_dice:.4f}")
    for cls_idx, dice_value in enumerate(mean_dice_per_class, start=1):
        if dice_counts[cls_idx - 1] > 0:
            print(f"  Clase {cls_idx}: Dice {dice_value.item():.4f}")

    model.train()
    return mean_dice


def validation_function(model, loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    num_samples = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device=device)
            targets = targets.to(device=device).long()
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            val_loss += loss.item() * data.size(0)
            num_samples += data.size(0)

    model.train()
    return val_loss / max(num_samples, 1)


def save_predictions_as_imgs(loader, model, folder, device, max_batches=4):
    os.makedirs(folder, exist_ok=True)
    model.eval()

    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            if idx >= max_batches:
                break

            x = x.to(device=device)
            logits = model(x)
            preds = torch.argmax(logits, dim=1).float()

            # Normalizamos para visualizacion; no usar estas imagenes como mascaras reales.
            pred_vis = preds.unsqueeze(1) / max(preds.max().item(), 1.0)
            y_vis = y.unsqueeze(1).float() / max(y.max().item(), 1.0)

            torchvision.utils.save_image(pred_vis, f"{folder}/{idx}_pred.png")
            torchvision.utils.save_image(y_vis, f"{folder}/{idx}_label.png")

    model.train()


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
