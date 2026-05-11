# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from tqdm import tqdm
import csv
from pathlib import Path

from Unet import UNET
from utils import (
    DiceCELoss,
    check_accuracy,
    get_loaders,
    get_train_transforms,
    get_val_transforms,
    save_checkpoint,
    save_predictions_as_imgs,
    validation_function,
)


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 80
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True

# Ajusta este numero a tus etiquetas reales.
# Si tienes fondo + 15 organos, usa 16.
NUM_CLASSES = 16

# Este train.py espera slices 2D PNG ya convertidos desde los .nii.gz.
# Primero ejecuta convert_amos_nii_to_slices.py.
TRAIN_IMG_DIR = "D:/Clase/TFG/versión abril/amos_slices_2d/imagesTr/"
TRAIN_MASK_DIR = "D:/Clase/TFG/versión abril/amos_slices_2d/labelsTr/"
VAL_IMG_DIR = "D:/Clase/TFG/versión abril/amos_slices_2d/imagesVa/"
VAL_MASK_DIR = "D:/Clase/TFG/versión abril/amos_slices_2d/labelsVa/"

CHECKPOINT_PATH = "best_amos2d_unet.pth.tar"
PREDICTIONS_DIR = "C:/Users/maria/Documents/TFG/saved_images_multiclass"
MASK_SUFFIX = ".png"
HISTORY_PATH = "training_history.csv"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader)
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=DEVICE == "cuda"):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        loop.set_postfix(loss=running_loss / (batch_idx + 1))

    return running_loss / max(len(loader), 1)


def save_history_row(history_path, row):
    history_path = Path(history_path)
    file_exists = history_path.exists()

    with history_path.open("a", newline="", encoding="utf-8-sig") as csvfile:
        fieldnames = [
            "epoch",
            "train_loss",
            "val_loss",
            "mean_dice",
            "learning_rate",
            "best_val_loss",
            "improvement",
            "epochs_without_improvement",
            "saved_checkpoint",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    train_transform = get_train_transforms(IMAGE_HEIGHT, IMAGE_WIDTH)
    val_transform = get_val_transforms(IMAGE_HEIGHT, IMAGE_WIDTH)

    model = UNET(
        in_channels=1,
        out_channels=NUM_CLASSES,
        features=(32, 64, 128, 256),
        dropout=0.1,
    ).to(DEVICE)

    loss_fn = DiceCELoss(num_classes=NUM_CLASSES, ce_weight=1.0, dice_weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transform,
        NUM_WORKERS,
        PIN_MEMORY,
        mask_suffix=MASK_SUFFIX,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE == "cuda")

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    patience = 8
    min_delta = 0.001

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_loss = validation_function(model, val_loader, loss_fn, DEVICE)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        mean_dice = check_accuracy(val_loader, model, num_classes=NUM_CLASSES, device=DEVICE)

        previous_best_val_loss = best_val_loss
        saved_checkpoint = False

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            saved_checkpoint = True
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "mean_dice": mean_dice,
                "num_classes": NUM_CLASSES,
            }
            save_checkpoint(checkpoint, CHECKPOINT_PATH)
            save_predictions_as_imgs(val_loader, model, folder=PREDICTIONS_DIR, device=DEVICE)
        else:
            epochs_without_improvement += 1
            print(f"Sin mejora: {epochs_without_improvement}/{patience}")

        if previous_best_val_loss == float("inf"):
            improvement = 0.0
        else:
            improvement = previous_best_val_loss - val_loss

        current_lr = optimizer.param_groups[0]["lr"]
        save_history_row(
            HISTORY_PATH,
            {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 6),
                "val_loss": round(val_loss, 6),
                "mean_dice": round(mean_dice, 6),
                "learning_rate": current_lr,
                "best_val_loss": round(best_val_loss, 6),
                "improvement": round(improvement, 6),
                "epochs_without_improvement": epochs_without_improvement,
                "saved_checkpoint": int(saved_checkpoint),
            },
        )
        print(f"Historial actualizado: {HISTORY_PATH}")

        if epochs_without_improvement >= patience:
            print("Early stopping activado.")
            break


if __name__ == "__main__":
    main()
