# -*- coding: utf-8 -*-

import torch
import torch.optim as optim
from tqdm import tqdm
import csv
from pathlib import Path
from datetime import datetime

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
OUTPUT_ROOT = Path("D:/Clase/TFG/versión abril/scripts/runs")
MASK_SUFFIX = ".png"
TOP_K_CHECKPOINTS = 2


def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader)
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE).long()

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=DEVICE == "cuda"):
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
        fieldnames.extend(f"dice_class_{cls_idx}" for cls_idx in range(1, NUM_CLASSES))
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=";")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def update_top_checkpoints(records, checkpoint, metric_name, metric_value, mode, prefix):
    checkpoint_dir = CHECKPOINT_DIR
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
    kept = records[:TOP_K_CHECKPOINTS]
    dropped = records[TOP_K_CHECKPOINTS:]

    is_candidate_kept = any(item["path"] == candidate["path"] for item in kept)
    if is_candidate_kept:
        save_checkpoint(checkpoint, str(candidate["path"]))

    for item in dropped:
        if item["path"].exists():
            item["path"].unlink()

    return kept, is_candidate_kept


def main():
    global CHECKPOINT_DIR

    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / run_name
    CHECKPOINT_DIR = run_dir / "checkpoints"
    predictions_dir = run_dir / "saved_images_multiclass"
    history_path = run_dir / "training_history_top2.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run actual: {run_dir}")

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

    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

    best_val_loss = float("inf")
    best_mean_dice = 0.0
    top_val_loss_checkpoints = []
    top_mean_dice_checkpoints = []
    epochs_without_val_loss_improvement = 0
    epochs_without_dice_improvement = 0
    patience = 5
    min_delta = 0.001
    mean_dice_min_before_val_loss_stop = 0.55

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{NUM_EPOCHS}]")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_loss = validation_function(model, val_loader, loss_fn, DEVICE)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        mean_dice, pixel_acc, dice_by_class = check_accuracy(
            val_loader,
            model,
            num_classes=NUM_CLASSES,
            device=DEVICE,
        )

        previous_best_val_loss = best_val_loss
        previous_best_mean_dice = best_mean_dice

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "mean_dice": mean_dice,
            "num_classes": NUM_CLASSES,
        }

        if val_loss < best_val_loss:
            best_val_loss = val_loss

        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice

        top_val_loss_checkpoints, saved_top_val_loss = update_top_checkpoints(
            top_val_loss_checkpoints,
            checkpoint,
            metric_name="val_loss",
            metric_value=val_loss,
            mode="min",
            prefix="best_by_val_loss",
        )
        top_mean_dice_checkpoints, saved_top_mean_dice = update_top_checkpoints(
            top_mean_dice_checkpoints,
            checkpoint,
            metric_name="mean_dice",
            metric_value=mean_dice,
            mode="max",
            prefix="best_by_mean_dice",
        )

        val_loss_improvement = previous_best_val_loss - val_loss
        dice_improvement = mean_dice - previous_best_mean_dice

        if val_loss_improvement > min_delta:
            epochs_without_val_loss_improvement = 0
        else:
            epochs_without_val_loss_improvement += 1

        if dice_improvement > min_delta:
            epochs_without_dice_improvement = 0
            save_predictions_as_imgs(val_loader, model, folder=predictions_dir, device=DEVICE)
        else:
            epochs_without_dice_improvement += 1

        print(
            "Sin mejora "
            f"val_loss: {epochs_without_val_loss_improvement}/{patience + 5} | "
            f"mean_dice: {epochs_without_dice_improvement}/{patience}"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history_row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "pixel_accuracy": round(pixel_acc, 6),
            "mean_dice": round(mean_dice, 6),
            "learning_rate": current_lr,
            "best_val_loss": round(best_val_loss, 6),
            "best_mean_dice": round(best_mean_dice, 6),
            "val_loss_improvement": round(val_loss_improvement, 6),
            "dice_improvement": round(dice_improvement, 6),
            "epochs_without_val_loss_improvement": epochs_without_val_loss_improvement,
            "epochs_without_dice_improvement": epochs_without_dice_improvement,
            "saved_top_val_loss": int(saved_top_val_loss),
            "saved_top_mean_dice": int(saved_top_mean_dice),
        }
        history_row.update(
            {
                key: (round(value, 6) if value is not None else "")
                for key, value in dice_by_class.items()
            }
        )
        save_history_row(
            history_path,
            history_row,
        )
        print(f"Historial actualizado: {history_path}")

        stop_by_val_loss = epochs_without_val_loss_improvement >= (patience + 5)
        stop_by_mean_dice = epochs_without_dice_improvement >= patience

        if stop_by_val_loss and mean_dice < mean_dice_min_before_val_loss_stop:
            print(
                "Val_loss llego a patience, pero mean_dice todavia es menor que "
                f"{mean_dice_min_before_val_loss_stop:.2f}. Reinicio contador de val_loss."
            )
            epochs_without_val_loss_improvement = 0
        elif stop_by_val_loss or stop_by_mean_dice:
            print("Early stopping activado.")
            break


if __name__ == "__main__":
    main()
