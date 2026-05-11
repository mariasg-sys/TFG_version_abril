# -*- coding: utf-8 -*-

from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from Unet import UNET
from utils_25d import (
    DiceCELoss,
    get_loaders,
    get_train_transforms,
    get_val_transforms,
    save_history_row,
    save_predictions_as_imgs,
    update_top_checkpoints,
    validation_and_metrics,
)


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
NUM_EPOCHS = 80
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
NUM_CLASSES = 16
TOP_K_CHECKPOINTS = 2

TRAIN_IMG_DIR = "D:/Clase/TFG/versión abril/amos_slices_2d/imagesTr/"
TRAIN_MASK_DIR = "D:/Clase/TFG/versión abril/amos_slices_2d/labelsTr/"
VAL_IMG_DIR = "D:/Clase/TFG/versión abril/amos_slices_2d/imagesVa/"
VAL_MASK_DIR = "D:/Clase/TFG/versión abril/amos_slices_2d/labelsVa/"
OUTPUT_ROOT = Path("D:/Clase/TFG/versión abril/amos_25d/runs")


def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, desc="Entrenamiento 2.5D")
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


def main():
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / run_name
    checkpoint_dir = run_dir / "checkpoints"
    predictions_dir = run_dir / "saved_images_multiclass"
    history_path = run_dir / "training_history_25d.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run 2.5D actual: {run_dir}")
    print(f"Dispositivo: {DEVICE}")

    train_transform = get_train_transforms(IMAGE_HEIGHT, IMAGE_WIDTH)
    val_transform = get_val_transforms(IMAGE_HEIGHT, IMAGE_WIDTH)

    model = UNET(in_channels=3, out_channels=NUM_CLASSES, features=(32, 64, 128, 256), dropout=0.1).to(DEVICE)
    loss_fn = DiceCELoss(num_classes=NUM_CLASSES, ce_weight=1.0, dice_weight=1.0)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

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
    )

    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

    best_val_loss = float("inf")
    best_mean_dice = 0.0
    top_val_loss_checkpoints = []
    top_mean_dice_checkpoints = []
    epochs_without_val_loss_improvement = 0
    epochs_without_dice_improvement = 0
    dice_patience = 8
    val_loss_patience = 10
    min_delta = 0.001
    mean_dice_min_before_val_loss_stop = 0.55

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch 2.5D [{epoch + 1}/{NUM_EPOCHS}]")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_loss, pixel_acc, mean_dice, dice_by_class = validation_and_metrics(
            model,
            val_loader,
            loss_fn,
            NUM_CLASSES,
            DEVICE,
        )
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        previous_best_val_loss = best_val_loss
        previous_best_mean_dice = best_mean_dice

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "mean_dice": mean_dice,
            "num_classes": NUM_CLASSES,
            "mode": "2.5D",
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
            prefix="best_by_val_loss_25d",
            checkpoint_dir=checkpoint_dir,
            top_k=TOP_K_CHECKPOINTS,
        )
        top_mean_dice_checkpoints, saved_top_mean_dice = update_top_checkpoints(
            top_mean_dice_checkpoints,
            checkpoint,
            metric_name="mean_dice",
            metric_value=mean_dice,
            mode="max",
            prefix="best_by_mean_dice_25d",
            checkpoint_dir=checkpoint_dir,
            top_k=TOP_K_CHECKPOINTS,
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
            f"val_loss: {epochs_without_val_loss_improvement}/{val_loss_patience} | "
            f"mean_dice: {epochs_without_dice_improvement}/{dice_patience}"
        )

        history_row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "pixel_accuracy": round(pixel_acc, 6),
            "mean_dice": round(mean_dice, 6),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "best_val_loss": round(best_val_loss, 6),
            "best_mean_dice": round(best_mean_dice, 6),
            "val_loss_improvement": round(val_loss_improvement, 6),
            "dice_improvement": round(dice_improvement, 6),
            "epochs_without_val_loss_improvement": epochs_without_val_loss_improvement,
            "epochs_without_dice_improvement": epochs_without_dice_improvement,
            "saved_top_val_loss": int(saved_top_val_loss),
            "saved_top_mean_dice": int(saved_top_mean_dice),
        }
        history_row.update({key: (round(value, 6) if value is not None else "") for key, value in dice_by_class.items()})
        save_history_row(history_path, history_row, NUM_CLASSES)
        print(f"Historial actualizado: {history_path}")

        stop_by_val_loss = epochs_without_val_loss_improvement >= val_loss_patience
        stop_by_mean_dice = epochs_without_dice_improvement >= dice_patience

        if stop_by_val_loss and mean_dice < mean_dice_min_before_val_loss_stop:
            print(
                "Val_loss llego a patience, pero mean_dice todavia es menor que "
                f"{mean_dice_min_before_val_loss_stop:.2f}. Reinicio contador de val_loss."
            )
            epochs_without_val_loss_improvement = 0
        elif stop_by_val_loss or stop_by_mean_dice:
            print("Early stopping 2.5D activado.")
            break


if __name__ == "__main__":
    main()
