# -*- coding: utf-8 -*-

from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from tqdm import tqdm

from Unet3D import UNET3D
from utils_3d import (
    DiceCELoss3D,
    check_accuracy,
    compute_class_weights,
    get_loaders,
    save_history_row,
    update_top_checkpoints,
    validation_function,
)


LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 1
NUM_EPOCHS = 80
NUM_WORKERS = 0
PIN_MEMORY = True
NUM_CLASSES = 16
TOP_K_CHECKPOINTS = 2

TRAIN_PATCH_DIR = "D:/Clase/TFG/versión abril/amos_patches_3d/train"
VAL_PATCH_DIR = "D:/Clase/TFG/versión abril/amos_patches_3d/val"
OUTPUT_ROOT = Path("D:/Clase/TFG/versión abril/amos_3d/runs")


def train_fn(loader, model, optimizer, loss_fn, scaler):
    model.train()
    loop = tqdm(loader, desc="Entrenamiento 3D")
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
    history_path = run_dir / "training_history_3d.csv"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Run 3D actual: {run_dir}")
    print(f"Dispositivo: {DEVICE}")

    model = UNET3D(
        in_channels=1,
        out_channels=NUM_CLASSES,
        features=(16, 32, 64, 128),
        dropout=0.15,
    ).to(DEVICE)

    class_weights = compute_class_weights(TRAIN_PATCH_DIR, NUM_CLASSES).to(DEVICE)
    print("Pesos de clases:", class_weights.detach().cpu().numpy())
    loss_fn = DiceCELoss3D(
        num_classes=NUM_CLASSES,
        ce_weight=0.7,
        dice_weight=1.3,
        class_weights=class_weights,
    )
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    train_loader, val_loader = get_loaders(
        TRAIN_PATCH_DIR,
        VAL_PATCH_DIR,
        BATCH_SIZE,
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
        print(f"\nEpoch 3D [{epoch + 1}/{NUM_EPOCHS}]")
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        val_loss = validation_function(model, val_loader, loss_fn, DEVICE)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        mean_dice, voxel_acc, dice_by_class = check_accuracy(
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
            "patch_based_training": True,
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
            prefix="best_by_val_loss_3d",
            checkpoint_dir=checkpoint_dir,
            top_k=TOP_K_CHECKPOINTS,
        )
        top_mean_dice_checkpoints, saved_top_mean_dice = update_top_checkpoints(
            top_mean_dice_checkpoints,
            checkpoint,
            metric_name="mean_dice",
            metric_value=mean_dice,
            mode="max",
            prefix="best_by_mean_dice_3d",
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
        else:
            epochs_without_dice_improvement += 1

        print(
            "Sin mejora "
            f"val_loss: {epochs_without_val_loss_improvement}/{val_loss_patience} | "
            f"mean_dice: {epochs_without_dice_improvement}/{dice_patience}"
        )

        current_lr = optimizer.param_groups[0]["lr"]
        history_row = {
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "voxel_accuracy": round(voxel_acc, 6),
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
            print("Early stopping 3D activado.")
            break


if __name__ == "__main__":
    main()
