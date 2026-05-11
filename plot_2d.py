# -*- coding: utf-8 -*-

"""
Graficas para entrenamiento 2D.

Uso recomendado:
    python plot_training_history_2d.py "D:/Clase/TFG/versión abril/scripts/runs/run_xxx/training_history_top2.csv"

Si no pasas ruta, busca automaticamente el run mas reciente en:
    D:/Clase/TFG/versión abril/scripts/runs
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RUNS_ROOT = Path("D:/Clase/TFG/versión abril/scripts/runs")
HISTORY_NAME = "training_history_top2.csv"


def find_latest_history():
    histories = sorted(RUNS_ROOT.glob(f"run_*/{HISTORY_NAME}"))
    if not histories:
        raise FileNotFoundError(f"No encuentro {HISTORY_NAME} dentro de {RUNS_ROOT}")
    return histories[-1]


def read_history(path):
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    df = pd.read_csv(path, sep=";")
    if len(df.columns) == 1:
        df = pd.read_csv(path)
    return df


def mark_best(ax, df, y_col, mode, label):
    values = pd.to_numeric(df[y_col], errors="coerce")
    best_idx = values.idxmin() if mode == "min" else values.idxmax()
    best_x = df.loc[best_idx, "epoch"]
    best_y = values.loc[best_idx]
    ax.scatter(best_x, best_y, s=95, edgecolor="black", linewidth=1.2, zorder=6)
    ax.annotate(
        f"Mejor {label}\nEpoch {int(best_x)}: {best_y:.4f}",
        xy=(best_x, best_y),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9),
        arrowprops=dict(arrowstyle="->", color="black"),
    )


def metric_plot(df, output_dir, y_col, title, ylabel, mode):
    if y_col not in df.columns:
        print(f"No existe la columna {y_col}; salto esta grafica.")
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df["epoch"], pd.to_numeric(df[y_col], errors="coerce"), marker="o", linewidth=2)
    mark_best(ax, df, y_col, mode, y_col)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = output_dir / f"{y_col}.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    print(f"Guardada: {out}")


def dice_plot(df, output_dir):
    dice_cols = sorted(
        [col for col in df.columns if col.startswith("dice_class_")],
        key=lambda col: int(col.replace("dice_class_", "")),
    )

    fig, ax = plt.subplots(figsize=(12, 7))
    for col in dice_cols:
        ax.plot(
            df["epoch"],
            pd.to_numeric(df[col], errors="coerce"),
            linewidth=1.2,
            alpha=0.55,
            label=col.replace("dice_class_", "Clase "),
        )

    ax.plot(
        df["epoch"],
        pd.to_numeric(df["mean_dice"], errors="coerce"),
        color="black",
        linewidth=3.3,
        marker="o",
        markersize=5,
        label="Mean Dice",
        zorder=5,
    )
    mark_best(ax, df, "mean_dice", "max", "mean_dice")
    ax.set_title("Dice por clase y Mean Dice - U-Net 2D")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    out = output_dir / "dice_by_class_2d.png"
    fig.savefig(out, dpi=240)
    plt.close(fig)
    print(f"Guardada: {out}")


def main():
    history_path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_history()
    print(f"Leyendo historial 2D: {history_path}")
    df = read_history(history_path)
    output_dir = history_path.parent / "plots_2d"
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_plot(df, output_dir, "train_loss", "Train Loss - U-Net 2D", "Train Loss", "min")
    metric_plot(df, output_dir, "val_loss", "Validation Loss - U-Net 2D", "Validation Loss", "min")
    metric_plot(df, output_dir, "pixel_accuracy", "Pixel Accuracy - U-Net 2D", "Pixel Accuracy (%)", "max")
    dice_plot(df, output_dir)
    print(f"\nGraficas 2D generadas en: {output_dir}")


if __name__ == "__main__":
    main()
