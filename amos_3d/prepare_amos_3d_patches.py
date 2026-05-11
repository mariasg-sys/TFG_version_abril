# -*- coding: utf-8 -*-

"""
Convierte AMOS (.nii.gz) a parches 3D .npz para entrenar una U-Net 3D.

Entrada esperada:
    D:/Clase/TFG/versión abril/amos/amos22/imagesTr
    D:/Clase/TFG/versión abril/amos/amos22/labelsTr
    D:/Clase/TFG/versión abril/amos/amos22/imagesVa
    D:/Clase/TFG/versión abril/amos/amos22/labelsVa

Salida:
    D:/Clase/TFG/versión abril/amos_patches_3d/train
    D:/Clase/TFG/versión abril/amos_patches_3d/val

Requiere:
    pip install nibabel numpy
"""

from pathlib import Path

import nibabel as nib
import numpy as np


AMOS_ROOT = Path("D:/Clase/TFG/versión abril/amos/amos22")
OUTPUT_ROOT = Path("D:/Clase/TFG/versión abril/amos_patches_3d")

WINDOW_MIN = -200
WINDOW_MAX = 300
PATCH_SIZE = (64, 128, 128)  # D, H, W
PATCHES_PER_TRAIN_VOLUME = 24
PATCHES_PER_VAL_VOLUME = 6
FOREGROUND_PROBABILITY = 0.90
RANDOM_SEED = 42
NUM_CLASSES = 16


def normalize_ct(volume):
    volume = np.clip(volume, WINDOW_MIN, WINDOW_MAX)
    volume = (volume - WINDOW_MIN) / (WINDOW_MAX - WINDOW_MIN)
    return volume.astype(np.float32)


def pad_to_patch(volume, patch_size, pad_value=0):
    pad_width = []
    for dim, patch_dim in zip(volume.shape, patch_size):
        missing = max(patch_dim - dim, 0)
        before = missing // 2
        after = missing - before
        pad_width.append((before, after))
    return np.pad(volume, pad_width, mode="constant", constant_values=pad_value)


def choose_patch_start(shape, patch_size, label, rng):
    use_foreground = rng.random() < FOREGROUND_PROBABILITY and np.any(label > 0)

    if use_foreground:
        present_classes = np.array(
            [cls for cls in range(1, NUM_CLASSES) if np.any(label == cls)]
        )
        target_class = present_classes[rng.integers(0, len(present_classes))]
        class_voxels = np.argwhere(label == target_class)
        center = class_voxels[rng.integers(0, len(class_voxels))]
    else:
        center = np.array([rng.integers(0, dim) for dim in shape])

    starts = []
    for center_coord, dim, patch_dim in zip(center, shape, patch_size):
        start = int(center_coord - patch_dim // 2)
        start = max(0, min(start, dim - patch_dim))
        starts.append(start)
    return tuple(starts)


def crop_patch(volume, start, patch_size):
    z, y, x = start
    d, h, w = patch_size
    return volume[z:z + d, y:y + h, x:x + w]


def save_patch(image_patch, label_patch, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        image=image_patch.astype(np.float32),
        label=label_patch.astype(np.uint8),
    )


def load_case(image_path, label_path):
    image = nib.load(str(image_path)).get_fdata(dtype=np.float32)
    label = np.asarray(nib.load(str(label_path)).dataobj, dtype=np.uint8)

    # Nibabel suele devolver [H, W, D]. Lo pasamos a [D, H, W].
    image = np.transpose(image, (2, 0, 1))
    label = np.transpose(label, (2, 0, 1))

    image = normalize_ct(image)
    image = pad_to_patch(image, PATCH_SIZE, pad_value=0)
    label = pad_to_patch(label, PATCH_SIZE, pad_value=0)
    return image, label


def convert_split(split_name, input_image_folder, input_label_folder, output_folder, patches_per_volume):
    image_dir = AMOS_ROOT / input_image_folder
    label_dir = AMOS_ROOT / input_label_folder
    output_dir = OUTPUT_ROOT / output_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_patch in output_dir.glob("*.npz"):
        old_patch.unlink()
    rng = np.random.default_rng(RANDOM_SEED)

    image_paths = sorted(image_dir.glob("amos_*.nii.gz"))
    if not image_paths:
        raise RuntimeError(f"No se encontraron volumenes en {image_dir}")

    for image_path in image_paths:
        label_path = label_dir / image_path.name
        if not label_path.exists():
            print(f"Sin etiqueta, salto: {image_path.name}")
            continue

        image, label = load_case(image_path, label_path)
        saved = 0

        for patch_idx in range(patches_per_volume):
            start = choose_patch_start(image.shape, PATCH_SIZE, label, rng)
            image_patch = crop_patch(image, start, PATCH_SIZE)
            label_patch = crop_patch(label, start, PATCH_SIZE)

            output_name = f"{image_path.name.replace('.nii.gz', '')}_patch{patch_idx:03d}.npz"
            save_patch(image_patch, label_patch, output_dir / output_name)
            saved += 1

        print(f"{split_name} {image_path.name}: {saved} parches guardados")


def main():
    print(f"Leyendo AMOS desde: {AMOS_ROOT}")
    print(f"Guardando parches 3D en: {OUTPUT_ROOT}")
    convert_split("train", "imagesTr", "labelsTr", "train", PATCHES_PER_TRAIN_VOLUME)
    convert_split("val", "imagesVa", "labelsVa", "val", PATCHES_PER_VAL_VOLUME)
    print("Conversion 3D terminada.")


if __name__ == "__main__":
    main()
